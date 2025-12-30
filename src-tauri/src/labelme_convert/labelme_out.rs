//! LabelMe to LabelMe conversion
//!
//! This module handles LabelMe → LabelMe conversion, which is useful for:
//! - Filtering labels (keeping only selected labels)
//! - Reordering labels (according to a predefined list)
//! - Removing embedded imageData to reduce file size
//! - Copying only annotated images to a new location
//! - Validating annotation format consistency
//! - Including background images (images without annotations)
//!
//! Unlike YOLO/COCO conversion, this does NOT split the dataset into
//! train/val/test sets.

use crate::labelme_convert::config::{ConversionConfig, LabelMeOutputFormat};
use crate::labelme_convert::detection::validate_shape_points;
use crate::labelme_convert::io::{
    copy_image, find_background_images, find_json_files, read_labelme_json, resolve_image_path,
    setup_labelme_directories, write_labelme_json,
};
use crate::labelme_convert::pipeline::{
    ConversionPipeline, FileType, OutputDirectories, ProcessedFileResult, ProcessingContext, Split,
};
use crate::labelme_convert::types::{
    ConversionResult, InputAnnotationFormat, InvalidAnnotation, Shape,
};
use std::collections::HashSet;
use std::path::Path;

/// LabelMe to LabelMe conversion pipeline
pub struct LabelMePipeline;

// ============================================================================
// Bounding Box Helper
// ============================================================================

/// Axis-aligned bounding box for coordinate calculations
#[derive(Debug, Clone, Copy)]
struct BoundingBox {
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
}

impl BoundingBox {
    /// Tolerance for floating point comparisons (1 pixel)
    const TOLERANCE: f64 = 1.0;

    /// Create bounding box from a slice of points
    fn from_points(points: &[(f64, f64)]) -> Option<Self> {
        if points.is_empty() {
            return None;
        }

        let (min_x, max_x, min_y, max_y) = points.iter().fold(
            (
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::INFINITY,
                f64::NEG_INFINITY,
            ),
            |(min_x, max_x, min_y, max_y), &(x, y)| {
                (min_x.min(x), max_x.max(x), min_y.min(y), max_y.max(y))
            },
        );

        Some(Self {
            min_x,
            min_y,
            max_x,
            max_y,
        })
    }

    /// Convert to 2-point representation (top-left, bottom-right)
    fn to_2point(&self) -> Vec<(f64, f64)> {
        vec![(self.min_x, self.min_y), (self.max_x, self.max_y)]
    }

    /// Convert to 4-point representation (clockwise from top-left)
    fn to_4point(&self) -> Vec<(f64, f64)> {
        vec![
            (self.min_x, self.min_y), // top-left
            (self.max_x, self.min_y), // top-right
            (self.max_x, self.max_y), // bottom-right
            (self.min_x, self.max_y), // bottom-left
        ]
    }

    /// Check if given points form an axis-aligned rectangle within this bbox
    fn is_axis_aligned_rectangle(&self, points: &[(f64, f64)]) -> bool {
        points.iter().all(|(x, y)| {
            let x_at_edge = (x - self.min_x).abs() < Self::TOLERANCE
                || (x - self.max_x).abs() < Self::TOLERANCE;
            let y_at_edge = (y - self.min_y).abs() < Self::TOLERANCE
                || (y - self.max_y).abs() < Self::TOLERANCE;
            x_at_edge && y_at_edge
        })
    }
}

// ============================================================================
// Shape Extension for Conversion
// ============================================================================

/// Extension trait for Shape conversion operations
trait ShapeExt {
    /// Create a new shape with different points and shape_type, preserving other fields
    fn with_new_geometry(&self, points: Vec<(f64, f64)>, shape_type: &str) -> Shape;
}

impl ShapeExt for Shape {
    fn with_new_geometry(&self, points: Vec<(f64, f64)>, shape_type: &str) -> Shape {
        Shape {
            label: self.label.clone(),
            points,
            group_id: self.group_id,
            shape_type: shape_type.to_string(),
            description: self.description.clone(),
            mask: self.mask.clone(),
            flags: self.flags.clone(),
        }
    }
}

// ============================================================================
// Shape Conversion Functions
// ============================================================================

/// Convert a 2-point rectangle to a 4-point polygon
fn convert_2point_to_4point(shape: &Shape) -> Option<Shape> {
    if shape.points.len() != 2 {
        return None;
    }
    let bbox = BoundingBox::from_points(&shape.points)?;
    Some(shape.with_new_geometry(bbox.to_4point(), "rectangle"))
}

/// Convert a 4-point polygon to a 2-point rectangle (strict: must be axis-aligned)
fn convert_4point_to_2point_strict(shape: &Shape) -> Option<Shape> {
    if shape.points.len() != 4 {
        return None;
    }
    let bbox = BoundingBox::from_points(&shape.points)?;
    if !bbox.is_axis_aligned_rectangle(&shape.points) {
        return None;
    }
    Some(shape.with_new_geometry(bbox.to_2point(), "rectangle"))
}

/// Convert any polygon (N >= 3 points) to a 2-point bounding box
fn convert_polygon_to_2point(shape: &Shape) -> Option<Shape> {
    if shape.points.len() < 3 {
        return None;
    }
    let bbox = BoundingBox::from_points(&shape.points)?;
    Some(shape.with_new_geometry(bbox.to_2point(), "rectangle"))
}

/// Convert any polygon (N >= 3 points) to a 4-point bounding box
fn convert_polygon_to_4point(shape: &Shape) -> Option<Shape> {
    if shape.points.len() < 3 {
        return None;
    }
    let bbox = BoundingBox::from_points(&shape.points)?;
    Some(shape.with_new_geometry(bbox.to_4point(), "rectangle"))
}

/// Check if a 4-point shape is an axis-aligned rectangle
fn is_axis_aligned_rectangle(shape: &Shape) -> bool {
    if shape.points.len() != 4 {
        return false;
    }
    BoundingBox::from_points(&shape.points)
        .map(|bbox| bbox.is_axis_aligned_rectangle(&shape.points))
        .unwrap_or(false)
}

/// Transform a shape based on the output format configuration
///
/// - Original: Keep shape as-is
/// - Bbox2Point: Convert all shapes to 2-point rectangles
/// - Bbox4Point: Convert all shapes to 4-point polygons (bbox corners)
fn transform_shape_for_output(shape: &Shape, output_format: LabelMeOutputFormat) -> Shape {
    match output_format {
        LabelMeOutputFormat::Original => shape.clone(),

        LabelMeOutputFormat::Bbox2Point => match shape.points.len() {
            2 => shape.clone(),
            4 => convert_4point_to_2point_strict(shape)
                .or_else(|| convert_polygon_to_2point(shape))
                .unwrap_or_else(|| shape.clone()),
            n if n >= 3 => convert_polygon_to_2point(shape).unwrap_or_else(|| shape.clone()),
            _ => shape.clone(),
        },

        LabelMeOutputFormat::Bbox4Point => match shape.points.len() {
            2 => convert_2point_to_4point(shape).unwrap_or_else(|| shape.clone()),
            4 if is_axis_aligned_rectangle(shape) => shape.clone(),
            n if n >= 3 => convert_polygon_to_4point(shape).unwrap_or_else(|| shape.clone()),
            _ => shape.clone(),
        },
    }
}

impl ConversionPipeline for LabelMePipeline {
    fn needs_split(&self) -> bool {
        false
    }

    fn setup_output_dirs(
        &self,
        config: &ConversionConfig,
    ) -> Result<Box<dyn OutputDirectories>, String> {
        let dirs = setup_labelme_directories(config)
            .map_err(|e| format!("Failed to create output directories: {}", e))?;
        Ok(Box::new(dirs))
    }

    fn process_file(
        &self,
        json_path: &Path,
        config: &ConversionConfig,
        output_dirs: &dyn OutputDirectories,
        context: &mut ProcessingContext,
    ) -> Result<ProcessedFileResult, String> {
        // Read the original LabelMe JSON
        let mut annotation = read_labelme_json(json_path)?;

        // Resolve image path
        let image_path = resolve_image_path(json_path, &annotation.image_path);
        let image_key = image_path.to_string_lossy().to_string();

        // Check for duplicate processing
        if context.is_image_processed(&image_key) {
            return Ok(ProcessedFileResult::default());
        }
        context.mark_image_processed(image_key);

        // Get JSON filename for error reporting
        let json_filename = json_path
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown.json".to_string());

        // Get input format from config
        let input_format = config
            .detected_input_format
            .unwrap_or(InputAnnotationFormat::Unknown);

        // Filter, validate, and transform shapes based on configuration
        let (filtered_shapes, skipped_count, invalid_annotations) = filter_and_validate_shapes(
            &annotation.shapes,
            &config.label_list,
            input_format,
            config.labelme_output_format,
            &json_filename,
            context,
        );

        // Update annotation with filtered shapes
        annotation.shapes = filtered_shapes;

        // Remove imageData if configured
        if config.remove_image_data {
            annotation.image_data = None;
        }

        let annotations_processed = annotation.shapes.len();

        // Check if this image became empty after label filtering
        // Note: We need to check original_shape_count vs annotations_processed
        let original_shape_count =
            skipped_count + annotations_processed + invalid_annotations.len();
        let is_filtered_empty =
            annotations_processed == 0 && original_shape_count > 0 && !config.label_list.is_empty();

        // If image became empty after filtering and user doesn't want to include empty images, skip it
        if is_filtered_empty && !config.include_background {
            return Ok(ProcessedFileResult {
                annotations_processed: 0,
                annotations_skipped: skipped_count,
                invalid_annotations,
                is_filtered_empty: true,
                filtered_empty_file_name: Some(json_filename),
            });
        }

        // Get output directory (no split for LabelMe)
        let output_dir = output_dirs.get_output_dir(Split::None, FileType::Annotation);

        // Write the new LabelMe JSON
        let output_json_path = output_dir.join(&json_filename);
        write_labelme_json(&output_json_path, &annotation)?;

        // Copy the image file (if it exists and imageData is not embedded)
        if image_path.exists() {
            copy_image(&image_path, output_dir)
                .map_err(|e| format!("Failed to copy image: {}", e))?;
        }

        Ok(ProcessedFileResult {
            annotations_processed,
            annotations_skipped: skipped_count,
            invalid_annotations,
            is_filtered_empty,
            filtered_empty_file_name: if is_filtered_empty {
                Some(json_filename)
            } else {
                None
            },
        })
    }

    fn finalize(
        &self,
        config: &ConversionConfig,
        output_dirs: &dyn OutputDirectories,
        context: &ProcessingContext,
    ) -> Result<(), String> {
        // Optionally create a labels.txt file listing all labels
        if !context.label_map.is_empty() {
            let labels_path = output_dirs.base_dir().join("labels.txt");

            // Sort labels by ID
            let mut sorted_labels: Vec<_> = context.label_map.iter().collect();
            sorted_labels.sort_by_key(|(_, id)| *id);

            let content = sorted_labels
                .iter()
                .map(|(label, _)| label.as_str())
                .collect::<Vec<_>>()
                .join("\n");

            std::fs::write(&labels_path, content)
                .map_err(|e| format!("Failed to write labels.txt: {}", e))?;
        }

        // Create a summary file
        let summary_path = output_dirs.base_dir().join("conversion_summary.txt");
        let summary = format!(
            "LabelMe Conversion Summary\n\
             ==========================\n\
             Source: {}\n\
             Files processed: {}\n\
             Total annotations: {}\n\
             Skipped annotations: {}\n\
             Labels: {}\n",
            config.input_dir.display(),
            context.stats.processed_files,
            context.stats.total_annotations,
            context.stats.skipped_annotations,
            context
                .label_map
                .keys()
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        );

        std::fs::write(&summary_path, summary)
            .map_err(|e| format!("Failed to write summary: {}", e))?;

        Ok(())
    }
}

/// Filter and validate shapes based on label list and input format
/// Also applies shape transformation based on output format configuration
///
/// Returns (filtered_shapes, skipped_count, invalid_annotations)
fn filter_and_validate_shapes(
    shapes: &[Shape],
    label_list: &[String],
    input_format: InputAnnotationFormat,
    output_format: LabelMeOutputFormat,
    file_name: &str,
    context: &mut ProcessingContext,
) -> (Vec<Shape>, usize, Vec<InvalidAnnotation>) {
    let mut filtered = Vec::new();
    let mut skipped = 0;
    let mut invalid_annotations = Vec::new();

    // Create a set for fast lookup if label_list is provided
    let allowed_labels: Option<HashSet<&str>> = if label_list.is_empty() {
        None
    } else {
        Some(label_list.iter().map(|s| s.as_str()).collect())
    };

    for shape in shapes {
        // Check label filter first
        let label_allowed = match &allowed_labels {
            Some(allowed) => allowed.contains(shape.label.as_str()),
            None => true, // No filter, allow all labels
        };

        if !label_allowed {
            context.add_skipped_label(&shape.label);
            skipped += 1;
            continue;
        }

        // Validate points count based on detected input format
        if let Err(reason) = validate_shape_points(shape, input_format) {
            invalid_annotations.push(InvalidAnnotation {
                file: file_name.to_string(),
                label: shape.label.clone(),
                reason: reason.as_str(),
                shape_type: shape.shape_type.clone(),
                points_count: shape.points.len(),
            });
            skipped += 1;
            continue;
        }

        // Transform shape based on output format
        let transformed_shape = transform_shape_for_output(shape, output_format);

        // Shape passed all checks, add it
        context.ensure_label(&transformed_shape.label);
        filtered.push(transformed_shape);
    }

    (filtered, skipped, invalid_annotations)
}

/// Main conversion function for LabelMe → LabelMe
pub fn convert_to_labelme(config: &ConversionConfig) -> ConversionResult {
    // Validate configuration
    if let Err(e) = config.validate() {
        return ConversionResult::failure(vec![e]);
    }

    let pipeline = LabelMePipeline;

    // Set up output directories
    let output_dirs = match pipeline.setup_output_dirs(config) {
        Ok(dirs) => dirs,
        Err(e) => return ConversionResult::failure(vec![e]),
    };

    // Initialize processing context
    let mut context = if config.label_list.is_empty() {
        ProcessingContext::new()
    } else {
        ProcessingContext::with_labels(&config.label_list)
    };

    // Find all JSON files
    let json_files = find_json_files(&config.input_dir);
    context.stats.total_files = json_files.len();

    // Process each JSON file
    for json_path in &json_files {
        match pipeline.process_file(json_path, config, output_dirs.as_ref(), &mut context) {
            Ok(result) => {
                context.stats.increment_processed();
                context.stats.add_annotations(result.annotations_processed);
                context
                    .stats
                    .add_skipped_annotations(result.annotations_skipped);
                // Add invalid annotations to stats
                for invalid in result.invalid_annotations {
                    context.stats.add_invalid_annotation(invalid);
                }
                // Track filtered empty images
                if result.is_filtered_empty {
                    if let Some(file_name) = result.filtered_empty_file_name {
                        context.stats.add_filtered_empty_file(file_name);
                    }
                }
            }
            Err(e) => {
                context.stats.increment_failed();
                context.add_error(format!("{}: {}", json_path.display(), e));
            }
        }
    }

    // Process background images if enabled
    if config.include_background {
        let bg_files =
            process_background_images(config, output_dirs.as_ref(), &context.processed_images);
        for file_name in bg_files {
            context.stats.add_background_file(file_name);
        }
    }

    // Update stats with labels
    for label in context.label_map.keys() {
        context.stats.add_label(label.clone());
    }

    // Add skipped labels to stats
    for label in &context.skipped_labels {
        context.stats.add_skipped_label(label.clone());
    }

    // Finalize
    if let Err(e) = pipeline.finalize(config, output_dirs.as_ref(), &context) {
        context.add_error(e);
    }

    if context.errors.is_empty() {
        ConversionResult::success(
            output_dirs.base_dir().to_string_lossy().to_string(),
            context.stats,
        )
    } else {
        let mut result = ConversionResult::success(
            output_dirs.base_dir().to_string_lossy().to_string(),
            context.stats,
        );
        result.errors = context.errors;
        result
    }
}

/// Process background images (images without annotations)
/// Returns the list of background image file names
fn process_background_images(
    config: &ConversionConfig,
    output_dirs: &dyn OutputDirectories,
    processed_images: &HashSet<String>,
) -> Vec<String> {
    let bg_images = find_background_images(&config.input_dir, processed_images);
    let mut bg_files = Vec::new();

    let output_dir = output_dirs.get_output_dir(Split::None, FileType::Image);

    for image_path in bg_images {
        // Copy image
        if let Err(e) = copy_image(&image_path, output_dir) {
            eprintln!(
                "Failed to copy background image {}: {}",
                image_path.display(),
                e
            );
            continue;
        }

        // Get file name for reporting
        let file_name = image_path
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_string());
        bg_files.push(file_name);
    }

    bg_files
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::labelme_convert::types::InputAnnotationFormat;

    fn create_test_shape(label: &str, points_count: usize) -> Shape {
        Shape {
            label: label.to_string(),
            points: (0..points_count).map(|i| (i as f64, i as f64)).collect(),
            group_id: None,
            shape_type: "polygon".to_string(),
            description: None,
            mask: None,
            flags: None,
        }
    }

    #[test]
    fn test_filter_and_validate_shapes_empty_list() {
        let shapes = vec![create_test_shape("cat", 4), create_test_shape("dog", 4)];

        let mut context = ProcessingContext::new();
        let (filtered, skipped, invalid) = filter_and_validate_shapes(
            &shapes,
            &[],
            InputAnnotationFormat::Bbox4Point,
            LabelMeOutputFormat::Original,
            "test.json",
            &mut context,
        );

        assert_eq!(filtered.len(), 2);
        assert_eq!(skipped, 0);
        assert_eq!(invalid.len(), 0);
        assert!(context.label_map.contains_key("cat"));
        assert!(context.label_map.contains_key("dog"));
    }

    #[test]
    fn test_filter_and_validate_shapes_with_label_filter() {
        let shapes = vec![create_test_shape("cat", 4), create_test_shape("dog", 4)];

        let label_list = vec!["cat".to_string()];
        let mut context = ProcessingContext::new();
        let (filtered, skipped, invalid) = filter_and_validate_shapes(
            &shapes,
            &label_list,
            InputAnnotationFormat::Bbox4Point,
            LabelMeOutputFormat::Original,
            "test.json",
            &mut context,
        );

        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].label, "cat");
        assert_eq!(skipped, 1);
        assert_eq!(invalid.len(), 0);
        assert!(context.label_map.contains_key("cat"));
        assert!(!context.label_map.contains_key("dog"));
        assert!(context.skipped_labels.contains("dog"));
    }

    #[test]
    fn test_filter_and_validate_shapes_with_invalid_points() {
        let shapes = vec![
            create_test_shape("cat", 4), // Valid for Bbox4Point
            create_test_shape("dog", 2), // Invalid for Bbox4Point (should be 4)
        ];

        let mut context = ProcessingContext::new();
        let (filtered, skipped, invalid) = filter_and_validate_shapes(
            &shapes,
            &[],
            InputAnnotationFormat::Bbox4Point,
            LabelMeOutputFormat::Original,
            "test.json",
            &mut context,
        );

        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].label, "cat");
        assert_eq!(skipped, 1);
        assert_eq!(invalid.len(), 1);
        assert_eq!(invalid[0].label, "dog");
        assert_eq!(invalid[0].points_count, 2);
    }

    #[test]
    fn test_pipeline_needs_split() {
        let pipeline = LabelMePipeline;
        assert!(!pipeline.needs_split());
    }

    // ========================================================================
    // Shape Transformation Tests
    // ========================================================================

    fn create_rectangle_shape(label: &str, x1: f64, y1: f64, x2: f64, y2: f64) -> Shape {
        Shape {
            label: label.to_string(),
            points: vec![(x1, y1), (x2, y2)],
            group_id: None,
            shape_type: "rectangle".to_string(),
            description: None,
            mask: None,
            flags: None,
        }
    }

    fn create_polygon_4point_shape(label: &str, points: Vec<(f64, f64)>) -> Shape {
        Shape {
            label: label.to_string(),
            points,
            group_id: None,
            shape_type: "polygon".to_string(),
            description: None,
            mask: None,
            flags: None,
        }
    }

    #[test]
    fn test_bbox_2point_to_4point_conversion() {
        // Create a 2-point rectangle
        let rect = create_rectangle_shape("cat", 10.0, 20.0, 100.0, 80.0);

        // Convert to 4-point bbox
        let polygon = convert_2point_to_4point(&rect).expect("Should convert successfully");

        // Verify 4 points
        assert_eq!(polygon.points.len(), 4);
        assert_eq!(polygon.shape_type, "polygon");
        assert_eq!(polygon.label, "cat");

        // Verify correct corners (clockwise from top-left)
        assert_eq!(polygon.points[0], (10.0, 20.0)); // top-left
        assert_eq!(polygon.points[1], (100.0, 20.0)); // top-right
        assert_eq!(polygon.points[2], (100.0, 80.0)); // bottom-right
        assert_eq!(polygon.points[3], (10.0, 80.0)); // bottom-left
    }

    #[test]
    fn test_bbox_2point_to_4point_inverted_coords() {
        // Create a rectangle with inverted coordinates (x2 < x1, y2 < y1)
        let rect = create_rectangle_shape("dog", 100.0, 80.0, 10.0, 20.0);

        let polygon = convert_2point_to_4point(&rect).expect("Should convert successfully");

        // Should normalize to proper order
        assert_eq!(polygon.points[0], (10.0, 20.0)); // top-left
        assert_eq!(polygon.points[1], (100.0, 20.0)); // top-right
        assert_eq!(polygon.points[2], (100.0, 80.0)); // bottom-right
        assert_eq!(polygon.points[3], (10.0, 80.0)); // bottom-left
    }

    #[test]
    fn test_bbox_4point_to_2point_conversion() {
        // Create a 4-point polygon that forms a rectangle
        let polygon = create_polygon_4point_shape(
            "bird",
            vec![
                (10.0, 20.0),  // top-left
                (100.0, 20.0), // top-right
                (100.0, 80.0), // bottom-right
                (10.0, 80.0),  // bottom-left
            ],
        );

        let rect = convert_4point_to_2point_strict(&polygon).expect("Should convert successfully");

        assert_eq!(rect.points.len(), 2);
        assert_eq!(rect.shape_type, "rectangle");
        assert_eq!(rect.label, "bird");

        // Verify bbox corners
        assert_eq!(rect.points[0], (10.0, 20.0)); // top-left
        assert_eq!(rect.points[1], (100.0, 80.0)); // bottom-right
    }

    #[test]
    fn test_polygon_4point_non_rectangular_fallback() {
        // Create a 4-point polygon that is NOT a rectangle (parallelogram)
        let polygon = create_polygon_4point_shape(
            "fish",
            vec![
                (10.0, 20.0),
                (110.0, 30.0), // Not aligned with y axis
                (100.0, 80.0),
                (0.0, 70.0),
            ],
        );

        // Should return None for strict rectangle check
        let result = convert_4point_to_2point_strict(&polygon);
        assert!(result.is_none());

        // But convert_polygon_to_2point should work (computes bounding box)
        let bbox = convert_polygon_to_2point(&polygon).expect("Should compute bounding box");
        assert_eq!(bbox.points.len(), 2);
        assert_eq!(bbox.points[0], (0.0, 20.0)); // min_x, min_y
        assert_eq!(bbox.points[1], (110.0, 80.0)); // max_x, max_y
    }

    #[test]
    fn test_transform_shape_original_keeps_unchanged() {
        let rect = create_rectangle_shape("cat", 10.0, 20.0, 100.0, 80.0);

        let transformed = transform_shape_for_output(&rect, LabelMeOutputFormat::Original);

        assert_eq!(transformed.points.len(), 2);
        assert_eq!(transformed.shape_type, "rectangle");
    }

    #[test]
    fn test_transform_shape_to_bbox_4point() {
        let rect = create_rectangle_shape("cat", 10.0, 20.0, 100.0, 80.0);

        let transformed = transform_shape_for_output(&rect, LabelMeOutputFormat::Bbox4Point);

        assert_eq!(transformed.points.len(), 4);
        assert_eq!(transformed.shape_type, "polygon");
    }

    #[test]
    fn test_transform_shape_to_bbox_2point() {
        let polygon = create_polygon_4point_shape(
            "dog",
            vec![(10.0, 20.0), (100.0, 20.0), (100.0, 80.0), (10.0, 80.0)],
        );

        let transformed = transform_shape_for_output(&polygon, LabelMeOutputFormat::Bbox2Point);

        assert_eq!(transformed.points.len(), 2);
        assert_eq!(transformed.shape_type, "rectangle");
    }

    #[test]
    fn test_filter_shapes_with_bbox_4point_output() {
        // Create 2-point rectangles
        let shapes = vec![
            create_rectangle_shape("cat", 10.0, 20.0, 100.0, 80.0),
            create_rectangle_shape("dog", 50.0, 60.0, 150.0, 160.0),
        ];

        let mut context = ProcessingContext::new();
        let (filtered, skipped, invalid) = filter_and_validate_shapes(
            &shapes,
            &[],
            InputAnnotationFormat::Bbox2Point,
            LabelMeOutputFormat::Bbox4Point,
            "test.json",
            &mut context,
        );

        // Both shapes should be converted to 4-point polygons
        assert_eq!(filtered.len(), 2);
        assert_eq!(skipped, 0);
        assert_eq!(invalid.len(), 0);

        for shape in &filtered {
            assert_eq!(shape.points.len(), 4);
            assert_eq!(shape.shape_type, "polygon");
        }
    }

    #[test]
    fn test_filter_shapes_with_bbox_2point_output() {
        // Create 4-point polygons that are rectangles
        let shapes = vec![
            create_polygon_4point_shape(
                "cat",
                vec![(10.0, 20.0), (100.0, 20.0), (100.0, 80.0), (10.0, 80.0)],
            ),
            create_polygon_4point_shape(
                "dog",
                vec![(50.0, 60.0), (150.0, 60.0), (150.0, 160.0), (50.0, 160.0)],
            ),
        ];

        let mut context = ProcessingContext::new();
        let (filtered, skipped, invalid) = filter_and_validate_shapes(
            &shapes,
            &[],
            InputAnnotationFormat::Bbox4Point,
            LabelMeOutputFormat::Bbox2Point,
            "test.json",
            &mut context,
        );

        // Both shapes should be converted to 2-point rectangles
        assert_eq!(filtered.len(), 2);
        assert_eq!(skipped, 0);
        assert_eq!(invalid.len(), 0);

        for shape in &filtered {
            assert_eq!(shape.points.len(), 2);
            assert_eq!(shape.shape_type, "rectangle");
        }
    }

    // ========================================================================
    // Polygon to Bbox Tests
    // ========================================================================

    fn create_polygon_shape(label: &str, points: Vec<(f64, f64)>) -> Shape {
        Shape {
            label: label.to_string(),
            points,
            group_id: None,
            shape_type: "polygon".to_string(),
            description: None,
            mask: None,
            flags: None,
        }
    }

    #[test]
    fn test_polygon_to_bbox_2point() {
        // Create a 5-point polygon (pentagon-ish)
        let polygon = create_polygon_shape(
            "cat",
            vec![
                (50.0, 10.0), // top
                (90.0, 40.0), // right
                (70.0, 90.0), // bottom-right
                (30.0, 90.0), // bottom-left
                (10.0, 40.0), // left
            ],
        );

        let bbox = convert_polygon_to_2point(&polygon).expect("Should compute bounding box");

        assert_eq!(bbox.points.len(), 2);
        assert_eq!(bbox.shape_type, "rectangle");
        assert_eq!(bbox.points[0], (10.0, 10.0)); // min_x, min_y
        assert_eq!(bbox.points[1], (90.0, 90.0)); // max_x, max_y
    }

    #[test]
    fn test_polygon_to_bbox_4point() {
        // Create a 5-point polygon (pentagon-ish)
        let polygon = create_polygon_shape(
            "dog",
            vec![
                (50.0, 10.0), // top
                (90.0, 40.0), // right
                (70.0, 90.0), // bottom-right
                (30.0, 90.0), // bottom-left
                (10.0, 40.0), // left
            ],
        );

        let bbox = convert_polygon_to_4point(&polygon).expect("Should compute bounding box");

        assert_eq!(bbox.points.len(), 4);
        assert_eq!(bbox.shape_type, "polygon");
        // Clockwise from top-left
        assert_eq!(bbox.points[0], (10.0, 10.0)); // top-left
        assert_eq!(bbox.points[1], (90.0, 10.0)); // top-right
        assert_eq!(bbox.points[2], (90.0, 90.0)); // bottom-right
        assert_eq!(bbox.points[3], (10.0, 90.0)); // bottom-left
    }

    #[test]
    fn test_transform_polygon_to_bbox_2point() {
        // Create a 6-point polygon
        let polygon = create_polygon_shape(
            "fish",
            vec![
                (20.0, 10.0),
                (80.0, 10.0),
                (100.0, 50.0),
                (80.0, 90.0),
                (20.0, 90.0),
                (0.0, 50.0),
            ],
        );

        let transformed = transform_shape_for_output(&polygon, LabelMeOutputFormat::Bbox2Point);

        assert_eq!(transformed.points.len(), 2);
        assert_eq!(transformed.shape_type, "rectangle");
        assert_eq!(transformed.points[0], (0.0, 10.0));
        assert_eq!(transformed.points[1], (100.0, 90.0));
    }

    #[test]
    fn test_transform_polygon_to_bbox_4point() {
        // Create a 6-point polygon
        let polygon = create_polygon_shape(
            "bird",
            vec![
                (20.0, 10.0),
                (80.0, 10.0),
                (100.0, 50.0),
                (80.0, 90.0),
                (20.0, 90.0),
                (0.0, 50.0),
            ],
        );

        let transformed = transform_shape_for_output(&polygon, LabelMeOutputFormat::Bbox4Point);

        assert_eq!(transformed.points.len(), 4);
        assert_eq!(transformed.shape_type, "polygon");
    }
}
