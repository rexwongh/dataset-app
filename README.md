# Dataset App

A desktop application for exploring and processing image datasets annotated in LabelMe, with tools for visualization, conversion to YOLO, label extraction, basic auto-annotation, and crop/remap utilities. Built with SvelteKit (frontend) and Tauri v2 (Rust backend).

**Version**: 0.1.5

## Status and Caution
- This project is under rapid development.
- APIs, commands, and UI may change frequently and introduce breaking changes.

## Features
- Dataset browsing with pagination and lazy previews
- Image details and on-demand metadata loading
- LabelMe dataset summary (label counts, annotation types, totals)
- Export to YOLO (train/val/test splits, shape type, specific labels)
- Extract specific labels from LabelMe datasets
- Convert YOLO → LabelMe
- Draw bounding boxes or polygons into rendered images
- Visualization-only scan without saving outputs
- Auto-annotate skeleton (backend hook for extensions)
- Crop and remap annotations around a parent label with padding

## Getting Started

### Prerequisites
- Node.js & Yarn
- Rust toolchain
- **Windows**: LLVM (for OpenCV bindings) + OpenCV
- **macOS/Linux**: OpenCV (via Homebrew/apt)

**⚠️ Platform-specific notes**:
- **Windows**: If you encounter compilation errors or zombie processes, see [TROUBLESHOOTING_WINDOWS.md](.vscode/.tmp/TROUBLESHOOTING_WINDOWS.md)
- **All platforms**:
  - 🚑 **Quick fix guide**: [QUICK_FIX.md](docs/QUICK_FIX.md) ← Start here if your system is stuck
  - 📚 **Detailed cleanup guide**: [ZOMBIE_CLEANUP.md](docs/ZOMBIE_CLEANUP.md)

### Install dependencies and build
```bash
yarn install

# Standard start
yarn tauri dev

# If you had previous crashes or zombie processes
yarn tauri:clean
```

### Clean up zombie processes (All platforms)
If your CPU/RAM usage is high after closing the app:
```bash
# Standard clean (recommended for daily use)
yarn clean

# Aggressive clean (if zombies keep reappearing)
# ⚠️ This will also kill rust-analyzer, requiring a VSCode window reload
yarn clean:hard
```

**Common scenario**: If zombies keep reappearing, you likely have multiple editors open (VSCode, Cursor, etc.) or `rust-analyzer` is stuck. Use `yarn clean:hard` then reload your editor.

### Proper shutdown procedure
- ✅ **Recommended**: Press `Ctrl+C` in the terminal running `yarn tauri dev`
- ❌ **Avoid**: Clicking the X button on the app window (may leave zombie build processes)

**Why?** Clicking X only closes the GUI, leaving `cargo build` and its child processes (`build-script-build`) running in the background, consuming memory.
This will:
- Start SvelteKit dev server on `http://localhost:1420`
- Launch the Tauri shell pointing at the dev server

### Build Production Bundle
```bash
yarn tauri build
```
This will:
- Build SvelteKit to `build/`
- Produce platform bundles via Tauri (see `src-tauri/tauri.conf.json`)
- For more details on building and bundling, see the Tauri 2 docs: [https://v2.tauri.app/](https://v2.tauri.app/)



## Project Structure
```
.
├─ build/                       # SvelteKit output
├─ docs/                        # Design and analysis docs
├─ src/                         # Frontend (SvelteKit)
│  ├─ app.html, app.css
│  ├─ lib/                      # Components and services
│  │  ├─ services/datasetService.ts
│  │  └─ ... Svelte components
│  ├─ routes/                   # Pages (+page.svelte / +page.ts)
│  └─ funcs/                    # Plain JS helpers
├─ src-tauri/                   # Backend (Rust, Tauri v2)
│  ├─ src/main.rs               # Tauri commands and wiring
│  ├─ src/*.rs                  # Modules: handlers, converters, drawers
│  ├─ Cargo.toml                # Rust dependencies
│  └─ tauri.conf.json           # Tauri app config
├─ static/                      # Static assets
```

## Key Commands (Rust Backend)
**If you're interested in the Rust backend architecture and Tauri command details, please refer to [`docs/RUST_BACKEND.md`](docs/RUST_BACKEND.md) for comprehensive documentation including hierarchical code structure, function signatures, and implementation details.**

See `src-tauri/src/main.rs` for the full list and signatures.

