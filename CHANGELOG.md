# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - First stable release

### Added
- **Parallelization System**: User-configurable CPU cores via `--cpu` parameter
  - Default auto mode (0) uses 80% of available cores
  - Users can specify exact number of cores to use
  - Automatically caps at maximum available cores
  - Centralized multiprocessing in Analysis base class
- **Configurable Parameters**: Exposed 32+ adjustable parameters across all analyses via CLI
  - Segmentation: 8 parameters including conf, iou, row_tolerance, and visualization options
  - StarchArea: 3 parameters including starch_threshold, blur_kernel, mask_alpha
  - BlushColor: 8 parameters including fruit_threshold, colors, and text positioning
  - SuperficialScald: 6 parameters including morph_kernel, thresholds, and blur settings
  - PeelColor: 7 parameters including purple_threshold and lightness/green/yellow ranges
- **Confidence Threshold**: Added `--conf` parameter to segmentation analysis for detection sensitivity control
- **Developer's Guide**: Comprehensive developer documentation (2,694+ lines)
  - Step-by-step guide for adding new analysis modules
  - Step-by-step guide for adding new interfaces
  - Complete API reference for all core classes
  - Full working example of bruise detection analysis
- **User Documentation**:
  - Comprehensive adjustable parameters reference in `docs/users_guide/adjustable_parameters.rst`
  - Complete CLI parameter documentation for all 5 analyses
- **Test Coverage**: Extensive test suite additions
  - Tests for BlushColor analysis (0% → 72% coverage)
  - Tests for PeelColor analysis (0% → 43% coverage)
  - Tests for SuperficialScald analysis (0% → 48% coverage)
  - Tests for YoloModel and Scheduler
  - Comprehensive tests for Value types and Image classes
  - Total tests increased from 48 to 54
  - Overall coverage increased from 43% to 59% (+16%)

### Changed
- Refactored all child analyses to use new `_preRun()` and `_postRun()` pattern
- Renamed `_rateImageInstance()` to `_processImage()` for clarity across analyses
- Improved FileDirValue structure: moved directory creation to `setValue()` method
- Eliminated ~90 lines of duplicated parallelization code across analyses
- Cleaned up validation architecture across Value types

### Fixed
- **Critical**: Fixed FileDirValue validation bug that caused crashes during segmentation initialization
  - Directory is now created before validation instead of after
- **Critical**: Fixed FileNameValue circular validation logic preventing model name resolution
  - Value is set before validation to allow model names that aren't file paths yet
- Fixed zero-padding for fruit and tray numbering in segmentation output
  - Format changed from `_fruit_1, _fruit_10` to `_fruit_01, _fruit_10`
  - Ensures proper lexicographic sorting in CSV and file listings
- Fixed zero-padding for starch demo image numbering
- Fixed various test failures on dev branch

## [1.0a1] - Initial Alpha Release

### Added
- Initial release of Granny fruit quality assessment tool
- YOLO-based AI segmentation for pome fruits (apples and pears)
- Five core analyses:
  - Segmentation: Instance segmentation with YOLOv8
  - StarchArea: Starch content assessment
  - BlushColor: Blush color rating
  - PeelColor: Peel color classification
  - SuperficialScald: Superficial scald detection
- CLI and GUI interfaces
- Analysis dependency scheduler with DAG-based workflow
- Automatic model downloading from OSF
- Comprehensive Sphinx documentation
- Basic test suite

[Unreleased]: https://github.com/SystemsGenetics/granny/compare/v1.0a1...HEAD
[1.1.0]: https://github.com/SystemsGenetics/granny/compare/v1.0a1...dev
[1.0a1]: https://github.com/SystemsGenetics/granny/releases/tag/v1.0a1
