# GitHub Copilot Instructions for Auto-Cropper

## Project Overview
Auto-Cropper is a CLI tool that automatically crops videos to follow a person using YOLOv8 object detection. The tool maintains a 16:9 aspect ratio and applies smoothing to reduce jitter.

## Architecture & Design Patterns

### Core Components
- **CLI Interface** (`auto_cropper/cli.py`): Click-based command line interface
- **Person Detection** (`auto_cropper/detector.py`): YOLOv8-based person detection and tracking
- **Video Cropping** (`auto_cropper/video_cropper.py`): Video processing and cropping logic
- **Error Handling**: Comprehensive validation with user-friendly error messages

### Processing Pipeline
1. **Detection**: Analyze video frames to detect people using YOLOv8
2. **Tracking**: Select a specific person to track across frames
3. **Cropping**: Crop video to follow the tracked person with smoothing

## Code Style & Conventions

### General Principles
- **Type Hints**: Always use type hints for function parameters and return values
- **Docstrings**: Use comprehensive docstrings for all public functions and classes
- **Error Handling**: Provide clear, actionable error messages without emojis
- **Testing**: Write comprehensive unit tests for all functionality
- **Running Tests**: Always run tests before committing changes

### Naming Conventions
- **Functions**: Use snake_case (e.g., `detect_people_in_video`)
- **Classes**: Use PascalCase (e.g., `PersonDetector`, `VideoCropper`)
- **Constants**: Use UPPER_CASE (e.g., `DEFAULT_CONFIDENCE`)
- **Private methods**: Prefix with underscore (e.g., `_calculate_crop_position`)

### File Organization
```
auto_cropper/
├── cli.py              # Command line interface
├── detector.py         # Person detection and tracking
├── video_cropper.py    # Video processing and cropping
└── __init__.py         # Package initialization
```

## Dependencies & Versions

### Core Dependencies (Pinned)
- `click==8.2.1` - CLI framework
- `opencv-python==4.11.0.86` - Video processing
- `ultralytics==8.3.161` - YOLOv8 object detection
- `torch==2.7.1` - PyTorch for ML models
- `numpy==2.3.1` - Numerical operations

### Development Dependencies
- `pytest==8.4.1` - Testing framework
- `black==25.1.0` - Code formatting
- `mypy==1.16.1` - Type checking

## Error Handling Guidelines

### Validation Functions
Use the standardized validation functions in `cli.py`:
- `validate_input_file()` - Validate video file inputs
- `ensure_output_directory()` - Create and validate output directories
- `validate_json_file()` - Validate JSON detection/tracking files

### Error Message Format
- Use plain text (no emojis)
- Provide specific, actionable error messages
- Include suggestions for fixing issues
- Use `click.ClickException` for CLI errors

Example:
```python
if not path.exists():
    raise click.ClickException(f"Input file does not exist: {file_path}")
```

## Testing Guidelines

### Test Structure
- Unit tests in `tests/test_cropper.py` for core functionality
- Error handling tests in `tests/test_cli_error_handling.py`
- Use pytest fixtures for reusable test data
- Mock heavy operations (YOLO model loading, video processing)

### Test Data
- Create temporary files using `tmp_path` fixtures
- Use OpenCV to generate sample videos for testing
- Create structured JSON data for detection/tracking tests

## CLI Design Patterns

### Command Structure
```bash
auto-cropper <command> [options]
```

Commands:
- `detect` - Detect people in video frames
- `track` - Select person to track from detection data
- `crop` - Crop video based on tracking data
- `process` - Complete pipeline in one command
- `summary` - Show detection statistics

### Option Patterns
- Use short (`-o`) and long (`--output-dir`) option forms
- Provide sensible defaults for all options
- Use `click.Path()` with custom validation for file inputs

## Model & Performance

### YOLO Model Usage
- Default model: `yolov8l.pt` (best accuracy/performance balance)
- Models auto-download on first use
- All `.pt` files are gitignored to avoid repository bloat

### Performance Considerations
- Process videos frame-by-frame for memory efficiency
- Apply smoothing to reduce jitter in cropped output
- Provide progress indicators for long-running operations

## Project Policies

### Contributing Policy
- This is a fork-only repository
- Pull requests are not accepted
- Users should fork for modifications
- Bug reports welcome via GitHub issues

### Dependencies
- All dependencies are pinned for reproducibility
- Comprehensive license review completed (all permissive licenses)
- No GPL-licensed dependencies

## Common Patterns

### File Processing
```python
def process_video(video_path: str, output_dir: str) -> str:
    """Process video with comprehensive error handling."""
    validated_path = validate_input_file(video_path)
    validated_output = ensure_output_directory(output_dir)
    # ... processing logic
    return output_file_path
```

### Progress Reporting
```python
if self.verbose:
    pbar = tqdm(total=total_frames, desc="Processing frames")
    # ... use pbar.update(1) in loop
```

When suggesting code changes or new features, please follow these patterns and conventions to maintain consistency with the existing codebase.
