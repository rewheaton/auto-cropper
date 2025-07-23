# Batch Processing Implementation Plan

## Overview
Add a new optional argument `--input-directory` to the `process` command that allows processing multiple video files in a directory automatically.

## Requirements
- Add `--input-directory` option to the process command
- When present, process all compatible video files in the specified directory
- Maintain all existing functionality when processing individual files
- Provide clear progress feedback for batch operations
- Handle errors gracefully (skip failed files, continue with others)
- Generate organized output structure for multiple files

## Implementation Steps

### 1. Add Helper Function for Directory Processing
**Location**: `auto_cropper/cli.py` (before the CLI group definition)

```python
def find_video_files(directory_path: str, allowed_extensions: Optional[List[str]] = None) -> List[Path]:
    """
    Find all compatible video files in a directory.
    
    Args:
        directory_path: Path to the directory to search
        allowed_extensions: List of allowed file extensions (e.g., ['.mp4', '.avi'])
    
    Returns:
        List of Path objects for compatible video files
        
    Raises:
        click.ClickException: If directory is invalid
    """
    if allowed_extensions is None:
        allowed_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.m4v', '.wmv']
    
    path = Path(directory_path)
    
    # Check if directory exists
    if not path.exists():
        raise click.ClickException(f"Directory does not exist: {directory_path}")
    
    # Check if it's a directory
    if not path.is_dir():
        raise click.ClickException(f"Path is not a directory: {directory_path}")
    
    # Find all video files
    video_files = []
    allowed_extensions_lower = [ext.lower() for ext in allowed_extensions]
    
    for file_path in path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in allowed_extensions_lower:
            video_files.append(file_path)
    
    return sorted(video_files)  # Sort for consistent ordering
```

### 2. Modify Process Command Signature
**Location**: `auto_cropper/cli.py` - process command decorator

**Current**:
```python
@main.command()
@click.argument('video_path', type=click.Path())
@click.option(...)
def process(ctx, video_path: str, ...):
```

**New**:
```python
@main.command()
@click.argument('video_path', type=click.Path(), required=False)
@click.option(
    '--input-directory', '-id',
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help='Process all compatible video files in this directory'
)
@click.option(...)
def process(ctx, video_path: Optional[str], input_directory: Optional[str], ...):
```

### 3. Add Input Validation Logic
**Location**: Process command function - beginning of function

```python
def process(ctx, video_path: Optional[str], input_directory: Optional[str], ...):
    verbose = ctx.obj['verbose']
    
    # Validate that exactly one input method is provided
    if not video_path and not input_directory:
        raise click.ClickException("Must provide either VIDEO_PATH or --input-directory")
    
    if video_path and input_directory:
        raise click.ClickException("Cannot specify both VIDEO_PATH and --input-directory")
    
    # Validate output directory
    validated_output_dir = ensure_output_directory(output_dir)
    
    if input_directory:
        # Batch processing mode
        video_files = find_video_files(input_directory)
        if not video_files:
            raise click.ClickException(f"No compatible video files found in directory: {input_directory}")
        
        process_batch(video_files, validated_output_dir, confidence, margin, smoothing, duration, verbose)
    else:
        # Single file processing mode (existing logic)
        validated_video_path = validate_input_file(video_path)
        process_single_file(validated_video_path, validated_output_dir, confidence, margin, smoothing, duration, verbose)
```

### 4. Create Batch Processing Function
**Location**: `auto_cropper/cli.py` (before the process command)

```python
def process_batch(video_files: List[Path], output_dir: Path, confidence: float, 
                 margin: int, smoothing: int, duration: Optional[int], verbose: bool):
    """
    Process multiple video files in batch.
    
    Args:
        video_files: List of video file paths to process
        output_dir: Output directory for all processing results
        confidence: Detection confidence threshold
        margin: Margin around person in pixels
        smoothing: Smoothing window size
        duration: Optional duration limit in seconds
        verbose: Enable verbose output
    """
    total_files = len(video_files)
    successful_files = []
    failed_files = []
    
    click.echo(f"Starting batch processing of {total_files} video files...")
    click.echo("=" * 60)
    
    for i, video_file in enumerate(video_files, 1):
        click.echo(f"\n[{i}/{total_files}] Processing: {video_file.name}")
        click.echo("-" * 40)
        
        try:
            # Create subdirectory for this video's output
            video_output_dir = output_dir / video_file.stem
            video_output_dir.mkdir(exist_ok=True)
            
            # Process this video
            process_single_file(video_file, video_output_dir, confidence, margin, smoothing, duration, verbose)
            successful_files.append(video_file)
            click.echo(f"✓ Successfully processed: {video_file.name}")
            
        except Exception as e:
            failed_files.append((video_file, str(e)))
            click.echo(f"✗ Failed to process {video_file.name}: {e}", err=True)
            if verbose:
                import traceback
                traceback.print_exc()
            continue
    
    # Summary
    click.echo("\n" + "=" * 60)
    click.echo("BATCH PROCESSING SUMMARY")
    click.echo("=" * 60)
    click.echo(f"Total files: {total_files}")
    click.echo(f"Successful: {len(successful_files)}")
    click.echo(f"Failed: {len(failed_files)}")
    
    if successful_files:
        click.echo(f"\nSuccessfully processed files:")
        for file_path in successful_files:
            click.echo(f"  ✓ {file_path.name}")
    
    if failed_files:
        click.echo(f"\nFailed files:")
        for file_path, error in failed_files:
            click.echo(f"  ✗ {file_path.name}: {error}")
    
    # Batch cleanup option
    if successful_files and click.confirm("\nDelete all intermediate files from batch processing?"):
        cleanup_count = 0
        for video_file in successful_files:
            video_output_dir = output_dir / video_file.stem
            for json_file in video_output_dir.glob("*.json"):
                json_file.unlink()
                cleanup_count += 1
        click.echo(f"Deleted {cleanup_count} intermediate files")
```

### 5. Extract Single File Processing Logic
**Location**: `auto_cropper/cli.py` (before the process command)

```python
def process_single_file(video_path: Path, output_dir: Path, confidence: float, 
                       margin: int, smoothing: int, duration: Optional[int], verbose: bool):
    """
    Process a single video file through the complete pipeline.
    
    Args:
        video_path: Path to the video file
        output_dir: Output directory for processing results
        confidence: Detection confidence threshold
        margin: Margin around person in pixels
        smoothing: Smoothing window size
        duration: Optional duration limit in seconds
        verbose: Enable verbose output
    """
    # Step 1: Detection
    click.echo("Step 1: Detecting people in video...")
    detector = PersonDetector(model_name='yolov8l.pt', confidence=confidence, verbose=verbose)
    detection_file = detector.detect_people_in_video(str(video_path), str(output_dir))
    
    # Show detection summary
    summary = detector.get_detection_summary(detection_file)
    click.echo(f"Detection complete! Coverage: {summary['detection_coverage']}%")
    
    # Step 2: Tracking
    click.echo("Step 2: Selecting person to track...")
    tracker = PersonTracker(verbose=verbose)
    tracking_file = tracker.select_person_to_track(detection_file)
    click.echo("Tracking complete!")
    
    # Step 3: Cropping
    click.echo("Step 3: Cropping video...")
    cropper = VideoCropper(margin=margin, smoothing_window=smoothing, verbose=verbose)
    output_path = cropper.crop_video(str(video_path), tracking_file, output_dir=str(output_dir), duration_limit=duration)
    click.echo("Video cropping complete!")
    
    click.echo(f"Final video saved to: {output_path}")
    
    return detection_file, tracking_file, output_path
```

### 6. Update Process Command Function
**Location**: `auto_cropper/cli.py` - process command function body

Replace the existing function body with the validation logic from step 3, which calls either `process_batch()` or the existing single-file logic.

### 7. Update Documentation
**Location**: Process command docstring

```python
def process(ctx, video_path: Optional[str], input_directory: Optional[str], ...):
    """
    Complete pipeline: detect, track, and crop in one command.
    
    Process either a single video file or all compatible video files in a directory.
    
    VIDEO_PATH: Path to the input video file (required unless --input-directory is used)
    """
```

## Output Structure

### Single File Mode (existing)
```
./output/
├── video_name_detections.json
├── video_name_tracking.json
└── video_name_cropped.mp4
```

### Batch Mode (new)
```
./output/
├── video1/
│   ├── video1_detections.json
│   ├── video1_tracking.json
│   └── video1_cropped.mp4
├── video2/
│   ├── video2_detections.json
│   ├── video2_tracking.json
│   └── video2_cropped.mp4
└── ...
```

## Error Handling Strategy

1. **Invalid Directory**: Fail early with clear error message
2. **No Compatible Files**: Fail early with clear error message
3. **Individual File Failures**: Skip failed files, continue with remaining files
4. **Partial Success**: Complete successfully processed files, report failures in summary

## CLI Usage Examples

```bash
# Process single file (existing functionality)
auto-cropper process video.mp4

# Process all videos in a directory
auto-cropper process --input-directory ./videos/

# Process directory with custom settings
auto-cropper process --input-directory ./videos/ --confidence 0.7 --margin 75
```

## Testing Considerations

1. Test single file processing still works (regression test)
2. Test directory processing with multiple compatible files
3. Test directory processing with mixed file types (should skip incompatible)
4. Test directory processing with some files that fail
5. Test error cases (invalid directory, no compatible files)
6. Test output directory structure for batch mode

## Implementation Order

1. Add `find_video_files()` helper function
2. Add `process_single_file()` function (extract existing logic)
3. Add `process_batch()` function
4. Modify process command signature and add input validation
5. Update process command function body
6. Test thoroughly
7. Update documentation (README.md)

## Backward Compatibility

This change maintains 100% backward compatibility:
- Existing `auto-cropper process video.mp4` commands work unchanged
- All existing options continue to work
- Output structure for single files remains the same
- Error messages and behavior for single files remain the same
