"""Command line interface for auto-cropper video processing."""

import os
import sys
from pathlib import Path
from typing import Optional, List

import click

from auto_cropper.person_detector import PersonDetector
from auto_cropper.person_tracker import PersonTracker
from auto_cropper.video_cropper import VideoCropper

def validate_input_file(file_path: str, allowed_extensions: Optional[List[str]] = None) -> Path:
    """
    Validate that input file exists and has a valid extension.
    
    Args:
        file_path: Path to the input file
        allowed_extensions: List of allowed file extensions (e.g., ['.mp4', '.avi'])
    
    Returns:
        Path object if valid
        
    Raises:
        click.ClickException: If file is invalid
    """
    if allowed_extensions is None:
        allowed_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.m4v', '.wmv']
    
    path = Path(file_path)
    
    # Check if file exists
    if not path.exists():
        raise click.ClickException(f"Input file does not exist: {file_path}")
    
    # Check if it's a file (not directory)
    if not path.is_file():
        raise click.ClickException(f"Path is not a file: {file_path}")
    
    # Check file extension
    if path.suffix.lower() not in [ext.lower() for ext in allowed_extensions]:
        raise click.ClickException(
            f"Unsupported file format: {path.suffix}\n"
            f"   Supported formats: {', '.join(allowed_extensions)}"
        )
    
    # Check file size (warn if very large)
    file_size_mb = path.stat().st_size / 1024 / 1024
    if file_size_mb > 1000:  # 1GB
        click.echo(f"WARNING: Large file detected ({file_size_mb:.1f} MB). Processing may take a while.", err=True)
    
    # Check if file is readable
    try:
        with open(path, 'rb') as f:
            f.read(1)
    except PermissionError:
        raise click.ClickException(f"Permission denied: Cannot read file {file_path}")
    except Exception as e:
        raise click.ClickException(f"Cannot access file {file_path}: {e}")
    
    return path


def ensure_output_directory(output_dir: str) -> Path:
    """
    Ensure output directory exists and is writable.
    
    Args:
        output_dir: Path to the output directory
        
    Returns:
        Path object for the directory
        
    Raises:
        click.ClickException: If directory cannot be created or accessed
    """
    path = Path(output_dir)
    
    try:
        # Create directory if it doesn't exist
        path.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        raise click.ClickException(f"Permission denied: Cannot create directory {output_dir}")
    except Exception as e:
        raise click.ClickException(f"Cannot create directory {output_dir}: {e}")
    
    # Check if directory is writable
    if not os.access(path, os.W_OK):
        raise click.ClickException(f"Directory is not writable: {output_dir}")
    
    return path


def validate_json_file(file_path: str, expected_keys: Optional[List[str]] = None) -> Path:
    """
    Validate that a JSON file exists and has expected structure.
    
    Args:
        file_path: Path to the JSON file
        expected_keys: List of required top-level keys
        
    Returns:
        Path object if valid
        
    Raises:
        click.ClickException: If file is invalid
    """
    import json
    
    path = Path(file_path)
    
    # Basic file existence check
    if not path.exists():
        raise click.ClickException(f"JSON file does not exist: {file_path}")
    
    if not path.is_file():
        raise click.ClickException(f"Path is not a file: {file_path}")
    
    # Check if it's a JSON file
    if path.suffix.lower() != '.json':
        raise click.ClickException(f"File must be a JSON file: {file_path}")
    
    # Try to parse JSON
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise click.ClickException(f"Invalid JSON file {file_path}: {e}")
    except PermissionError:
        raise click.ClickException(f"Permission denied: Cannot read file {file_path}")
    except Exception as e:
        raise click.ClickException(f"Cannot read JSON file {file_path}: {e}")
    
    # Check expected keys
    if expected_keys:
        missing_keys = [key for key in expected_keys if key not in data]
        if missing_keys:
            raise click.ClickException(
                f"JSON file {file_path} is missing required keys: {missing_keys}"
            )
    
    return path


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


@click.group()
@click.option('--verbose', '-v', is_flag=True, default=False, help='Enable verbose output')
@click.pass_context
def main(ctx, verbose):
    """
    Auto-Cropper: Automatically crop videos to follow a person with 16:9 aspect ratio.
    
    This tool works in stages:
    1. Detect people in video frames
    2. Track a specific person across frames  
    3. Crop the video to follow the tracked person
    """
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose


@main.command()
@click.argument('video_path', type=click.Path())
@click.option(
    '--output-dir', '-o',
    type=click.Path(),
    default='./output',
    help='Output directory for detection data (default: ./output)'
)
@click.option(
    '--confidence', '-c',
    type=float,
    default=0.5,
    help='Minimum confidence threshold for person detection (default: 0.5)'
)
@click.pass_context
def detect(ctx, video_path: str, output_dir: str, confidence: float):
    """
    Detect people in video frames and save detection data.
    
    VIDEO_PATH: Path to the input video file (.mp4, .avi, etc.)
    """
    verbose = ctx.obj['verbose']
    
    # Validate inputs
    try:
        validated_video_path = validate_input_file(video_path)
        validated_output_dir = ensure_output_directory(output_dir)
    except click.ClickException:
        raise  # Re-raise click exceptions as-is
    
    try:
        detector = PersonDetector(
            model_name='yolov8l.pt',
            confidence=confidence,
            verbose=verbose
        )
        
        detection_file = detector.detect_people_in_video(str(validated_video_path), str(validated_output_dir))
        
        # Show summary
        summary = detector.get_detection_summary(detection_file)
        
        click.echo(f"\nDetection complete!")
        click.echo(f"Detection data saved to: {detection_file}")
        click.echo(f"\nSummary:")
        click.echo(f"   Total frames: {summary['total_frames']}")
        click.echo(f"   Frames with people: {summary['frames_with_people']}")
        click.echo(f"   Detection coverage: {summary['detection_coverage']}%")
        click.echo(f"   Total detections: {summary['total_detections']}")
        click.echo(f"   Average people per frame: {summary['average_people_per_frame']}")
        click.echo(f"   Max people in single frame: {summary['max_people_in_frame']}")
        
        if summary['frames_with_people'] == 0:
            click.echo("\nWARNING: No people detected in the video. Try lowering the confidence threshold.")
        else:
            click.echo(f"\nNext step: Run tracking to select which person to follow:")
            click.echo(f"   auto-cropper track {detection_file}")
            
    except Exception as e:
        click.echo(f"Error during detection: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument('detection_file', type=click.Path())
@click.pass_context
def track(ctx, detection_file: str):
    """
    Select a person to track from detection data.
    
    DETECTION_FILE: Path to the detection JSON file from the detect command.
    """
    verbose = ctx.obj['verbose']
    
    # Validate inputs
    try:
        validated_detection_file = validate_json_file(detection_file, ['video_info', 'frames'])
    except click.ClickException:
        raise  # Re-raise click exceptions as-is
    
    try:
        tracker = PersonTracker(verbose=verbose)
        tracking_file = tracker.select_person_to_track(str(validated_detection_file))
        
        # Load and show tracking summary
        import json
        with open(tracking_file, 'r') as f:
            tracking_data = json.load(f)
        
        tracking_info = tracking_data['tracking_info']
        
        click.echo(f"\nTracking complete!")
        click.echo(f"Tracking data saved to: {tracking_file}")
        click.echo(f"\nTracking Summary:")
        click.echo(f"   Tracked frames: {tracking_info['total_tracked_frames']}")
        click.echo(f"   Tracking coverage: {tracking_info['tracking_coverage']}%")
        
        if tracking_info['total_tracked_frames'] == 0:
            click.echo("\nWARNING: No person could be tracked consistently across frames.")
        else:
            # Get original video path from detection data
            video_path = tracking_data['video_info']['video_path']
            click.echo(f"\nNext step: Crop the video:")
            click.echo(f"   auto-cropper crop {video_path} {tracking_file}")
            
    except Exception as e:
        click.echo(f"Error during tracking: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument('video_path', type=click.Path())
@click.argument('tracking_file', type=click.Path())
@click.option(
    '--output', '-o',
    type=click.Path(),
    help='Output video path (default: auto-generated in output directory)'
)
@click.option(
    '--margin', '-mg',
    type=int,
    default=50,
    help='Margin around person in pixels (default: 50)'
)
@click.option(
    '--smoothing', '-s',
    type=int,
    default=10,
    help='Smoothing window size to reduce jitter (default: 10)'
)
@click.option(
    '--duration', '-d',
    type=int,
    help='Limit cropping to first N seconds of video (optional)'
)
@click.pass_context
def crop(ctx, video_path: str, tracking_file: str, output: Optional[str], margin: int, smoothing: int, duration: Optional[int]):
    """
    Crop video to follow the tracked person with 16:9 aspect ratio.
    
    VIDEO_PATH: Path to the original video file
    TRACKING_FILE: Path to the tracking JSON file from the track command
    """
    verbose = ctx.obj['verbose']
    
    # Validate inputs
    try:
        validated_video_path = validate_input_file(video_path)
        validated_tracking_file = validate_json_file(tracking_file, ['video_info', 'tracking_info', 'frames'])
        
        # Ensure output directory exists if output path is specified
        if output:
            output_path = Path(output)
            if output_path.parent != Path('.'):
                ensure_output_directory(str(output_path.parent))
    except click.ClickException:
        raise  # Re-raise click exceptions as-is
    
    try:
        cropper = VideoCropper(
            margin=margin,
            smoothing_window=smoothing,
            verbose=verbose
        )
        
        output_path = cropper.crop_video(str(validated_video_path), str(validated_tracking_file), output, duration_limit=duration)
        
        click.echo(f"\nVideo cropping complete!")
        click.echo(f"Cropped video saved to: {output_path}")
        
        # Show file size info
        original_size = Path(validated_video_path).stat().st_size
        cropped_size = Path(output_path).stat().st_size
        size_ratio = cropped_size / original_size * 100
        
        click.echo(f"\nFile size comparison:")
        click.echo(f"   Original: {original_size / 1024 / 1024:.1f} MB")
        click.echo(f"   Cropped: {cropped_size / 1024 / 1024:.1f} MB ({size_ratio:.1f}% of original)")
        
    except Exception as e:
        click.echo(f"Error during cropping: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument('video_path', type=click.Path(), required=False)
@click.option(
    '--input-directory', '-id',
    type=click.Path(),
    help='Process all compatible video files in this directory'
)
@click.option(
    '--output-dir', '-o',
    type=click.Path(),
    default='./output',
    help='Output directory (default: ./output)'
)
@click.option(
    '--confidence', '-c',
    type=float,
    default=0.5,
    help='Minimum confidence threshold for person detection (default: 0.5)'
)
@click.option(
    '--margin', '-mg',
    type=int,
    default=50,
    help='Margin around person in pixels (default: 50)'
)
@click.option(
    '--smoothing', '-s',
    type=int,
    default=10,
    help='Smoothing window size to reduce jitter (default: 10)'
)
@click.option(
    '--duration', '-d',
    type=int,
    help='Limit cropping to first N seconds of video (optional)'
)
@click.pass_context
def process(ctx, video_path: Optional[str], input_directory: Optional[str], output_dir: str, confidence: float, 
           margin: int, smoothing: int, duration: Optional[int]):
    """
    Complete pipeline: detect, track, and crop in one command.
    
    Process either a single video file or all compatible video files in a directory.
    
    VIDEO_PATH: Path to the input video file (required unless --input-directory is used)
    """
    verbose = ctx.obj['verbose']
    
    # Validate that exactly one input method is provided
    if not video_path and not input_directory:
        raise click.ClickException("Must provide either VIDEO_PATH or --input-directory")
    
    if video_path and input_directory:
        raise click.ClickException("Cannot specify both VIDEO_PATH and --input-directory")
    
    # Validate output directory
    try:
        validated_output_dir = ensure_output_directory(output_dir)
    except click.ClickException:
        raise  # Re-raise click exceptions as-is
    
    if input_directory:
        # Batch processing mode
        try:
            video_files = find_video_files(input_directory)
            if not video_files:
                raise click.ClickException(f"No compatible video files found in directory: {input_directory}")
            
            process_batch(video_files, validated_output_dir, confidence, margin, smoothing, duration, verbose)
        except Exception as e:
            click.echo(f"Error during batch processing: {e}", err=True)
            sys.exit(1)
    else:
        # Single file processing mode (existing logic)
        try:
            # video_path is guaranteed to be not None here due to validation above
            validated_video_path = validate_input_file(video_path)  # type: ignore
            
            click.echo("Starting complete video processing pipeline...")
            
            # Process single file and handle cleanup
            detection_file, tracking_file, output_path = process_single_file(
                validated_video_path, validated_output_dir, confidence, margin, smoothing, duration, verbose
            )
            
            click.echo(f"\nPipeline complete!")
            click.echo(f"Final video saved to: {output_path}")
            
            # Clean up intermediate files option
            click.echo(f"\nIntermediate files:")
            click.echo(f"   Detection data: {detection_file}")
            click.echo(f"   Tracking data: {tracking_file}")
            
            if click.confirm("Delete intermediate files?"):
                Path(detection_file).unlink()
                Path(tracking_file).unlink()
                click.echo("Intermediate files deleted")
                
        except Exception as e:
            click.echo(f"Error during processing: {e}", err=True)
            sys.exit(1)


@main.command()
@click.argument('detection_file', type=click.Path())
@click.pass_context
def summary(ctx, detection_file: str):
    """
    Show summary of detection data.
    
    DETECTION_FILE: Path to detection JSON file
    """
    verbose = ctx.obj['verbose']
    
    # Validate input
    try:
        validated_detection_file = validate_json_file(detection_file, ['video_info', 'frames'])
    except click.ClickException:
        raise  # Re-raise click exceptions as-is
    
    try:
        detector = PersonDetector(verbose=verbose)
        summary_data = detector.get_detection_summary(str(validated_detection_file))
        
        click.echo(f"\nDetection Summary for {Path(detection_file).name}")
        click.echo("=" * 50)
        
        video_info = summary_data['video_info']
        click.echo(f"Video: {Path(video_info['video_path']).name}")
        click.echo(f"Resolution: {video_info['width']}x{video_info['height']}")
        click.echo(f"Duration: {video_info['total_frames']} frames at {video_info['fps']:.1f} FPS")
        click.echo(f"         ({video_info['total_frames'] / video_info['fps']:.1f} seconds)")
        
        click.echo(f"\nDetection Results:")
        click.echo(f"  Total frames: {summary_data['total_frames']}")
        click.echo(f"  Frames with people: {summary_data['frames_with_people']}")
        click.echo(f"  Frames without people: {summary_data['frames_without_people']}")
        click.echo(f"  Detection coverage: {summary_data['detection_coverage']}%")
        click.echo(f"  Total detections: {summary_data['total_detections']}")
        click.echo(f"  Average people per frame: {summary_data['average_people_per_frame']}")
        click.echo(f"  Max people in single frame: {summary_data['max_people_in_frame']}")
        
    except Exception as e:
        click.echo(f"Error reading detection file: {e}", err=True)
        sys.exit(1)


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


if __name__ == '__main__':
    main()
