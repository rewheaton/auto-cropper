"""Command line interface for auto-cropper video processing."""

import os
import sys
from pathlib import Path
from typing import Optional

import click

from .detector import PersonDetector, PersonTracker
from .video_cropper import VideoCropper


@click.group()
@click.option('--verbose', '-v', is_flag=True, default=True, help='Enable verbose output')
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
@click.argument('video_path', type=click.Path(exists=True))
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
    
    try:
        detector = PersonDetector(
            model_name='yolov8l.pt',
            confidence=confidence,
            verbose=verbose
        )
        
        detection_file = detector.detect_people_in_video(video_path, output_dir)
        
        # Show summary
        summary = detector.get_detection_summary(detection_file)
        
        click.echo(f"\n✓ Detection complete!")
        click.echo(f"📁 Detection data saved to: {detection_file}")
        click.echo(f"\n📊 Summary:")
        click.echo(f"   Total frames: {summary['total_frames']}")
        click.echo(f"   Frames with people: {summary['frames_with_people']}")
        click.echo(f"   Detection coverage: {summary['detection_coverage']}%")
        click.echo(f"   Total detections: {summary['total_detections']}")
        click.echo(f"   Average people per frame: {summary['average_people_per_frame']}")
        click.echo(f"   Max people in single frame: {summary['max_people_in_frame']}")
        
        if summary['frames_with_people'] == 0:
            click.echo("\n⚠️  No people detected in the video. Try lowering the confidence threshold.")
        else:
            click.echo(f"\n➡️  Next step: Run tracking to select which person to follow:")
            click.echo(f"   auto-cropper track {detection_file}")
            
    except Exception as e:
        click.echo(f"❌ Error during detection: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument('detection_file', type=click.Path(exists=True))
@click.pass_context
def track(ctx, detection_file: str):
    """
    Select a person to track from detection data.
    
    DETECTION_FILE: Path to the detection JSON file from the detect command.
    """
    verbose = ctx.obj['verbose']
    
    try:
        tracker = PersonTracker(verbose=verbose)
        tracking_file = tracker.select_person_to_track(detection_file, 'most_consistent')
        
        # Load and show tracking summary
        import json
        with open(tracking_file, 'r') as f:
            tracking_data = json.load(f)
        
        tracking_info = tracking_data['tracking_info']
        
        click.echo(f"\n✓ Tracking complete!")
        click.echo(f"📁 Tracking data saved to: {tracking_file}")
        click.echo(f"\n📊 Tracking Summary:")
        click.echo(f"   Selection method: most_consistent")
        click.echo(f"   Tracked frames: {tracking_info['total_tracked_frames']}")
        click.echo(f"   Tracking coverage: {tracking_info['tracking_coverage']}%")
        
        if tracking_info['total_tracked_frames'] == 0:
            click.echo("\n⚠️  No person could be tracked. Try a different tracking method.")
        else:
            # Get original video path from detection data
            video_path = tracking_data['video_info']['video_path']
            click.echo(f"\n➡️  Next step: Crop the video:")
            click.echo(f"   auto-cropper crop {video_path} {tracking_file}")
            
    except Exception as e:
        click.echo(f"❌ Error during tracking: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument('video_path', type=click.Path(exists=True))
@click.argument('tracking_file', type=click.Path(exists=True))
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
    
    try:
        cropper = VideoCropper(
            margin=margin,
            smoothing_window=smoothing,
            verbose=verbose
        )
        
        output_path = cropper.crop_video(video_path, tracking_file, output, duration_limit=duration)
        
        click.echo(f"\n✓ Video cropping complete!")
        click.echo(f"🎬 Cropped video saved to: {output_path}")
        
        # Show file size info
        original_size = Path(video_path).stat().st_size
        cropped_size = Path(output_path).stat().st_size
        size_ratio = cropped_size / original_size * 100
        
        click.echo(f"\n📏 File size comparison:")
        click.echo(f"   Original: {original_size / 1024 / 1024:.1f} MB")
        click.echo(f"   Cropped: {cropped_size / 1024 / 1024:.1f} MB ({size_ratio:.1f}% of original)")
        
    except Exception as e:
        click.echo(f"❌ Error during cropping: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument('video_path', type=click.Path(exists=True))
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
def process(ctx, video_path: str, output_dir: str, confidence: float, 
           margin: int, smoothing: int, duration: Optional[int]):
    """
    Complete pipeline: detect, track, and crop in one command.
    
    VIDEO_PATH: Path to the input video file
    """
    verbose = ctx.obj['verbose']
    
    click.echo("🎬 Starting complete video processing pipeline...")
    
    try:
        # Step 1: Detection
        click.echo("\n📍 Step 1: Detecting people in video...")
        detector = PersonDetector(model_name='yolov8l.pt', confidence=confidence, verbose=verbose)
        detection_file = detector.detect_people_in_video(video_path, output_dir)
        
        # Step 2: Tracking
        click.echo("\n🎯 Step 2: Selecting person to track...")
        tracker = PersonTracker(verbose=verbose)
        tracking_file = tracker.select_person_to_track(detection_file, 'most_consistent')
        
        # Step 3: Cropping
        click.echo("\n✂️  Step 3: Cropping video...")
        cropper = VideoCropper(margin=margin, smoothing_window=smoothing, verbose=verbose)
        output_path = cropper.crop_video(video_path, tracking_file, output_dir=output_dir, duration_limit=duration)
        
        click.echo(f"\n🎉 Complete! Final video saved to: {output_path}")
        
        # Clean up intermediate files option
        click.echo(f"\n🗑️  Intermediate files:")
        click.echo(f"   Detection data: {detection_file}")
        click.echo(f"   Tracking data: {tracking_file}")
        
        if click.confirm("Delete intermediate files?"):
            Path(detection_file).unlink()
            Path(tracking_file).unlink()
            click.echo("✓ Intermediate files deleted")
        
    except Exception as e:
        click.echo(f"❌ Error in processing pipeline: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument('detection_file', type=click.Path(exists=True))
def summary(detection_file: str):
    """
    Show summary of detection data.
    
    DETECTION_FILE: Path to detection JSON file
    """
    try:
        detector = PersonDetector()
        summary_data = detector.get_detection_summary(detection_file)
        
        click.echo(f"\n📊 Detection Summary for {Path(detection_file).name}")
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
        click.echo(f"❌ Error reading detection file: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
