#!/usr/bin/env python3
"""
Demo script showing how to use auto-cropper with step-by-step examples.
"""

import subprocess
import sys
from pathlib import Path


def run_command(command, description):
    """Run a command and show the description."""
    print(f"\n{description}")
    print(f"Command: {command}")
    print("-" * 50)
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("Command completed successfully")
            if result.stdout:
                print(result.stdout)
        else:
            print("Command failed")
            if result.stderr:
                print(f"Error: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"Error running command: {e}")
        return False


def main():
    """Run demo of auto-cropper functionality."""
    print("Auto-Cropper Demo")
    print("=" * 50)
    
    # Check if video file exists
    video_file = "input/GX010314.MP4"
    if not Path(video_file).exists():
        print(f"Video file not found: {video_file}")
        print("Please add a video file to the input/ directory")
        return
    
    print(f"Using video file: {video_file}")
    
    # Demo 1: Show help
    run_command("auto-cropper --help", "Show main help")
    
    # Demo 2: Show detect help
    run_command("auto-cropper detect --help", "Show detection command help")
    
    # Demo 3: Check if we have existing detection data
    detection_file = "output/GX010314_detections.json"
    if Path(detection_file).exists():
        print(f"\nFound existing detection data: {detection_file}")
        
        # Show summary of existing detection data
        run_command(f"auto-cropper summary {detection_file}", "Show detection summary")
        
        # Demo 4: Show tracking options
        run_command("auto-cropper track --help", "Show tracking command help")
        
        # Demo 5: Show cropping options  
        run_command("auto-cropper crop --help", "Show cropping command help")
        
        print("\nTo run the complete pipeline:")
        print("   auto-cropper --verbose process input/GX010314.MP4")
        
        print("\nOr step by step:")
        print("   1. auto-cropper --verbose detect input/GX010314.MP4")
        print("   2. auto-cropper --verbose track output/GX010314_detections.json --method largest")
        print("   3. auto-cropper --verbose crop input/GX010314.MP4 output/GX010314_tracking.json")
        
    else:
        print(f"\nNo detection data found. To start processing:")
        print(f"   auto-cropper --verbose detect {video_file}")
        print("\n   This will take several minutes depending on video length...")
    
    print("\nAvailable tracking methods:")
    print("   - largest: Select person with largest average bounding box")
    print("   - most_consistent: Select person who appears in most frames")
    print("   - center: Select person closest to center of frame")
    
    print("\nModel options (speed vs accuracy):")
    print("   - yolov8n.pt: Fastest, good for testing")
    print("   - yolov8s.pt: Good balance")
    print("   - yolov8m.pt: Better accuracy, slower")
    print("   - yolov8l.pt, yolov8x.pt: Best accuracy, slowest")


if __name__ == "__main__":
    main()
