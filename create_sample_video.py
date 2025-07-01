#!/usr/bin/env python3
"""Create a sample video for testing the auto-cropper."""

import cv2
import numpy as np
from pathlib import Path

def create_sample_video(output_path="sample_video.mp4", duration_seconds=5):
    """Create a sample video with a moving person-like rectangle."""
    
    # Video parameters
    fps = 30.0
    width, height = 1280, 720  # 16:9 aspect ratio
    total_frames = int(duration_seconds * fps)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Creating sample video: {output_path}")
    print(f"Resolution: {width}x{height}, Duration: {duration_seconds}s, FPS: {fps}")
    
    for frame_num in range(total_frames):
        # Create background (indoor scene simulation)
        frame = np.full((height, width, 3), (40, 50, 60), dtype=np.uint8)
        
        # Add some background elements (walls, furniture)
        cv2.rectangle(frame, (0, 0), (width, height//3), (80, 70, 60), -1)  # Wall
        cv2.rectangle(frame, (50, height//2), (200, height-50), (100, 80, 60), -1)  # Furniture
        cv2.rectangle(frame, (width-250, height//2), (width-50, height-50), (90, 75, 65), -1)  # More furniture
        
        # Calculate person position (moving across the room)
        progress = frame_num / total_frames
        
        # Person moves from left to right, then back
        if progress < 0.5:
            # Moving right
            x_progress = progress * 2
        else:
            # Moving left
            x_progress = 2 - (progress * 2)
        
        # Person position
        person_width = 80
        person_height = 200
        person_x = int(100 + (width - 300) * x_progress)
        person_y = height - 250  # Standing on the ground
        
        # Add some slight vertical movement (walking bounce)
        bounce = int(10 * np.sin(frame_num * 0.3))
        person_y += bounce
        
        # Draw person (simplified human shape)
        # Body
        cv2.rectangle(frame, 
                     (person_x, person_y), 
                     (person_x + person_width, person_y + person_height), 
                     (180, 150, 120), -1)
        
        # Head
        head_radius = 30
        cv2.circle(frame, 
                  (person_x + person_width//2, person_y - head_radius), 
                  head_radius, 
                  (200, 180, 160), -1)
        
        # Arms (moving)
        arm_swing = int(20 * np.sin(frame_num * 0.5))
        # Left arm
        cv2.rectangle(frame,
                     (person_x - 15 + arm_swing, person_y + 40),
                     (person_x + 15 + arm_swing, person_y + 120),
                     (180, 150, 120), -1)
        # Right arm  
        cv2.rectangle(frame,
                     (person_x + person_width - 15 - arm_swing, person_y + 40),
                     (person_x + person_width + 15 - arm_swing, person_y + 120),
                     (180, 150, 120), -1)
        
        # Legs
        leg_width = 25
        # Left leg
        cv2.rectangle(frame,
                     (person_x + 15, person_y + person_height - 80),
                     (person_x + 15 + leg_width, person_y + person_height + 50),
                     (160, 130, 100), -1)
        # Right leg
        cv2.rectangle(frame,
                     (person_x + person_width - 15 - leg_width, person_y + person_height - 80),
                     (person_x + person_width - 15, person_y + person_height + 50),
                     (160, 130, 100), -1)
        
        # Add some noise/texture to make it more realistic
        noise = np.random.randint(-10, 10, (height, width, 3), dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Write frame
        out.write(frame)
        
        # Progress indicator
        if frame_num % 30 == 0:
            print(f"Progress: {frame_num}/{total_frames} frames ({progress*100:.1f}%)")
    
    out.release()
    print(f"✓ Sample video created: {output_path}")
    
    # Show file size
    file_size = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"File size: {file_size:.1f} MB")

if __name__ == '__main__':
    create_sample_video()
    print("\nTo test the auto-cropper with this video:")
    print("1. Process the entire pipeline:")
    print("   auto-cropper process sample_video.mp4")
    print("\n2. Or step by step:")
    print("   auto-cropper detect sample_video.mp4")
    print("   auto-cropper track ./output/sample_video_detections.json")
    print("   auto-cropper crop sample_video.mp4 ./output/sample_video_tracking.json")
