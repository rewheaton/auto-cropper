#!/usr/bin/env python3
"""Create a sample image for testing the auto-cropper."""

from PIL import Image, ImageDraw
import os

def create_sample_image():
    """Create a sample image with borders for testing."""
    # Create a 400x300 image with white borders
    img = Image.new('RGB', (400, 300), 'white')
    draw = ImageDraw.Draw(img)
    
    # Add some colorful content in the center
    # Background rectangle
    draw.rectangle([80, 60, 320, 240], fill='lightblue')
    
    # Main content rectangle
    draw.rectangle([100, 80, 300, 220], fill='darkblue')
    
    # Some text-like rectangles
    draw.rectangle([120, 100, 280, 110], fill='white')
    draw.rectangle([120, 120, 250, 130], fill='white')
    draw.rectangle([120, 140, 270, 150], fill='white')
    
    # A circle for visual interest
    draw.ellipse([240, 160, 280, 200], fill='red')
    
    return img

if __name__ == '__main__':
    img = create_sample_image()
    img.save('sample_image.png')
    print("Created sample_image.png for testing")
