# Auto-Cropper

This CLI app will take a .mp4 file as an input and use object detection to "follow" a single person in the video and crop to 16:9 ratio around the person.

## Features

- **Person Detection**: Uses YOLOv8 for accurate person detection in video frames
- **Person Tracking**: Automatically selects the most consistent person across frames
- **Smart Cropping**: Crops video to 16:9 aspect ratio following the tracked person
- **Smooth Motion**: Applies smoothing to reduce jitter in the cropped video
- **Memory-Efficient Processing**: Handles large video files with chunked processing and memory monitoring
- **Pipeline Architecture**: Separates detection, tracking, and cropping for flexibility and debugging
- **Comprehensive CLI**: Multiple commands with proper validation and helpful error messages

## Installation

### From Source

1. Clone the repository:
```bash
git clone <repository-url>
cd auto-cropper
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3. Install in development mode (optional):
```bash
pip install -e .
```

4. For development with additional tools:
```bash
pip install -r requirements-dev.txt
```

### Using pip (when published)

```bash
pip install auto-cropper
```

**Note**: All dependencies are pinned to specific versions for reproducibility and stability.

## Usage

Auto-cropper works in three main stages:

1. **Detection**: Detect people in video frames
2. **Tracking**: Automatically select the most consistent person to track
3. **Cropping**: Crop the video to follow the tracked person

### Quick Start (Complete Pipeline)

Process a video in one command:
```bash
auto-cropper process input_video.mp4
```

### Step-by-Step Process

#### Step 1: Detect People

```bash
auto-cropper detect input_video.mp4
```

This will:
- Analyze every frame of the video
- Detect all people using YOLOv8
- Save detection data to `./output/input_video_detections.json`

**Supported video formats**: .mp4, .avi, .mov, .mkv, .m4v, .wmv

#### Step 2: Select Person to Track

```bash
auto-cropper track ./output/input_video_detections.json
```

This will:
- Analyze the detection data
- Automatically select the most consistent person (person appearing in most frames)
- Save tracking data to `./output/input_video_tracking.json`

#### Step 3: Crop the Video

```bash
auto-cropper crop input_video.mp4 ./output/input_video_tracking.json
```

This will:
- Crop the video to follow the tracked person
- Output a 16:9 aspect ratio video
- Save to `./output/input_video_cropped.mp4`

## Command Reference

### Global Options

| Option | Description |
|--------|-------------|
| `--verbose, -v` | Enable verbose output for detailed processing information |

### Main Commands

| Command | Description |
|---------|-------------|
| `detect` | Detect people in video frames |
| `track` | Select the most consistent person to track from detection data |
| `crop` | Crop video based on tracking data |
| `process` | Run complete pipeline in one command |
| `summary` | Show detection statistics |

### Detection Options

```bash
auto-cropper detect video.mp4 [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--output-dir, -o` | `./output` | Output directory for detection data |
| `--confidence, -c` | `0.5` | Minimum detection confidence |

### Tracking Options

```bash
auto-cropper track detections.json
```

The track command automatically selects the person who appears most consistently across frames. No additional options are needed.

### Cropping Options

```bash
auto-cropper crop video.mp4 tracking.json [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--output, -o` | Auto-generated | Output video path |
| `--margin, -mg` | `50` | Margin around person (pixels) |
| `--smoothing, -s` | `10` | Smoothing window size |
| `--duration, -d` | None | Limit to first N seconds (optional) |

### Complete Pipeline

```bash
auto-cropper process video.mp4 [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--output-dir, -o` | `./output` | Output directory |
| `--confidence, -c` | `0.5` | Minimum detection confidence |
| `--margin, -mg` | `50` | Margin around person (pixels) |
| `--smoothing, -s` | `10` | Smoothing window size |
| `--duration, -d` | None | Limit to first N seconds (optional) |

Runs the complete detection, tracking, and cropping pipeline in one command. At the end, you'll be prompted to delete intermediate files (detection and tracking data) to save space.

## Examples

### Basic Usage

```bash
# Process a video with default settings
auto-cropper process my_video.mp4

# Enable verbose output for detailed information
auto-cropper --verbose process my_video.mp4

# Process with custom output directory
auto-cropper process my_video.mp4 --output-dir ./my_output

# Add more margin around the person
auto-cropper process my_video.mp4 --margin 100

# Limit processing to first 30 seconds
auto-cropper process my_video.mp4 --duration 30
```

### Step-by-Step with Custom Settings

```bash
# 1. Detect with high confidence threshold and verbose output
auto-cropper --verbose detect video.mp4 --confidence 0.7

# 2. Track the most consistent person
auto-cropper --verbose track ./output/video_detections.json

# 3. Crop with extra margin and smoothing
auto-cropper --verbose crop video.mp4 ./output/video_tracking.json --margin 75 --smoothing 15
```

### Check Detection Quality

```bash
# View detection statistics
auto-cropper summary ./output/video_detections.json
```

### Output Information

When processing is complete, the tool provides:
- File size comparison between original and cropped video
- Detection and tracking coverage statistics
- Processing time and performance information
- Location of all output files

### Performance Notes

- **Processing Time**: Expect 1-3 minutes per minute of video for detection (varies by hardware)
- **GPU Acceleration**: If you have a CUDA-capable GPU, detection will be significantly faster
- **Model Selection**: The tool uses `yolov8l.pt` by default for the best balance of accuracy and performance
  - The model is automatically downloaded on first use (87.8MB)
  - For faster processing with slightly lower accuracy, you can modify the code to use `yolov8n.pt` or `yolov8s.pt`

## Output Files

The tool creates several files during processing:

```
./output/
├── video_name_detections.json    # Raw detection data
├── video_name_tracking.json      # Selected person tracking data
└── video_name_cropped.mp4        # Final cropped video
```

### Detection Data Format

```json
{
  "video_info": {
    "video_path": "path/to/video.mp4",
    "total_frames": 1500,
    "fps": 30.0,
    "width": 1920,
    "height": 1080
  },
  "frames": [
    {
      "frame_number": 0,
      "timestamp": 0.0,
      "people": [
        {
          "bbox": {"x1": 100, "y1": 200, "x2": 300, "y2": 600},
          "confidence": 0.85,
          "center": {"x": 200, "y": 400}
        }
      ]
    }
  ]
}
```

## Requirements

- **Python**: 3.8 or higher
- **PyTorch**: For YOLOv8 model inference
- **OpenCV**: For video processing
- **Ultralytics**: YOLOv8 implementation

### Hardware Recommendations

- **CPU**: Multi-core processor for faster processing
- **GPU**: CUDA-capable GPU significantly speeds up detection
- **RAM**: 8GB+ recommended for larger videos
- **Storage**: Ensure adequate space for output files

## Troubleshooting

### Common Issues

**No people detected:**
- Try lowering the `--confidence` threshold
- The tool uses the accurate `yolov8l.pt` model by default
- Check if the video quality is sufficient

**Poor tracking:**
- The tool uses the 'most_consistent' tracking method which selects the person appearing in the most frames
- Check the detection summary to see if the person is consistently detected

**Jerky video:**
- Increase the `--smoothing` window size (default is 10 frames)
- Reduce the detection confidence to get more consistent detections

**Large output files:**
- The output video maintains the original frame rate and quality
- Consider post-processing to reduce file size if needed

## Contributing

**This is a personal project maintained by [@rewheaton](https://github.com/rewheaton).** 

### Usage & Forking
- **Fork this repository** for your own use and modifications
- **Clone and use** the code according to the MIT License
- **Report bugs** by creating an issue (if enabled)

### Direct Contributions
- **Pull requests are not accepted** - This project is not seeking direct contributions
- **Feature requests may not be implemented** - This is a personal tool

### Alternative Contribution Methods
- **Fork and improve** - Create your own enhanced version
- **Share your improvements** - Link to your fork in discussions or issues
- **Blog/write about it** - Share your experience using the tool

*If you've created a significant improvement or found a critical bug, feel free to reach out via GitHub issues.*

## License

This project is licensed under the MIT License - see the LICENSE file for details.