# Auto-Cropper

This CLI app will take a .mp4 file as an input and use object detection to "follow" a single person in the video and crop to 16:9 ratio around the person.

## Features

- **Person Detection**: Uses YOLOv8 for accurate person detection in video frames
- **Person Tracking**: Intelligently selects and tracks a specific person across frames
- **Smart Cropping**: Crops video to 16:9 aspect ratio following the tracked person
- **Smooth Motion**: Applies smoothing to reduce jitter in the cropped video
- **Two-Stage Process**: Separates detection from cropping for flexibility and debugging
- **Multiple Selection Methods**: Choose how to select which person to track

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

3. Install in development mode:
```bash
pip install -e .
```

### Using pip (when published)

```bash
pip install auto-cropper
```

## Usage

Auto-cropper works in two main stages:

1. **Detection & Tracking**: Detect people and select one to track
2. **Cropping**: Crop the video to follow the tracked person

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

#### Step 2: Select Person to Track

```bash
auto-cropper track ./output/input_video_detections.json --method largest
```

This will:
- Analyze the detection data
- Select which person to track based on the chosen method
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

### Main Commands

| Command | Description |
|---------|-------------|
| `detect` | Detect people in video frames |
| `track` | Select a person to track from detection data |
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
| `--model, -m` | `yolov8l.pt` | YOLO model (n/s/m/l/x) |
| `--confidence, -c` | `0.5` | Minimum detection confidence |

### Tracking Options

```bash
auto-cropper track detections.json [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--method, -m` | `largest` | Person selection method |

**Selection Methods:**
- `largest`: Select person with largest average bounding box
- `most_consistent`: Select person who appears in most frames
- `center`: Select person closest to center of frame

### Cropping Options

```bash
auto-cropper crop video.mp4 tracking.json [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--output, -o` | Auto-generated | Output video path |
| `--margin, -mg` | `50` | Margin around person (pixels) |
| `--smoothing, -s` | `10` | Smoothing window size |

### Complete Pipeline

```bash
auto-cropper process video.mp4 [OPTIONS]
```

Combines all detection, tracking, and cropping options.

## Examples

### Basic Usage

```bash
# Process a video with default settings
auto-cropper process my_video.mp4

# Use a more accurate model (slower but better detection)
auto-cropper --verbose process my_video.mp4 --model yolov8m.pt

# Track the most consistent person
auto-cropper --verbose process my_video.mp4 --track-method most_consistent

# Add more margin around the person
auto-cropper --verbose process my_video.mp4 --margin 100
```

### Step-by-Step with Custom Settings

```bash
# 1. Detect with high confidence threshold
auto-cropper --verbose detect video.mp4 --confidence 0.7 --model yolov8s.pt

# 2. Track the person closest to center
auto-cropper --verbose track ./output/video_detections.json --method center

# 3. Crop with extra margin and smoothing
auto-cropper --verbose crop video.mp4 ./output/video_tracking.json --margin 75 --smoothing 15
```

### Check Detection Quality

```bash
# View detection statistics
auto-cropper summary ./output/video_detections.json
```

### Performance Notes

- **Processing Time**: Expect 1-3 minutes per minute of video for detection (varies by hardware)
- **GPU Acceleration**: If you have a CUDA-capable GPU, detection will be significantly faster
- **Model Selection**: 
  - `yolov8n.pt`: Fastest, least accurate
  - `yolov8s.pt`: Good balance of speed and accuracy
  - `yolov8m.pt`: Better accuracy, slower
  - `yolov8l.pt` and `yolov8x.pt`: Best accuracy, slowest

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
- Use a more accurate model (e.g., `yolov8m.pt` instead of `yolov8n.pt`)
- Check if the video quality is sufficient

**Poor tracking:**
- Try a different `--track-method`
- Check the detection summary to see if the person is consistently detected
- Consider using manual detection data editing

**Jerky video:**
- Increase the `--smoothing` window size
- Reduce the detection confidence to get more consistent detections

**Large output files:**
- The output video maintains the original frame rate and quality
- Consider post-processing to reduce file size if needed

## Development

### Setting up Development Environment

```bash
git clone <repository-url>
cd auto-cropper
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black auto_cropper/ tests/
flake8 auto_cropper/ tests/
mypy auto_cropper/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.