# Large Video File Handling Enhancement Plan

## Current Limitations Analysis

### Memory Usage Issues
- **Frame-by-frame processing**: Currently loads entire video metadata and processes sequentially
- **Detection data storage**: All detection results stored in memory before writing to JSON
- **Crop position calculation**: Pre-calculates all crop positions for entire video
- **No memory management**: No mechanisms to limit memory usage for large files

### Performance Bottlenecks
- **Single-threaded processing**: Detection and cropping are sequential operations
- **No frame skipping**: Processes every frame regardless of video length
- **Large JSON files**: Detection data grows linearly with video length
- **No chunked processing**: Entire video processed in one operation

### Current Memory Footprint (Estimated)
- **Per frame detection**: ~1-2KB JSON data per frame
- **60min video @ 30fps**: ~108,000 frames = ~200MB detection data
- **4K video frames**: ~25MB per frame in memory during processing
- **Total peak memory**: Could exceed 8-16GB for large 4K videos

## Enhancement Strategy

### Phase 1: Memory Optimization (High Priority)

#### 1.1 Streaming Detection Processing
**Objective**: Reduce memory usage during detection phase

**Implementation**:
```python
class StreamingPersonDetector(PersonDetector):
    def __init__(self, max_memory_mb: int = 2048, **kwargs):
        super().__init__(**kwargs)
        self.max_memory_mb = max_memory_mb
        self.batch_size = self._calculate_batch_size()
    
    def detect_people_in_video_streaming(
        self, 
        video_path: str, 
        output_dir: str = "./output",
        checkpoint_interval: int = 1000
    ) -> str:
        """Process video in chunks with memory management."""
        # Implementation details below
```

**Features**:
- **Chunked processing**: Process video in configurable batches
- **Incremental JSON writing**: Stream detection results to file
- **Memory monitoring**: Track and limit memory usage
- **Checkpoint system**: Save progress periodically for recovery

#### 1.2 Optimized Data Structures
**Objective**: Reduce memory footprint of detection data

**Changes**:
- **Compact JSON format**: Remove redundant data, use shorter keys
- **Binary detection cache**: Optional binary format for faster I/O
- **Sparse storage**: Only store frames with detections above threshold
- **Compressed storage**: Optional gzip compression for detection files

#### 1.3 Lazy Loading for Crop Processing
**Objective**: Avoid loading all crop positions into memory

**Implementation**:
```python
class LazyVideoCropper(VideoCropper):
    def crop_video_lazy(
        self, 
        video_path: str, 
        tracking_file: str,
        chunk_size_frames: int = 1000,
        **kwargs
    ) -> str:
        """Process video cropping in chunks."""
        # Implementation details below
```

### Phase 2: Performance Optimization (Medium Priority)

#### 2.1 Multi-threading Enhancement
**Objective**: Parallel processing where possible

**Implementation Areas**:
- **Frame decoding**: Separate thread for video reading
- **Detection batching**: Process multiple frames in parallel
- **I/O operations**: Background writing of detection data
- **Crop processing**: Parallel frame cropping

#### 2.2 GPU Memory Management
**Objective**: Optimize YOLO model usage for large batches

**Features**:
- **Batch inference**: Process multiple frames per YOLO call
- **GPU memory monitoring**: Prevent out-of-memory errors
- **Model quantization**: Use lighter models for initial processing
- **Dynamic batching**: Adjust batch size based on available memory

## Implementation Details

### Memory Management System

#### Memory Monitor
```python
class MemoryMonitor:
    def __init__(self, max_memory_mb: int = 2048):
        self.max_memory_mb = max_memory_mb
        self.current_usage = 0
    
    def check_memory_usage(self) -> bool:
        """Return True if memory usage is within limits."""
        import psutil
        current = psutil.Process().memory_info().rss / 1024 / 1024
        return current < self.max_memory_mb
    
    def get_recommended_batch_size(self, frame_size_mb: float) -> int:
        """Calculate optimal batch size based on available memory."""
        available_mb = self.max_memory_mb * 0.8  # Leave 20% buffer
        return max(1, int(available_mb / frame_size_mb))
```

#### Chunked Detection Processing
```python
class ChunkedDetector(PersonDetector):
    def detect_people_chunked(
        self,
        video_path: str,
        output_dir: str,
        chunk_size_frames: int = 1000,
        overlap_frames: int = 10
    ) -> str:
        """
        Process video in overlapping chunks to manage memory.
        
        Args:
            chunk_size_frames: Number of frames per chunk
            overlap_frames: Overlap between chunks for continuity
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        detection_file = output_dir / f"{video_path.stem}_detections.json"
        
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize output file
        video_info = self._get_video_info(cap, video_path)
        with open(detection_file, 'w') as f:
            json.dump({"video_info": video_info, "frames": []}, f)
        
        # Process in chunks
        for start_frame in range(0, total_frames, chunk_size_frames - overlap_frames):
            end_frame = min(start_frame + chunk_size_frames, total_frames)
            self._process_chunk(cap, start_frame, end_frame, detection_file)
            
            # Memory cleanup
            if hasattr(self, 'model'):
                torch.cuda.empty_cache()  # Clear GPU memory
        
        cap.release()
        return str(detection_file)
```

### Progressive File Format

#### Streaming JSON Writer
```python
class StreamingJSONWriter:
    def __init__(self, filepath: str, video_info: Dict):
        self.filepath = filepath
        self.file = open(filepath, 'w')
        self.frame_count = 0
        
        # Write header
        self.file.write('{"video_info": ')
        json.dump(video_info, self.file)
        self.file.write(', "frames": [')
        self.file.flush()
    
    def write_frame(self, frame_data: Dict):
        """Write a single frame's detection data."""
        if self.frame_count > 0:
            self.file.write(',')
        json.dump(frame_data, self.file)
        self.file.flush()
        self.frame_count += 1
    
    def close(self):
        """Close the file properly."""
        self.file.write(']}')
        self.file.close()
```

### Configuration System

#### Performance Profiles
```python
class ProcessingProfile:
    """Predefined processing profiles for different use cases."""
    
    FAST = {
        'model': 'yolov8n.pt',
        'confidence': 0.6,
        'max_memory_mb': 1024,
        'chunk_size_frames': 500,
        'sampling_rate': 0.5,  # Process every other frame
    }
    
    BALANCED = {
        'model': 'yolov8s.pt',
        'confidence': 0.5,
        'max_memory_mb': 2048,
        'chunk_size_frames': 1000,
        'sampling_rate': 1.0,
    }
    
    QUALITY = {
        'model': 'yolov8l.pt',
        'confidence': 0.4,
        'max_memory_mb': 4096,
        'chunk_size_frames': 2000,
        'sampling_rate': 1.0,
    }
    
    LARGE_FILE = {
        'model': 'yolov8s.pt',
        'confidence': 0.5,
        'max_memory_mb': 1024,
        'chunk_size_frames': 250,
        'sampling_rate': 0.3,  # Heavy downsampling
    }
```

## CLI Enhancements

### New Command Options
```bash
# Memory management
auto-cropper detect video.mp4 --max-memory 2048 --profile large-file
auto-cropper detect video.mp4 --chunk-size 1000 --streaming

# Performance optimization
auto-cropper detect video.mp4 --sampling-rate 0.5 --parallel-jobs 4
auto-cropper detect video.mp4 --target-fps 15 --fast-mode

# Large file specific
auto-cropper detect video.mp4 --profile large-file --checkpoint-interval 1000
auto-cropper process video.mp4 --resume-from checkpoint_1000.json
```

### Enhanced Error Handling
```python
class LargeFileException(Exception):
    """Exception for large file processing issues."""
    pass

class MemoryLimitException(LargeFileException):
    """Raised when memory limit is exceeded."""
    pass

class ProcessingTimeoutException(LargeFileException):
    """Raised when processing takes too long."""
    pass
```

## Testing Strategy

### Large File Test Suite
```python
class TestLargeFiles:
    def test_4k_60fps_video(self):
        """Test with 4K 60fps video file."""
        pass
    
    def test_8_hour_video(self):
        """Test with very long duration video."""
        pass
    
    def test_memory_limits(self):
        """Test with constrained memory settings."""
        pass
    
    def test_checkpoint_recovery(self):
        """Test recovery from checkpoints."""
        pass
```

### Benchmark Suite
```python
class PerformanceBenchmarks:
    def benchmark_memory_usage(self, video_path: str, max_memory: int):
        """Benchmark memory usage during processing."""
        pass
    
    def benchmark_processing_speed(self, video_path: str, profile: str):
        """Benchmark processing speed with different profiles."""
        pass
```

## Migration Path

### Phase 1 Implementation (Weeks 1-2)
1. Implement `MemoryMonitor` class
2. Add chunked detection processing
3. Create streaming JSON writer
4. Add basic memory management CLI options
5. Update error handling for memory issues

### Phase 2 Implementation (Weeks 3-4)
1. Implement frame sampling options
2. Add multi-threading for I/O operations
3. Create processing profiles
4. Add GPU memory management
5. Implement checkpoint/resume functionality

## Expected Outcomes

### Memory Usage Improvements
- **70% reduction** in peak memory usage for large files
- **90% reduction** in detection data memory footprint
- Support for **unlimited video length** (within storage constraints)

### Performance Improvements
- **50% faster** processing for files >2GB through chunking
- **2-3x faster** processing with multi-threading

### User Experience Improvements
- **Real-time progress** with detailed memory/time estimates
- **Resumable processing** for interrupted operations
- **Automatic optimization** based on system capabilities
- **Clear guidance** for large file processing

## Additional Considerations

### Disk Space Management

#### Storage Optimization
```python
class StorageManager:
    def __init__(self, min_free_space_gb: float = 5.0):
        self.min_free_space_gb = min_free_space_gb
    
    def check_disk_space(self, video_path: str, estimated_output_size: int) -> bool:
        """Check if sufficient disk space is available."""
        import shutil
        free_space = shutil.disk_usage(video_path).free
        required_space = estimated_output_size * 2  # 2x buffer for temp files
        return free_space > (required_space + self.min_free_space_gb * 1024**3)
    
    def estimate_output_size(self, video_path: str, crop_ratio: float = 0.7) -> int:
        """Estimate final output file size."""
        input_size = Path(video_path).stat().st_size
        return int(input_size * crop_ratio)  # Cropping typically reduces size
```

#### Temporary File Management
- **Smart cleanup**: Remove temporary files as soon as they're no longer needed
- **Compression**: Use compressed intermediate formats where possible
- **Streaming output**: Write final video directly without large intermediate files
- **Progress checkpoints**: Minimize checkpoint file sizes

### Monitoring and Analytics Dashboard

#### Real-time Metrics
```python
class ProcessingMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.metrics = {
            'memory_usage': [],
            'processing_speed': [],
            'frames_processed': 0,
            'estimated_completion': None
        }
    
    def update_progress(self, frames_processed: int, total_frames: int):
        """Update processing metrics and estimates."""
        elapsed = time.time() - self.start_time
        progress_ratio = frames_processed / total_frames
        
        if progress_ratio > 0.1:  # After 10% processed
            estimated_total_time = elapsed / progress_ratio
            self.metrics['estimated_completion'] = estimated_total_time - elapsed
        
        self.metrics['frames_processed'] = frames_processed
        self.metrics['processing_speed'].append(frames_processed / elapsed)
```

#### Performance Analytics
- **Processing speed trends**: Track frames per second over time
- **Memory usage patterns**: Identify memory spikes and optimization opportunities
- **Error frequency**: Monitor and categorize processing errors
- **Hardware utilization**: CPU, GPU, and storage usage metrics

### Edge Cases and Error Recovery

#### Corrupted Video Handling
- **Frame validation**: Detect and skip corrupted frames
- **Format detection**: Handle unusual video formats and codecs
- **Metadata recovery**: Reconstruct missing video metadata
- **Partial processing**: Continue processing despite minor errors

### Integration Enhancements

#### API and Webhook Support
```python
class WebhookNotifier:
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
    
    def notify_progress(self, progress: Dict):
        """Send progress updates to external systems."""
        pass
    
    def notify_completion(self, result: Dict):
        """Notify when processing is complete."""
        pass
```

#### External Tool Integration
- **FFmpeg integration**: Use FFmpeg for specialized video operations

## Design Constraints

### Video Quality Preservation
- **No preprocessing resizing**: The original video must never be resized or downscaled before processing
- **Maintain original resolution**: All detection and tracking operations must work with the source video resolution
- **Quality preservation**: Any optimizations must preserve the original video quality and dimensions
- **Post-processing only**: Video compression or format conversion may only occur after cropping is complete

## Compatibility Considerations

### Backward Compatibility
- All existing CLI commands continue to work unchanged
- New features are opt-in through additional flags
- Existing detection/tracking file formats remain supported
- Legacy processing mode available via `--legacy` flag

### System Requirements
- **Minimum RAM**: 4GB (previously 8GB recommended)
- **Recommended RAM**: 8GB for large files
- **Storage**: Additional space for checkpoint files
- **Python**: Requires psutil for memory monitoring

## Documentation Updates

### User Guide Additions
- Large file processing best practices
- Memory optimization guide
- Troubleshooting guide for large files
- Performance tuning recommendations

### Developer Documentation
- Memory management architecture
- Chunked processing implementation details
- Extension points for custom processors
- Performance monitoring and profiling

This enhancement plan provides a comprehensive approach to handling very large video files while maintaining the tool's ease of use and reliability.
