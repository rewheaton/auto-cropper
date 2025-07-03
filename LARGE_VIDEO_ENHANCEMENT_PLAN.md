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

## Migration Path

### Phase 1 Implementation (Weeks 1-2)
1. Implement `MemoryMonitor` class
2. Add chunked detection processing
3. Create streaming JSON writer
4. Add basic memory management CLI options
5. Update error handling for memory issues

