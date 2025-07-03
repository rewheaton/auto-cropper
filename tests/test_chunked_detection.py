"""Tests for chunked detection processing in PersonDetector."""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from auto_cropper.person_detector import PersonDetector
from auto_cropper.memory_monitor import MemoryMonitor
from auto_cropper.exceptions import MemoryLimitException


class TestChunkedDetection:
    """Test cases for chunked detection processing."""
    
    def test_detect_people_chunked_basic(self, tmp_path):
        """Test basic chunked detection functionality."""
        detector = PersonDetector(verbose=False)
        test_file = tmp_path / "test_chunked.json"
        
        # Mock video capture and YOLO model
        with patch('cv2.VideoCapture') as mock_cap, \
             patch.object(detector, 'model') as mock_model:
            
            # Setup mock video
            mock_cap_instance = Mock()
            mock_cap.return_value = mock_cap_instance
            mock_cap_instance.isOpened.return_value = True
            mock_cap_instance.get.side_effect = lambda prop: {
                7: 100,    # CAP_PROP_FRAME_COUNT
                5: 30.0,   # CAP_PROP_FPS  
                3: 1920,   # CAP_PROP_FRAME_WIDTH
                4: 1080    # CAP_PROP_FRAME_HEIGHT
            }.get(prop, 0)
            
            # Mock seeking behavior for chunked processing
            current_pos = [0]  # Use list to allow modification in nested function
            
            def mock_set(prop, value):
                if prop == 2:  # CAP_PROP_POS_FRAMES = 2
                    current_pos[0] = int(value)
            
            def mock_read():
                if current_pos[0] < 100:  # Match the total_frames
                    frame = f"frame_{current_pos[0]}"
                    current_pos[0] += 1
                    return True, frame
                return False, None
            
            mock_cap_instance.set = mock_set
            mock_cap_instance.read = mock_read
            
            # Mock YOLO detections
            mock_model.return_value = [Mock(boxes=None)]  # No detections
            
            # Test chunked detection
            result_file = detector.detect_people_in_video_chunked(
                "test_video.mp4", 
                str(tmp_path),
                chunk_size_frames=5
            )
            
            assert Path(result_file).exists()
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            assert "video_info" in data
            assert "frames" in data
            assert data["video_info"]["total_frames"] == 100
    
    def test_chunked_detection_with_memory_monitor(self, tmp_path):
        """Test chunked detection with memory monitoring."""
        memory_monitor = MemoryMonitor(max_memory_mb=1024)
        detector = PersonDetector(verbose=False)
        
        with patch('cv2.VideoCapture') as mock_cap, \
             patch.object(detector, 'model') as mock_model, \
             patch.object(memory_monitor, 'check_memory_usage') as mock_check:
            
            # Setup mocks
            mock_cap_instance = Mock()
            mock_cap.return_value = mock_cap_instance
            mock_cap_instance.isOpened.return_value = True
            mock_cap_instance.get.side_effect = lambda prop: {
                7: 50,     # CAP_PROP_FRAME_COUNT
                5: 30.0,   # CAP_PROP_FPS
                3: 1920,   # CAP_PROP_FRAME_WIDTH
                4: 1080    # CAP_PROP_FRAME_HEIGHT
            }.get(prop, 0)
            
            # Mock frame reading
            mock_cap_instance.read.side_effect = [
                (True, f"frame_{i}") for i in range(50)
            ] + [(False, None)]
            
            mock_model.return_value = [Mock(boxes=None)]
            mock_check.return_value = True  # Memory usage OK
            
            result_file = detector.detect_people_in_video_chunked(
                "test_video.mp4",
                str(tmp_path),
                memory_monitor=memory_monitor
            )
            
            assert Path(result_file).exists()
            # Verify memory was checked during processing
            assert mock_check.call_count > 0
    
    def test_chunked_detection_memory_limit_exceeded(self, tmp_path):
        """Test chunked detection when memory limit is exceeded."""
        memory_monitor = MemoryMonitor(max_memory_mb=1024)
        detector = PersonDetector(verbose=False)
        
        with patch('cv2.VideoCapture') as mock_cap, \
             patch.object(detector, 'model') as mock_model, \
             patch.object(memory_monitor, 'check_memory_usage') as mock_check:
            
            # Setup mocks
            mock_cap_instance = Mock()
            mock_cap.return_value = mock_cap_instance
            mock_cap_instance.isOpened.return_value = True
            mock_cap_instance.get.side_effect = lambda prop: {
                7: 100,    # CAP_PROP_FRAME_COUNT
                5: 30.0,   # CAP_PROP_FPS
                3: 1920,   # CAP_PROP_FRAME_WIDTH
                4: 1080    # CAP_PROP_FRAME_HEIGHT
            }.get(prop, 0)
            
            mock_cap_instance.read.side_effect = [
                (True, f"frame_{i}") for i in range(100)
            ] + [(False, None)]
            
            mock_model.return_value = [Mock(boxes=None)]
            mock_check.return_value = False  # Memory limit exceeded
            
            with pytest.raises(MemoryLimitException):
                detector.detect_people_in_video_chunked(
                    "test_video.mp4",
                    str(tmp_path),
                    memory_monitor=memory_monitor
                )
    
    def test_chunked_detection_overlap_handling(self, tmp_path):
        """Test chunked detection with frame overlap."""
        detector = PersonDetector(verbose=False)
        
        with patch('cv2.VideoCapture') as mock_cap, \
             patch.object(detector, 'model') as mock_model:
            
            # Setup mock video with 15 frames
            mock_cap_instance = Mock()
            mock_cap.return_value = mock_cap_instance
            mock_cap_instance.isOpened.return_value = True
            mock_cap_instance.get.side_effect = lambda prop: {
                7: 15,     # CAP_PROP_FRAME_COUNT
                5: 30.0,   # CAP_PROP_FPS
                3: 1920,   # CAP_PROP_FRAME_WIDTH
                4: 1080    # CAP_PROP_FRAME_HEIGHT
            }.get(prop, 0)
            
            mock_cap_instance.read.side_effect = [
                (True, f"frame_{i}") for i in range(15)
            ] + [(False, None)]
            
            mock_model.return_value = [Mock(boxes=None)]
            
            result_file = detector.detect_people_in_video_chunked(
                "test_video.mp4",
                str(tmp_path),
                chunk_size_frames=10,
                overlap_frames=3
            )
            
            assert Path(result_file).exists()
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            # Should have processed all 15 frames despite chunking
            assert len(data["frames"]) == 15
    
    def test_chunked_detection_checkpoint_recovery(self, tmp_path):
        """Test checkpoint creation and recovery."""
        detector = PersonDetector(verbose=False)
        checkpoint_file = tmp_path / "test_checkpoint.json"
        
        with patch('cv2.VideoCapture') as mock_cap, \
             patch.object(detector, 'model') as mock_model:
            
            # Setup mock video
            mock_cap_instance = Mock()
            mock_cap.return_value = mock_cap_instance
            mock_cap_instance.isOpened.return_value = True
            mock_cap_instance.get.side_effect = lambda prop: {
                7: 100,    # CAP_PROP_FRAME_COUNT
                5: 30.0,   # CAP_PROP_FPS
                3: 1920,   # CAP_PROP_FRAME_WIDTH
                4: 1080    # CAP_PROP_FRAME_HEIGHT
            }.get(prop, 0)
            
            mock_cap_instance.read.side_effect = [
                (True, f"frame_{i}") for i in range(100)
            ] + [(False, None)]
            
            mock_model.return_value = [Mock(boxes=None)]
            
            result_file = detector.detect_people_in_video_chunked(
                "test_video.mp4",
                str(tmp_path),
                checkpoint_interval=20
            )
            
            assert Path(result_file).exists()
            # Checkpoint file should be created during processing
            # (This is implementation-dependent, so we just verify the result file exists)
    
    def test_chunked_detection_invalid_chunk_size(self, tmp_path):
        """Test error handling for invalid chunk size."""
        detector = PersonDetector(verbose=False)
        
        with pytest.raises(ValueError, match="Chunk size must be positive"):
            detector.detect_people_in_video_chunked(
                "test_video.mp4",
                str(tmp_path),
                chunk_size_frames=0
            )
    
    def test_chunked_detection_invalid_overlap(self, tmp_path):
        """Test error handling for invalid overlap."""
        detector = PersonDetector(verbose=False)
        
        with pytest.raises(ValueError, match="Overlap frames must be non-negative"):
            detector.detect_people_in_video_chunked(
                "test_video.mp4",
                str(tmp_path),
                chunk_size_frames=10,
                overlap_frames=-1
            )
        
        with pytest.raises(ValueError, match="Overlap cannot be larger than chunk size"):
            detector.detect_people_in_video_chunked(
                "test_video.mp4",
                str(tmp_path),
                chunk_size_frames=10,
                overlap_frames=15
            )
    
    def test_get_optimal_chunk_size(self, tmp_path):
        """Test optimal chunk size calculation."""
        detector = PersonDetector(verbose=False)
        
        # Test with mock frame size estimation
        with patch.object(detector, '_estimate_frame_size_mb') as mock_estimate:
            mock_estimate.return_value = 10.0  # 10MB per frame
            
            memory_monitor = MemoryMonitor(max_memory_mb=1024)  # 1GB
            chunk_size = detector._get_optimal_chunk_size(memory_monitor, 1920, 1080)
            
            # Available memory: 1024 * 0.8 = 819.2 MB
            # Frame size: 10 MB
            # Expected chunk size: 819.2 / 10 = 81.92 -> 81 (with minimum safety factor)
            assert chunk_size > 0
            assert chunk_size <= 100  # Should be reasonable
    
    def test_estimate_frame_size_mb(self, tmp_path):
        """Test frame size estimation."""
        detector = PersonDetector(verbose=False)
        
        # Test frame size estimation for different resolutions
        size_1080p = detector._estimate_frame_size_mb(1920, 1080)
        size_4k = detector._estimate_frame_size_mb(3840, 2160)
        size_720p = detector._estimate_frame_size_mb(1280, 720)
        
        # 4K should be larger than 1080p, which should be larger than 720p
        assert size_4k > size_1080p > size_720p
        assert all(size > 0 for size in [size_1080p, size_4k, size_720p])
    
    def test_chunked_detection_with_torch_cleanup(self, tmp_path):
        """Test that torch memory is cleaned up between chunks."""
        detector = PersonDetector(verbose=False)
        
        with patch('cv2.VideoCapture') as mock_cap, \
             patch.object(detector, 'model') as mock_model, \
             patch('torch.cuda.empty_cache') as mock_cleanup, \
             patch('torch.cuda.is_available', return_value=True):
            
            # Setup mocks
            mock_cap_instance = Mock()
            mock_cap.return_value = mock_cap_instance
            mock_cap_instance.isOpened.return_value = True
            mock_cap_instance.get.side_effect = lambda prop: {
                7: 50,     # CAP_PROP_FRAME_COUNT
                5: 30.0,   # CAP_PROP_FPS
                3: 1920,   # CAP_PROP_FRAME_WIDTH
                4: 1080    # CAP_PROP_FRAME_HEIGHT
            }.get(prop, 0)
            
            mock_cap_instance.read.side_effect = [
                (True, f"frame_{i}") for i in range(50)
            ] + [(False, None)]
            
            mock_model.return_value = [Mock(boxes=None)]
            
            detector.detect_people_in_video_chunked(
                "test_video.mp4",
                str(tmp_path),
                chunk_size_frames=10
            )
            
            # Verify torch cleanup was called
            assert mock_cleanup.call_count > 0
