"""Tests for the StreamingJSONWriter class."""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

from auto_cropper.streaming_json_writer import StreamingJSONWriter
from auto_cropper.exceptions import StreamingWriterException


class TestStreamingJSONWriter:
    """Test cases for StreamingJSONWriter class."""
    
    def test_init_creates_file_with_header(self, tmp_path):
        """Test StreamingJSONWriter initialization creates file with proper header."""
        test_file = tmp_path / "test_output.json"
        video_info = {
            "video_path": "test.mp4",
            "total_frames": 100,
            "fps": 30.0,
            "width": 1920,
            "height": 1080
        }
        
        writer = StreamingJSONWriter(str(test_file), video_info)
        writer.close()
        
        # Verify file exists and has correct header
        assert test_file.exists()
        with open(test_file, 'r') as f:
            content = f.read()
        
        assert content.startswith('{"video_info": ')
        assert '"frames": [' in content
        assert content.endswith(']}')
    
    def test_write_single_frame(self, tmp_path):
        """Test writing a single frame."""
        test_file = tmp_path / "test_single_frame.json"
        video_info = {"test": "data"}
        frame_data = {
            "frame_number": 0,
            "timestamp": 0.0,
            "people": [{"bbox": {"x1": 100, "y1": 100, "x2": 200, "y2": 200}}]
        }
        
        writer = StreamingJSONWriter(str(test_file), video_info)
        writer.write_frame(frame_data)
        writer.close()
        
        # Verify content
        with open(test_file, 'r') as f:
            data = json.load(f)
        
        assert data["video_info"] == video_info
        assert len(data["frames"]) == 1
        assert data["frames"][0] == frame_data
    
    def test_write_multiple_frames(self, tmp_path):
        """Test writing multiple frames."""
        test_file = tmp_path / "test_multiple_frames.json"
        video_info = {"test": "data"}
        
        frames = [
            {"frame_number": 0, "timestamp": 0.0, "people": []},
            {"frame_number": 1, "timestamp": 0.033, "people": [{"bbox": {"x1": 100, "y1": 100, "x2": 200, "y2": 200}}]},
            {"frame_number": 2, "timestamp": 0.066, "people": []}
        ]
        
        writer = StreamingJSONWriter(str(test_file), video_info)
        for frame in frames:
            writer.write_frame(frame)
        writer.close()
        
        # Verify content
        with open(test_file, 'r') as f:
            data = json.load(f)
        
        assert data["video_info"] == video_info
        assert len(data["frames"]) == 3
        assert data["frames"] == frames
    
    def test_context_manager(self, tmp_path):
        """Test StreamingJSONWriter as context manager."""
        test_file = tmp_path / "test_context.json"
        video_info = {"test": "context"}
        frame_data = {"frame_number": 0, "timestamp": 0.0, "people": []}
        
        with StreamingJSONWriter(str(test_file), video_info) as writer:
            writer.write_frame(frame_data)
        
        # Verify file is properly closed and content is correct
        with open(test_file, 'r') as f:
            data = json.load(f)
        
        assert data["video_info"] == video_info
        assert len(data["frames"]) == 1
        assert data["frames"][0] == frame_data
    
    def test_get_frame_count(self, tmp_path):
        """Test frame count tracking."""
        test_file = tmp_path / "test_count.json"
        video_info = {"test": "count"}
        
        writer = StreamingJSONWriter(str(test_file), video_info)
        assert writer.get_frame_count() == 0
        
        writer.write_frame({"frame_number": 0})
        assert writer.get_frame_count() == 1
        
        writer.write_frame({"frame_number": 1})
        assert writer.get_frame_count() == 2
        
        writer.close()
    
    def test_file_creation_error(self, tmp_path):
        """Test handling of file creation errors."""
        # Try to create file in a path that would cause permission error
        # Using a path that doesn't exist and can't be created
        import os
        if os.name == 'posix':  # Unix-like systems
            # Try to create file in /root which typically requires permissions
            invalid_path = "/root/nonexistent/test.json"
        else:
            # For Windows or other systems, skip this test
            pytest.skip("Permission test not applicable on this system")
        
        video_info = {"test": "error"}
        
        with pytest.raises(StreamingWriterException):
            StreamingJSONWriter(invalid_path, video_info)
    
    def test_write_after_close_error(self, tmp_path):
        """Test error when writing after close."""
        test_file = tmp_path / "test_closed.json"
        video_info = {"test": "closed"}
        
        writer = StreamingJSONWriter(str(test_file), video_info)
        writer.close()
        
        with pytest.raises(ValueError, match="Cannot write to closed writer"):
            writer.write_frame({"frame_number": 0})
    
    def test_double_close_safe(self, tmp_path):
        """Test that closing twice is safe."""
        test_file = tmp_path / "test_double_close.json"
        video_info = {"test": "double"}
        
        writer = StreamingJSONWriter(str(test_file), video_info)
        writer.close()
        writer.close()  # Should not raise an error
        
        # Verify file is still valid JSON
        with open(test_file, 'r') as f:
            data = json.load(f)
        assert data["video_info"] == video_info
    
    def test_flush_behavior(self, tmp_path):
        """Test that data is flushed to disk."""
        test_file = tmp_path / "test_flush.json"
        video_info = {"test": "flush"}
        frame_data = {"frame_number": 0, "timestamp": 0.0, "people": []}
        
        writer = StreamingJSONWriter(str(test_file), video_info)
        writer.write_frame(frame_data)
        
        # Even without closing, file should contain the frame data due to flushing
        # Read the raw file content to check if data was flushed
        with open(test_file, 'r') as f:
            content = f.read()
        
        # Should contain the frame data even before closing
        assert '"frame_number": 0' in content
        
        writer.close()
    
    def test_invalid_json_data_handling(self, tmp_path):
        """Test handling of data that can't be serialized to JSON."""
        test_file = tmp_path / "test_invalid_json.json"
        video_info = {"test": "invalid"}
        
        # Create data that can't be JSON serialized
        invalid_frame = {"frame_number": 0, "data": object()}  # object() can't be JSON serialized
        
        writer = StreamingJSONWriter(str(test_file), video_info)
        
        with pytest.raises(StreamingWriterException):  # JSON serialization wrapped in StreamingWriterException
            writer.write_frame(invalid_frame)
        
        writer.close()
    
    def test_large_frame_data(self, tmp_path):
        """Test writing large frame data."""
        test_file = tmp_path / "test_large.json"
        video_info = {"test": "large"}
        
        # Create a frame with many detections
        large_frame = {
            "frame_number": 0,
            "timestamp": 0.0,
            "people": [
                {
                    "bbox": {"x1": i, "y1": i, "x2": i+100, "y2": i+100},
                    "confidence": 0.9,
                    "center": {"x": i+50, "y": i+50}
                }
                for i in range(100)  # 100 detections
            ]
        }
        
        writer = StreamingJSONWriter(str(test_file), video_info)
        writer.write_frame(large_frame)
        writer.close()
        
        # Verify data integrity
        with open(test_file, 'r') as f:
            data = json.load(f)
        
        assert len(data["frames"][0]["people"]) == 100
        assert data["frames"][0]["people"][99]["bbox"]["x1"] == 99
    
    def test_file_path_as_pathlib_path(self, tmp_path):
        """Test using pathlib.Path as file path."""
        test_file = tmp_path / "test_pathlib.json"
        video_info = {"test": "pathlib"}
        
        # Pass Path object instead of string
        writer = StreamingJSONWriter(test_file, video_info)
        writer.write_frame({"frame_number": 0})
        writer.close()
        
        # Verify file was created correctly
        assert test_file.exists()
        with open(test_file, 'r') as f:
            data = json.load(f)
        assert data["video_info"] == video_info
    
    def test_write_empty_frame(self, tmp_path):
        """Test writing frame with no people detections."""
        test_file = tmp_path / "test_empty.json"
        video_info = {"test": "empty"}
        empty_frame = {"frame_number": 0, "timestamp": 0.0, "people": []}
        
        writer = StreamingJSONWriter(str(test_file), video_info)
        writer.write_frame(empty_frame)
        writer.close()
        
        with open(test_file, 'r') as f:
            data = json.load(f)
        
        assert len(data["frames"]) == 1
        assert data["frames"][0]["people"] == []
