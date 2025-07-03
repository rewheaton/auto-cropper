"""Tests for PersonTracker class."""

import pytest
import tempfile
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from auto_cropper.person_tracker import PersonTracker


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield Path(tmpdirname)


@pytest.fixture
def sample_detection_data(temp_dir):
    """Create sample detection data with multiple people."""
    detection_data = {
        "video_info": {
            "video_path": str(temp_dir / "test_video.mp4"),
            "total_frames": 10,
            "fps": 30.0,
            "width": 640,
            "height": 480,
            "detection_settings": {
                "model": "yolov8l.pt",
                "confidence": 0.5
            }
        },
        "frames": []
    }
    
    # Add detection data for each frame with multiple people
    for i in range(10):
        # Person 1 - Large person in center (consistent across frames)
        person1_x = 250 + i * 2  # Slight movement
        person1_y = 200
        
        # Person 2 - Smaller person on left (appears in fewer frames)
        person2_x = 50
        person2_y = 150 + i * 3  # More movement
        
        people = [
            {
                "bbox": {
                    "x1": person1_x,
                    "y1": person1_y,
                    "x2": person1_x + 120,
                    "y2": person1_y + 200,
                    "width": 120,
                    "height": 200
                },
                "confidence": 0.85,
                "center": {
                    "x": person1_x + 60,
                    "y": person1_y + 100
                }
            }
        ]
        
        # Person 2 only appears in some frames
        if i < 7:
            people.append({
                "bbox": {
                    "x1": person2_x,
                    "y1": person2_y,
                    "x2": person2_x + 80,
                    "y2": person2_y + 150,
                    "width": 80,
                    "height": 150
                },
                "confidence": 0.75,
                "center": {
                    "x": person2_x + 40,
                    "y": person2_y + 75
                }
            })
        
        frame_data = {
            "frame_number": i,
            "timestamp": i / 30.0,
            "people": people
        }
        detection_data["frames"].append(frame_data)
    
    detection_file = temp_dir / "test_detections.json"
    with open(detection_file, 'w') as f:
        json.dump(detection_data, f)
    
    return detection_file


@pytest.fixture
def empty_detection_data(temp_dir):
    """Create detection data with no people."""
    detection_data = {
        "video_info": {
            "video_path": str(temp_dir / "empty_video.mp4"),
            "total_frames": 5,
            "fps": 30.0,
            "width": 640,
            "height": 480,
            "detection_settings": {
                "model": "yolov8l.pt",
                "confidence": 0.5
            }
        },
        "frames": []
    }
    
    # Add frames with no people
    for i in range(5):
        frame_data = {
            "frame_number": i,
            "timestamp": i / 30.0,
            "people": []
        }
        detection_data["frames"].append(frame_data)
    
    detection_file = temp_dir / "empty_detections.json"
    with open(detection_file, 'w') as f:
        json.dump(detection_data, f)
    
    return detection_file


@pytest.fixture
def single_person_detection_data(temp_dir):
    """Create detection data with single person across all frames."""
    detection_data = {
        "video_info": {
            "video_path": str(temp_dir / "single_person_video.mp4"),
            "total_frames": 5,
            "fps": 30.0,
            "width": 640,
            "height": 480,
            "detection_settings": {
                "model": "yolov8l.pt",
                "confidence": 0.5
            }
        },
        "frames": []
    }
    
    # Add single person across all frames
    for i in range(5):
        person_x = 300 + i  # Slight movement
        person_y = 240
        
        frame_data = {
            "frame_number": i,
            "timestamp": i / 30.0,
            "people": [
                {
                    "bbox": {
                        "x1": person_x,
                        "y1": person_y,
                        "x2": person_x + 100,
                        "y2": person_y + 180,
                        "width": 100,
                        "height": 180
                    },
                    "confidence": 0.9,
                    "center": {
                        "x": person_x + 50,
                        "y": person_y + 90
                    }
                }
            ]
        }
        detection_data["frames"].append(frame_data)
    
    detection_file = temp_dir / "single_person_detections.json"
    with open(detection_file, 'w') as f:
        json.dump(detection_data, f)
    
    return detection_file


class TestPersonTracker:
    """Test cases for PersonTracker class."""
    
    def test_init_default(self):
        """Test default initialization."""
        tracker = PersonTracker()
        assert tracker.verbose is False
        assert hasattr(tracker, 'logger')
    
    def test_init_verbose(self):
        """Test verbose initialization."""
        tracker = PersonTracker(verbose=True)
        assert tracker.verbose is True
        assert hasattr(tracker, 'logger')
    
    @patch('auto_cropper.person_tracker.logging')
    def test_logging_setup(self, mock_logging):
        """Test logging configuration."""
        # Test with verbose=False
        tracker = PersonTracker(verbose=False)
        mock_logging.basicConfig.assert_not_called()
        
        # Test with verbose=True  
        tracker = PersonTracker(verbose=True)
        mock_logging.basicConfig.assert_called_with(level=mock_logging.INFO)
    
    def test_select_person_to_track_most_consistent(self, sample_detection_data, temp_dir):
        """Test selecting most consistent person (now the only method)."""
        tracker = PersonTracker()
        tracking_file = tracker.select_person_to_track(str(sample_detection_data))
        
        assert Path(tracking_file).exists()
        assert tracking_file.endswith("_tracking.json")
        
        # Load and verify tracking data
        with open(tracking_file, 'r') as f:
            data = json.load(f)
        
        assert "video_info" in data
        assert "tracking_info" in data
        assert "tracked_person" in data
        assert data["tracking_info"]["total_tracked_frames"] > 0
        assert data["tracking_info"]["tracking_coverage"] > 0
        assert len(data["tracked_person"]) > 0
        
        # Verify tracking entry structure
        person_entry = data["tracked_person"][0]
        assert "frame_number" in person_entry
        assert "timestamp" in person_entry
        assert "bbox" in person_entry
        assert "center" in person_entry
        assert "confidence" in person_entry
        
        # Should track the person that appears in more frames
        assert data["tracking_info"]["total_tracked_frames"] >= 7  # Person 1 appears in all 10 frames
    
    def test_select_person_to_track_invalid_file(self, temp_dir):
        """Test with non-existent detection file."""
        tracker = PersonTracker()
        invalid_file = temp_dir / "nonexistent.json"
        
        with pytest.raises(FileNotFoundError):
            tracker.select_person_to_track(str(invalid_file))
    
    def test_select_person_to_track_malformed_json(self, temp_dir):
        """Test with malformed JSON file."""
        tracker = PersonTracker()
        malformed_file = temp_dir / "malformed.json"
        
        with open(malformed_file, 'w') as f:
            f.write("invalid json content")
        
        with pytest.raises(json.JSONDecodeError):
            tracker.select_person_to_track(str(malformed_file))
    
    def test_select_person_empty_detection_data(self, empty_detection_data, temp_dir):
        """Test tracking with empty detection data."""
        tracker = PersonTracker()
        tracking_file = tracker.select_person_to_track(str(empty_detection_data))
        
        assert Path(tracking_file).exists()
        
        with open(tracking_file, 'r') as f:
            data = json.load(f)
        
        assert data["tracking_info"]["total_tracked_frames"] == 0
        assert data["tracking_info"]["tracking_coverage"] == 0.0
        assert len(data["tracked_person"]) == 0
    
    def test_select_most_consistent_person_internal(self, sample_detection_data):
        """Test _select_most_consistent_person method directly."""
        tracker = PersonTracker()
        
        with open(sample_detection_data, 'r') as f:
            detection_data = json.load(f)
        
        result = tracker._select_most_consistent_person(detection_data)
        
        assert "tracked_person" in result
        assert result["tracking_info"]["total_tracked_frames"] > 0
        # Should select person 1 who appears in all frames
        assert result["tracking_info"]["total_tracked_frames"] == 10
    
    def test_create_tracking_data_internal(self, sample_detection_data):
        """Test _create_tracking_data method directly."""
        tracker = PersonTracker()
        
        with open(sample_detection_data, 'r') as f:
            detection_data = json.load(f)
        
        # Create sample person frames data
        person_frames = [(0, 0), (1, 0), (2, 0)]  # (frame_number, person_index)
        
        result = tracker._create_tracking_data(detection_data, person_frames)
        
        assert "video_info" in result
        assert "tracking_info" in result
        assert "tracked_person" in result
        assert result["tracking_info"]["total_tracked_frames"] == 3
        assert len(result["tracked_person"]) == 3
        
        # Verify tracking data is sorted by frame number
        frame_numbers = [entry["frame_number"] for entry in result["tracked_person"]]
        assert frame_numbers == sorted(frame_numbers)
    
    def test_create_tracking_data_with_area_tuples(self, sample_detection_data):
        """Test _create_tracking_data with 3-tuple format (area, frame_number, person_index)."""
        tracker = PersonTracker()
        
        with open(sample_detection_data, 'r') as f:
            detection_data = json.load(f)
        
        # Create sample person frames with area data (from largest person selection)
        person_frames = [(24000, 0, 0), (24000, 1, 0), (24000, 2, 0)]  # (area, frame_number, person_index)
        
        result = tracker._create_tracking_data(detection_data, person_frames)
        
        assert result["tracking_info"]["total_tracked_frames"] == 3
        assert len(result["tracked_person"]) == 3
    
    def test_create_empty_tracking_data_internal(self, sample_detection_data):
        """Test _create_empty_tracking_data method directly."""
        tracker = PersonTracker()
        
        with open(sample_detection_data, 'r') as f:
            detection_data = json.load(f)
        
        result = tracker._create_empty_tracking_data(detection_data)
        
        assert "video_info" in result
        assert "tracking_info" in result
        assert "tracked_person" in result
        assert result["tracking_info"]["total_tracked_frames"] == 0
        assert result["tracking_info"]["tracking_coverage"] == 0.0
        assert len(result["tracked_person"]) == 0
        
        # Video info should be preserved
        assert result["video_info"] == detection_data["video_info"]
    
    def test_tracking_filename_format(self, sample_detection_data, temp_dir):
        """Test that tracking file has correct naming format."""
        tracker = PersonTracker()
        tracking_file = tracker.select_person_to_track(str(sample_detection_data))
        
        expected_name = sample_detection_data.stem.replace("_detections", "") + "_tracking.json"
        assert Path(tracking_file).name == expected_name
        assert Path(tracking_file).parent == temp_dir
    
    def test_tracking_coverage_calculation(self, single_person_detection_data):
        """Test tracking coverage calculation."""
        tracker = PersonTracker()
        tracking_file = tracker.select_person_to_track(str(single_person_detection_data))
        
        with open(tracking_file, 'r') as f:
            data = json.load(f)
        
        # Should track 100% of frames when person appears in all frames
        assert data["tracking_info"]["tracking_coverage"] == 100.0
        assert data["tracking_info"]["total_tracked_frames"] == 5
    
    def test_person_grouping_by_center(self, temp_dir):
        """Test that people are grouped correctly by center position."""
        # Create detection data with same person moving slightly
        detection_data = {
            "video_info": {
                "video_path": str(temp_dir / "moving_person.mp4"),
                "total_frames": 3,
                "fps": 30.0,
                "width": 640,
                "height": 480
            },
            "frames": []
        }
        
        # Same person with slight position changes (should be grouped together)
        for i in range(3):
            frame_data = {
                "frame_number": i,
                "timestamp": i / 30.0,
                "people": [{
                    "bbox": {
                        "x1": 100 + i * 10,  # Moving 10 pixels per frame
                        "y1": 200,
                        "x2": 200 + i * 10,
                        "y2": 400,
                        "width": 100,
                        "height": 200
                    },
                    "confidence": 0.9,
                    "center": {
                        "x": 150 + i * 10,
                        "y": 300
                    }
                }]
            }
            detection_data["frames"].append(frame_data)
        
        detection_file = temp_dir / "moving_person_detections.json"
        with open(detection_file, 'w') as f:
            json.dump(detection_data, f)
        
        tracker = PersonTracker()
        tracking_file = tracker.select_person_to_track(str(detection_file))
        
        with open(tracking_file, 'r') as f:
            result = json.load(f)
        
        # Should track all 3 frames as the same person
        assert result["tracking_info"]["total_tracked_frames"] == 3
        assert result["tracking_info"]["tracking_coverage"] == 100.0
    
    def test_invalid_person_index_handling(self, temp_dir):
        """Test handling of invalid person indices in tracking data creation."""
        # Create minimal detection data
        detection_data = {
            "video_info": {"video_path": str(temp_dir / "test.mp4"), "total_frames": 1},
            "frames": [{
                "frame_number": 0,
                "timestamp": 0.0,
                "people": [{
                    "bbox": {"x1": 100, "y1": 100, "x2": 200, "y2": 300, "width": 100, "height": 200},
                    "confidence": 0.9,
                    "center": {"x": 150, "y": 200}
                }]
            }]
        }
        
        tracker = PersonTracker()
        
        # Test with invalid person index (should be skipped)
        person_frames = [(0, 5)]  # Frame 0, person index 5 (doesn't exist)
        result = tracker._create_tracking_data(detection_data, person_frames)
        
        # The method still counts the input frames but skips invalid indices when creating entries
        assert result["tracking_info"]["total_tracked_frames"] == 1  # Input frame count
        assert len(result["tracked_person"]) == 0  # No valid entries created
    
    @patch('auto_cropper.person_tracker.logging')
    def test_verbose_logging_output(self, mock_logging, sample_detection_data, temp_dir):
        """Test that verbose mode produces logging output."""
        tracker = PersonTracker(verbose=True)
        
        # Mock the logger
        mock_logger = Mock()
        tracker.logger = mock_logger
        
        tracking_file = tracker.select_person_to_track(str(sample_detection_data))
        
        # Verify logging calls were made
        assert mock_logger.info.call_count >= 2  # At least method selection and file saved messages
        
        # Check specific log messages
        call_args = [call.args[0] for call in mock_logger.info.call_args_list]
        assert any("Selecting most consistent person to track" in msg for msg in call_args)
        assert any("Tracking data saved to:" in msg for msg in call_args)
    
    def test_edge_case_empty_frames_list(self, temp_dir):
        """Test handling of detection data with empty frames list."""
        detection_data = {
            "video_info": {
                "video_path": str(temp_dir / "empty_frames.mp4"),
                "total_frames": 0,
                "fps": 30.0,
                "width": 640,
                "height": 480
            },
            "frames": []
        }
        
        detection_file = temp_dir / "empty_frames_detections.json"
        with open(detection_file, 'w') as f:
            json.dump(detection_data, f)
        
        tracker = PersonTracker()
        tracking_file = tracker.select_person_to_track(str(detection_file))
        
        with open(tracking_file, 'r') as f:
            result = json.load(f)
        
        assert result["tracking_info"]["total_tracked_frames"] == 0
        assert result["tracking_info"]["tracking_coverage"] == 0.0  # Should handle division by zero
        assert len(result["tracked_person"]) == 0
