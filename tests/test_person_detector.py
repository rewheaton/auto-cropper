"""Tests for PersonDetector class."""

import pytest
import tempfile
import json
import os
from pathlib import Path
import numpy as np
import cv2
from unittest.mock import Mock, patch, MagicMock

from auto_cropper.person_detector import PersonDetector


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield Path(tmpdirname)


@pytest.fixture
def sample_video(temp_dir):
    """Create a sample video for testing."""
    video_path = temp_dir / "test_video.mp4"
    
    # Create a simple test video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30.0
    frame_size = (640, 480)
    
    out = cv2.VideoWriter(str(video_path), fourcc, fps, frame_size)
    
    # Create 30 frames (1 second of video)
    for i in range(30):
        # Create a frame with a moving rectangle (simulating a person)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (50, 50, 50)  # Dark gray background
        
        # Moving rectangle position
        x = 100 + i * 5  # Move right over time
        y = 200
        
        # Draw a rectangle (simulating a person)
        cv2.rectangle(frame, (x, y), (x + 100, y + 200), (255, 255, 255), -1)
        
        out.write(frame)
    
    out.release()
    return video_path


@pytest.fixture
def sample_detection_data(temp_dir):
    """Create sample detection data."""
    detection_data = {
        "video_info": {
            "video_path": str(temp_dir / "test_video.mp4"),
            "total_frames": 30,
            "fps": 30.0,
            "width": 640,
            "height": 480,
            "detection_settings": {
                "model": "yolov8n.pt",
                "confidence": 0.5
            }
        },
        "frames": []
    }
    
    # Add detection data for each frame
    for i in range(30):
        x = 100 + i * 5  # Moving person
        y = 200
        
        frame_data = {
            "frame_number": i,
            "timestamp": i / 30.0,
            "people": [
                {
                    "bbox": {
                        "x1": x,
                        "y1": y,
                        "x2": x + 100,
                        "y2": y + 200,
                        "width": 100,
                        "height": 200
                    },
                    "confidence": 0.85,
                    "center": {
                        "x": x + 50,
                        "y": y + 100
                    }
                }
            ]
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
            "video_path": str(temp_dir / "test_video.mp4"),
            "total_frames": 10,
            "fps": 30.0,
            "width": 640,
            "height": 480,
            "detection_settings": {
                "model": "yolov8n.pt",
                "confidence": 0.5
            }
        },
        "frames": []
    }
    
    # Add frames with no people
    for i in range(10):
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
def mock_yolo_model():
    """Create a mock YOLO model for testing."""
    with patch('auto_cropper.person_detector.YOLO') as mock_yolo_class:
        mock_model = Mock()
        mock_yolo_class.return_value = mock_model
        yield mock_model


class TestPersonDetector:
    """Test cases for PersonDetector class."""
    
    def test_init_default(self, mock_yolo_model):
        """Test default initialization."""
        detector = PersonDetector()
        assert detector.model_name == 'yolov8l.pt'
        assert detector.confidence == 0.5
        assert detector.verbose is False
        assert detector.person_class_id == 0
        assert detector.model == mock_yolo_model
    
    def test_init_custom(self, mock_yolo_model):
        """Test custom initialization."""
        detector = PersonDetector(
            model_name='yolov8s.pt',
            confidence=0.7,
            verbose=True
        )
        assert detector.model_name == 'yolov8s.pt'
        assert detector.confidence == 0.7
        assert detector.verbose is True
        assert detector.person_class_id == 0
    
    def test_init_invalid_confidence(self, mock_yolo_model):
        """Test initialization with edge case confidence values."""
        # Test very low confidence
        detector = PersonDetector(confidence=0.0)
        assert detector.confidence == 0.0
        
        # Test very high confidence
        detector = PersonDetector(confidence=1.0)
        assert detector.confidence == 1.0
    
    @pytest.mark.skipif(not os.environ.get('RUN_YOLO_TESTS'), 
                       reason="Skipping YOLO tests - set RUN_YOLO_TESTS=1 to enable")
    def test_detect_people_in_video_real(self, sample_video, temp_dir):
        """Test video detection with real YOLO model (only if YOLO tests enabled)."""
        detector = PersonDetector(verbose=True)
        detection_file = detector.detect_people_in_video(str(sample_video), str(temp_dir))
        
        assert Path(detection_file).exists()
        
        # Check detection file content
        with open(detection_file, 'r') as f:
            data = json.load(f)
        
        assert "video_info" in data
        assert "frames" in data
        assert data["video_info"]["total_frames"] == 30
        assert len(data["frames"]) == 30
    
    def test_detect_people_in_video_mocked(self, mock_yolo_model, sample_video, temp_dir):
        """Test video detection with mocked YOLO model."""
        # Mock the YOLO results
        mock_result = Mock()
        mock_box = Mock()
        mock_box.cls.item.return_value = 0  # Person class
        mock_box.conf.item.return_value = 0.8  # High confidence
        mock_box.xyxy = [np.array([100, 100, 200, 300])]  # Bounding box
        
        mock_boxes = Mock()
        mock_boxes.__iter__ = Mock(return_value=iter([mock_box]))
        mock_result.boxes = mock_boxes
        
        mock_yolo_model.return_value = [mock_result]
        
        detector = PersonDetector(verbose=False)
        detection_file = detector.detect_people_in_video(str(sample_video), str(temp_dir))
        
        assert Path(detection_file).exists()
        
        # Verify the detection data structure
        with open(detection_file, 'r') as f:
            data = json.load(f)
        
        assert "video_info" in data
        assert "frames" in data
        assert data["video_info"]["total_frames"] == 30
        
        # Check that each frame has the expected detection
        for frame_data in data["frames"]:
            assert "frame_number" in frame_data
            assert "timestamp" in frame_data
            assert "people" in frame_data
            if frame_data["people"]:  # If people detected
                person = frame_data["people"][0]
                assert "bbox" in person
                assert "confidence" in person
                assert "center" in person
    
    def test_detect_people_in_video_invalid_file(self, mock_yolo_model, temp_dir):
        """Test video detection with invalid video file."""
        detector = PersonDetector()
        invalid_video = temp_dir / "nonexistent.mp4"
        
        with pytest.raises(ValueError, match="Could not open video file"):
            detector.detect_people_in_video(str(invalid_video), str(temp_dir))
    
    def test_detect_people_in_video_output_directory_creation(self, mock_yolo_model, sample_video, temp_dir):
        """Test that output directory is created if it doesn't exist."""
        new_output_dir = temp_dir / "new_output"
        assert not new_output_dir.exists()
        
        # Mock YOLO to return no detections
        mock_yolo_model.return_value = []
        
        detector = PersonDetector()
        detection_file = detector.detect_people_in_video(str(sample_video), str(new_output_dir))
        
        assert new_output_dir.exists()
        assert Path(detection_file).exists()
    
    def test_detect_people_in_frame_mocked(self, mock_yolo_model):
        """Test single frame detection with mocked YOLO."""
        # Create a mock frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Mock the YOLO results
        mock_result = Mock()
        mock_box = Mock()
        mock_box.cls.item.return_value = 0  # Person class
        mock_box.conf.item.return_value = 0.75
        mock_box.xyxy = [np.array([50, 60, 150, 260])]  # x1, y1, x2, y2
        
        mock_boxes = Mock()
        mock_boxes.__iter__ = Mock(return_value=iter([mock_box]))
        mock_result.boxes = mock_boxes
        
        mock_yolo_model.return_value = [mock_result]
        
        detector = PersonDetector()
        detections = detector._detect_people_in_frame(frame, 0)
        
        assert len(detections) == 1
        detection = detections[0]
        
        assert detection["bbox"]["x1"] == 50
        assert detection["bbox"]["y1"] == 60
        assert detection["bbox"]["x2"] == 150
        assert detection["bbox"]["y2"] == 260
        assert detection["bbox"]["width"] == 100
        assert detection["bbox"]["height"] == 200
        assert detection["confidence"] == 0.75
        assert detection["center"]["x"] == 100  # (50 + 150) // 2
        assert detection["center"]["y"] == 160  # (60 + 260) // 2
    
    def test_detect_people_in_frame_no_detections(self, mock_yolo_model):
        """Test single frame detection with no people."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Mock YOLO to return no detections
        mock_yolo_model.return_value = []
        
        detector = PersonDetector()
        detections = detector._detect_people_in_frame(frame, 0)
        
        assert len(detections) == 0
    
    def test_detect_people_in_frame_wrong_class(self, mock_yolo_model):
        """Test single frame detection with non-person detections."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Mock the YOLO results with non-person class
        mock_result = Mock()
        mock_box = Mock()
        mock_box.cls.item.return_value = 2  # Car class (not person)
        mock_box.conf.item.return_value = 0.9
        mock_box.xyxy = [np.array([50, 60, 150, 260])]
        
        mock_boxes = Mock()
        mock_boxes.__iter__ = Mock(return_value=iter([mock_box]))
        mock_result.boxes = mock_boxes
        
        mock_yolo_model.return_value = [mock_result]
        
        detector = PersonDetector()
        detections = detector._detect_people_in_frame(frame, 0)
        
        assert len(detections) == 0
    
    def test_detect_people_in_frame_low_confidence(self, mock_yolo_model):
        """Test single frame detection with low confidence detections."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Mock the YOLO results with low confidence
        mock_result = Mock()
        mock_box = Mock()
        mock_box.cls.item.return_value = 0  # Person class
        mock_box.conf.item.return_value = 0.3  # Below default threshold of 0.5
        mock_box.xyxy = [np.array([50, 60, 150, 260])]
        
        mock_boxes = Mock()
        mock_boxes.__iter__ = Mock(return_value=iter([mock_box]))
        mock_result.boxes = mock_boxes
        
        mock_yolo_model.return_value = [mock_result]
        
        detector = PersonDetector(confidence=0.5)
        detections = detector._detect_people_in_frame(frame, 0)
        
        assert len(detections) == 0
    
    def test_detect_people_in_frame_multiple_people(self, mock_yolo_model):
        """Test single frame detection with multiple people."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Mock the YOLO results with multiple people
        mock_result = Mock()
        
        mock_box1 = Mock()
        mock_box1.cls.item.return_value = 0
        mock_box1.conf.item.return_value = 0.8
        mock_box1.xyxy = [np.array([50, 60, 150, 260])]
        
        mock_box2 = Mock()
        mock_box2.cls.item.return_value = 0
        mock_box2.conf.item.return_value = 0.9
        mock_box2.xyxy = [np.array([300, 100, 400, 300])]
        
        mock_boxes = Mock()
        mock_boxes.__iter__ = Mock(return_value=iter([mock_box1, mock_box2]))
        mock_result.boxes = mock_boxes
        
        mock_yolo_model.return_value = [mock_result]
        
        detector = PersonDetector()
        detections = detector._detect_people_in_frame(frame, 0)
        
        assert len(detections) == 2
        
        # Check first detection
        assert detections[0]["bbox"]["x1"] == 50
        assert detections[0]["confidence"] == 0.8
        
        # Check second detection
        assert detections[1]["bbox"]["x1"] == 300
        assert detections[1]["confidence"] == 0.9
    
    def test_detect_people_in_frame_no_boxes(self, mock_yolo_model):
        """Test single frame detection when YOLO returns result with no boxes."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Mock YOLO result with no boxes
        mock_result = Mock()
        mock_result.boxes = None
        
        mock_yolo_model.return_value = [mock_result]
        
        detector = PersonDetector()
        detections = detector._detect_people_in_frame(frame, 0)
        
        assert len(detections) == 0
    
    def test_get_detection_summary_with_people(self, sample_detection_data):
        """Test detection summary generation with people detected."""
        detector = PersonDetector()
        summary = detector.get_detection_summary(str(sample_detection_data))
        
        assert summary["total_frames"] == 30
        assert summary["frames_with_people"] == 30
        assert summary["frames_without_people"] == 0
        assert summary["total_detections"] == 30
        assert summary["average_people_per_frame"] == 1.0
        assert summary["max_people_in_frame"] == 1
        assert summary["detection_coverage"] == 100.0
        assert "video_info" in summary
    
    def test_get_detection_summary_empty(self, empty_detection_data):
        """Test detection summary generation with no people detected."""
        detector = PersonDetector()
        summary = detector.get_detection_summary(str(empty_detection_data))
        
        assert summary["total_frames"] == 10
        assert summary["frames_with_people"] == 0
        assert summary["frames_without_people"] == 10
        assert summary["total_detections"] == 0
        assert summary["average_people_per_frame"] == 0.0
        assert summary["max_people_in_frame"] == 0
        assert summary["detection_coverage"] == 0.0
    
    def test_get_detection_summary_invalid_file(self, temp_dir):
        """Test detection summary with invalid file."""
        detector = PersonDetector()
        invalid_file = temp_dir / "nonexistent.json"
        
        with pytest.raises(FileNotFoundError):
            detector.get_detection_summary(str(invalid_file))
    
    def test_get_detection_summary_malformed_json(self, temp_dir):
        """Test detection summary with malformed JSON."""
        detector = PersonDetector()
        malformed_file = temp_dir / "malformed.json"
        
        with open(malformed_file, 'w') as f:
            f.write("invalid json content")
        
        with pytest.raises(json.JSONDecodeError):
            detector.get_detection_summary(str(malformed_file))
    
    def test_detection_output_filename_format(self, mock_yolo_model, sample_video, temp_dir):
        """Test that detection output file has correct naming format."""
        mock_yolo_model.return_value = []
        
        detector = PersonDetector()
        detection_file = detector.detect_people_in_video(str(sample_video), str(temp_dir))
        
        expected_filename = f"{sample_video.stem}_detections.json"
        assert Path(detection_file).name == expected_filename
        assert Path(detection_file).parent == temp_dir
    
    def test_video_info_in_detection_data(self, mock_yolo_model, sample_video, temp_dir):
        """Test that video info is correctly stored in detection data."""
        mock_yolo_model.return_value = []
        
        detector = PersonDetector(model_name='yolov8s.pt', confidence=0.7)
        detection_file = detector.detect_people_in_video(str(sample_video), str(temp_dir))
        
        with open(detection_file, 'r') as f:
            data = json.load(f)
        
        video_info = data["video_info"]
        assert video_info["video_path"] == str(sample_video)
        assert video_info["total_frames"] == 30
        assert video_info["fps"] == 30.0
        assert video_info["width"] == 640
        assert video_info["height"] == 480
        assert video_info["detection_settings"]["model"] == 'yolov8s.pt'
        assert video_info["detection_settings"]["confidence"] == 0.7
    
    def test_frame_timestamp_calculation(self, mock_yolo_model, sample_video, temp_dir):
        """Test that frame timestamps are calculated correctly."""
        mock_yolo_model.return_value = []
        
        detector = PersonDetector()
        detection_file = detector.detect_people_in_video(str(sample_video), str(temp_dir))
        
        with open(detection_file, 'r') as f:
            data = json.load(f)
        
        # Check first few frames
        assert data["frames"][0]["timestamp"] == 0.0
        assert abs(data["frames"][1]["timestamp"] - (1/30.0)) < 0.001
        assert abs(data["frames"][2]["timestamp"] - (2/30.0)) < 0.001
    
    @patch('auto_cropper.person_detector.logging')
    def test_logging_behavior(self, mock_logging, mock_yolo_model):
        """Test logging behavior with verbose mode."""
        # Test with verbose=False
        detector = PersonDetector(verbose=False)
        mock_logging.basicConfig.assert_not_called()
        
        # Test with verbose=True
        detector = PersonDetector(verbose=True)
        mock_logging.basicConfig.assert_called_with(level=mock_logging.INFO)
