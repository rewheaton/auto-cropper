"""Tests for CLI error handling and input validation."""

import pytest
import tempfile
import json
import os
from pathlib import Path
from unittest.mock import patch

import click
from click.testing import CliRunner

from auto_cropper.cli import (
    validate_input_file,
    ensure_output_directory, 
    validate_json_file,
    main
)


def test_validate_input_file_success(tmp_path):
    """Test validation with a valid video file."""
    video_file = tmp_path / "test.mp4"
    video_file.write_bytes(b"fake video content")
    
    result = validate_input_file(str(video_file))
    assert result == video_file
    assert result.exists()


def test_validate_input_file_not_exists():
    """Test validation with non-existent file."""
    with pytest.raises(click.ClickException) as exc_info:
        validate_input_file("/nonexistent/path/video.mp4")
    
    assert "Input file does not exist" in str(exc_info.value)


def test_validate_input_file_is_directory(tmp_path):
    """Test validation when path points to a directory."""
    test_dir = tmp_path / "not_a_file"
    test_dir.mkdir()
    
    with pytest.raises(click.ClickException) as exc_info:
        validate_input_file(str(test_dir))
    
    assert "Path is not a file" in str(exc_info.value)


def test_validate_input_file_unsupported_format(tmp_path):
    """Test validation with unsupported file format."""
    text_file = tmp_path / "test.txt"
    text_file.write_text("not a video")
    
    with pytest.raises(click.ClickException) as exc_info:
        validate_input_file(str(text_file))
    
    assert "Unsupported file format" in str(exc_info.value)
    assert ".txt" in str(exc_info.value)


def test_validate_input_file_custom_extensions(tmp_path):
    """Test validation with custom allowed extensions."""
    custom_file = tmp_path / "test.custom"
    custom_file.write_bytes(b"fake content")
    
    # Should fail with default extensions
    with pytest.raises(click.ClickException):
        validate_input_file(str(custom_file))
    
    # Should pass with custom extensions
    result = validate_input_file(str(custom_file), ['.custom'])
    assert result == custom_file


def test_validate_input_file_case_insensitive(tmp_path):
    """Test that file extension validation is case insensitive."""
    video_file = tmp_path / "test.MP4"
    video_file.write_bytes(b"fake video content")
    
    result = validate_input_file(str(video_file))
    assert result == video_file


def test_ensure_output_directory_create_new(tmp_path):
    """Test creating a new output directory."""
    new_dir = tmp_path / "new_output"
    
    result = ensure_output_directory(str(new_dir))
    assert result == new_dir
    assert new_dir.exists()
    assert new_dir.is_dir()


def test_ensure_output_directory_nested(tmp_path):
    """Test creating nested directories."""
    nested_dir = tmp_path / "level1" / "level2" / "output"
    
    result = ensure_output_directory(str(nested_dir))
    assert result == nested_dir
    assert nested_dir.exists()
    assert nested_dir.is_dir()


def test_ensure_output_directory_existing(tmp_path):
    """Test with an existing directory."""
    existing_dir = tmp_path / "existing"
    existing_dir.mkdir()
    
    result = ensure_output_directory(str(existing_dir))
    assert result == existing_dir
    assert existing_dir.exists()


def test_validate_json_file_success(tmp_path):
    """Test validation with a valid JSON file."""
    json_file = tmp_path / "test.json"
    test_data = {"key1": "value1", "key2": {"nested": "value"}}
    
    with open(json_file, 'w') as f:
        json.dump(test_data, f)
    
    result = validate_json_file(str(json_file))
    assert result == json_file


def test_validate_json_file_not_exists():
    """Test validation with non-existent JSON file."""
    with pytest.raises(click.ClickException) as exc_info:
        validate_json_file("/nonexistent/file.json")
    
    assert "JSON file does not exist" in str(exc_info.value)


def test_validate_json_file_is_directory(tmp_path):
    """Test validation when path points to a directory."""
    test_dir = tmp_path / "not_a_file"
    test_dir.mkdir()
    
    with pytest.raises(click.ClickException) as exc_info:
        validate_json_file(str(test_dir))
    
    assert "Path is not a file" in str(exc_info.value)


def test_validate_json_file_wrong_extension(tmp_path):
    """Test validation with non-JSON file extension."""
    text_file = tmp_path / "test.txt"
    text_file.write_text('{"key": "value"}')  # Valid JSON content, wrong extension
    
    with pytest.raises(click.ClickException) as exc_info:
        validate_json_file(str(text_file))
    
    assert "File must be a JSON file" in str(exc_info.value)


def test_validate_json_file_invalid_content(tmp_path):
    """Test validation with invalid JSON content."""
    json_file = tmp_path / "invalid.json"
    json_file.write_text('{"invalid": json content}')  # Invalid JSON
    
    with pytest.raises(click.ClickException) as exc_info:
        validate_json_file(str(json_file))
    
    assert "Invalid JSON file" in str(exc_info.value)


def test_validate_json_file_expected_keys(tmp_path):
    """Test validation of expected JSON keys."""
    json_file = tmp_path / "test.json"
    test_data = {"video_info": {}, "frames": []}
    
    with open(json_file, 'w') as f:
        json.dump(test_data, f)
    
    # Should pass with correct keys
    result = validate_json_file(str(json_file), ['video_info', 'frames'])
    assert result == json_file
    
    # Should fail with missing keys
    with pytest.raises(click.ClickException) as exc_info:
        validate_json_file(str(json_file), ['video_info', 'frames', 'missing_key'])
    
    assert "missing required keys" in str(exc_info.value)
    assert "missing_key" in str(exc_info.value)


def test_cli_detect_invalid_video():
    """Test detect command with invalid video file."""
    runner = CliRunner()
    
    result = runner.invoke(main, ['detect', '/nonexistent/video.mp4'])
    assert result.exit_code != 0
    assert "Input file does not exist" in result.output


def test_cli_detect_invalid_format(tmp_path):
    """Test detect command with invalid file format."""
    runner = CliRunner()
    
    text_file = tmp_path / "not_video.txt"
    text_file.write_text("This is not a video")
    
    result = runner.invoke(main, ['detect', str(text_file)])
    assert result.exit_code != 0
    assert "Unsupported file format" in result.output


def test_cli_track_invalid_json():
    """Test track command with invalid JSON file."""
    runner = CliRunner()
    
    result = runner.invoke(main, ['track', '/nonexistent/detections.json'])
    assert result.exit_code != 0
    assert "JSON file does not exist" in result.output


def test_cli_track_missing_keys(tmp_path):
    """Test track command with JSON missing required keys."""
    runner = CliRunner()
    
    incomplete_json = tmp_path / "incomplete.json"
    test_data = {"some_key": "value"}  # Missing video_info and frames
    
    with open(incomplete_json, 'w') as f:
        json.dump(test_data, f)
    
    result = runner.invoke(main, ['track', str(incomplete_json)])
    assert result.exit_code != 0
    assert "missing required keys" in result.output


@patch('auto_cropper.cli.VideoCropper')
@patch('auto_cropper.cli.PersonTracker')  
@patch('auto_cropper.cli.PersonDetector')
def test_cli_process_command_calls_all_functions(mock_detector, mock_tracker, mock_cropper, tmp_path):
    """Test that the process command calls detect, track, and crop in sequence."""
    runner = CliRunner()
    
    # Create test video file
    video_file = tmp_path / "test_video.mp4"
    video_file.write_bytes(b"fake video content")
    
    # Create expected output directory
    output_dir = tmp_path / "output"
    
    # Mock detection results
    detection_file = tmp_path / "test_video_detections.json"
    detection_data = {
        "video_info": {"video_path": str(video_file), "total_frames": 10},
        "frames": [{"frame_number": 0, "people": []}]
    }
    
    with open(detection_file, 'w') as f:
        json.dump(detection_data, f)
    
    # Mock tracking results  
    tracking_file = tmp_path / "test_video_tracking.json"
    tracking_data = {
        "video_info": {"video_path": str(video_file)},
        "tracking_info": {"total_tracked_frames": 5},
        "tracked_person": []
    }
    
    with open(tracking_file, 'w') as f:
        json.dump(tracking_data, f)
    
    # Configure mocks
    mock_detector_instance = mock_detector.return_value
    mock_detector_instance.detect_people_in_video.return_value = str(detection_file)
    mock_detector_instance.get_detection_summary.return_value = {
        "total_frames": 10,
        "frames_with_people": 5,
        "detection_coverage": 50.0,
        "total_detections": 5,
        "average_people_per_frame": 0.5,
        "max_people_in_frame": 1
    }
    
    mock_tracker_instance = mock_tracker.return_value
    mock_tracker_instance.select_person_to_track.return_value = str(tracking_file)
    
    mock_cropper_instance = mock_cropper.return_value
    cropped_video = tmp_path / "test_video_cropped.mp4"
    mock_cropper_instance.crop_video.return_value = str(cropped_video)
    
    # Run the process command with input='\n' to automatically answer 'n' to delete prompt
    result = runner.invoke(main, ['process', str(video_file), '--output-dir', str(output_dir)], input='\n')
    
    # Verify command succeeded
    assert result.exit_code == 0
    
    # Verify all components were instantiated
    mock_detector.assert_called_once()
    mock_tracker.assert_called_once()
    mock_cropper.assert_called_once()
    
    # Verify detection was called
    mock_detector_instance.detect_people_in_video.assert_called_once_with(str(video_file), str(output_dir))
    mock_detector_instance.get_detection_summary.assert_called_once_with(str(detection_file))
    
    # Verify tracking was called with detection results
    mock_tracker_instance.select_person_to_track.assert_called_once_with(str(detection_file))
    
    # Verify cropping was called with video and tracking data
    mock_cropper_instance.crop_video.assert_called_once_with(str(video_file), str(tracking_file), output_dir=str(output_dir), duration_limit=None)
    
    # Verify output messages
    assert "Detection complete!" in result.output
    assert "Tracking complete!" in result.output  
    assert "Video cropping complete!" in result.output
    assert "Pipeline complete!" in result.output


@patch('auto_cropper.cli.PersonDetector')
def test_cli_process_command_handles_detection_failure(mock_detector, tmp_path):
    """Test that process command handles detection failure gracefully."""
    runner = CliRunner()
    
    # Create test video file
    video_file = tmp_path / "test_video.mp4"
    video_file.write_bytes(b"fake video content")
    
    output_dir = tmp_path / "output"
    
    # Mock detection to raise an exception
    mock_detector_instance = mock_detector.return_value
    mock_detector_instance.detect_people_in_video.side_effect = Exception("Detection failed")
    
    # Run the process command
    result = runner.invoke(main, ['process', str(video_file), '--output-dir', str(output_dir)])
    
    # Verify command failed gracefully
    assert result.exit_code == 1
    assert "Error during processing" in result.output
    assert "Detection failed" in result.output


@patch('auto_cropper.cli.VideoCropper')
@patch('auto_cropper.cli.PersonTracker')  
@patch('auto_cropper.cli.PersonDetector')
def test_cli_process_command_handles_tracking_failure(mock_detector, mock_tracker, mock_cropper, tmp_path):
    """Test that process command handles tracking failure gracefully."""
    runner = CliRunner()
    
    # Create test video file
    video_file = tmp_path / "test_video.mp4"
    video_file.write_bytes(b"fake video content")
    
    output_dir = tmp_path / "output"
    
    # Mock successful detection
    detection_file = tmp_path / "test_video_detections.json" 
    mock_detector_instance = mock_detector.return_value
    mock_detector_instance.detect_people_in_video.return_value = str(detection_file)
    mock_detector_instance.get_detection_summary.return_value = {
        "total_frames": 10, "frames_with_people": 5, "detection_coverage": 50.0,
        "total_detections": 5, "average_people_per_frame": 0.5, "max_people_in_frame": 1
    }
    
    # Mock tracking to raise an exception
    mock_tracker_instance = mock_tracker.return_value
    mock_tracker_instance.select_person_to_track.side_effect = Exception("Tracking failed")
    
    # Run the process command
    result = runner.invoke(main, ['process', str(video_file), '--output-dir', str(output_dir)])
    
    # Verify command failed gracefully
    assert result.exit_code == 1
    assert "Error during processing" in result.output
    assert "Tracking failed" in result.output
    
    # Verify detection was called but tracking failed before cropping
    mock_detector_instance.detect_people_in_video.assert_called_once()
    mock_tracker_instance.select_person_to_track.assert_called_once()
    mock_cropper.assert_not_called()  # Should not reach cropping
