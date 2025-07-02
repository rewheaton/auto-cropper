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
