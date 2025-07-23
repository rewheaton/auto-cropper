"""Tests for batch processing functionality in CLI."""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import click

from auto_cropper.cli import find_video_files, process_batch, process_single_file, main


class TestFindVideoFiles:
    """Test cases for find_video_files function."""
    
    def test_find_video_files_success(self, tmp_path):
        """Test finding video files in a directory with compatible files."""
        # Create test video files
        video_files = [
            tmp_path / "video1.mp4",
            tmp_path / "video2.avi", 
            tmp_path / "video3.mov",
            tmp_path / "video4.mkv"
        ]
        
        for video_file in video_files:
            video_file.touch()
        
        # Create non-video files that should be ignored
        (tmp_path / "document.txt").touch()
        (tmp_path / "image.jpg").touch()
        
        result = find_video_files(str(tmp_path))
        
        assert len(result) == 4
        assert all(isinstance(f, Path) for f in result)
        assert {f.name for f in result} == {"video1.mp4", "video2.avi", "video3.mov", "video4.mkv"}
        
        # Check that results are sorted
        result_names = [f.name for f in result]
        assert result_names == sorted(result_names)
    
    def test_find_video_files_custom_extensions(self, tmp_path):
        """Test finding video files with custom allowed extensions."""
        # Create test files
        (tmp_path / "video1.mp4").touch()
        (tmp_path / "video2.avi").touch()
        (tmp_path / "video3.mov").touch()
        
        # Only allow .mp4 and .mov
        result = find_video_files(str(tmp_path), ['.mp4', '.mov'])
        
        assert len(result) == 2
        assert {f.name for f in result} == {"video1.mp4", "video3.mov"}
    
    def test_find_video_files_case_insensitive(self, tmp_path):
        """Test that file extension matching is case insensitive."""
        # Create files with different case extensions
        (tmp_path / "video1.MP4").touch()
        (tmp_path / "video2.Mp4").touch()
        (tmp_path / "video3.AVI").touch()
        
        result = find_video_files(str(tmp_path))
        
        assert len(result) == 3
        assert {f.name for f in result} == {"video1.MP4", "video2.Mp4", "video3.AVI"}
    
    def test_find_video_files_empty_directory(self, tmp_path):
        """Test finding video files in an empty directory."""
        result = find_video_files(str(tmp_path))
        assert result == []
    
    def test_find_video_files_no_compatible_files(self, tmp_path):
        """Test finding video files when no compatible files exist."""
        # Create non-video files
        (tmp_path / "document.txt").touch()
        (tmp_path / "image.jpg").touch()
        (tmp_path / "data.json").touch()
        
        result = find_video_files(str(tmp_path))
        assert result == []
    
    def test_find_video_files_directory_not_exists(self):
        """Test error when directory doesn't exist."""
        with pytest.raises(click.ClickException) as exc_info:
            find_video_files("/nonexistent/directory")
        
        assert "Directory does not exist" in str(exc_info.value)
    
    def test_find_video_files_path_is_file(self, tmp_path):
        """Test error when path is a file, not a directory."""
        test_file = tmp_path / "test.txt"
        test_file.touch()
        
        with pytest.raises(click.ClickException) as exc_info:
            find_video_files(str(test_file))
        
        assert "Path is not a directory" in str(exc_info.value)
    
    def test_find_video_files_subdirectories_ignored(self, tmp_path):
        """Test that subdirectories are ignored, only files are considered."""
        # Create video files
        (tmp_path / "video1.mp4").touch()
        
        # Create subdirectory with video file inside
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "video2.mp4").touch()
        
        result = find_video_files(str(tmp_path))
        
        # Should only find the file in the root directory, not in subdirectory
        assert len(result) == 1
        assert result[0].name == "video1.mp4"


class TestProcessSingleFile:
    """Test cases for process_single_file function."""
    
    @patch('auto_cropper.cli.VideoCropper')
    @patch('auto_cropper.cli.PersonTracker')
    @patch('auto_cropper.cli.PersonDetector')
    @patch('auto_cropper.cli.click.echo')
    def test_process_single_file_success(self, mock_echo, mock_detector_class, 
                                       mock_tracker_class, mock_cropper_class, tmp_path):
        """Test successful processing of a single file."""
        # Setup mocks
        mock_detector = Mock()
        mock_detector_class.return_value = mock_detector
        mock_detector.detect_people_in_video.return_value = "detection_file.json"
        mock_detector.get_detection_summary.return_value = {"detection_coverage": 85}
        
        mock_tracker = Mock()
        mock_tracker_class.return_value = mock_tracker
        mock_tracker.select_person_to_track.return_value = "tracking_file.json"
        
        mock_cropper = Mock()
        mock_cropper_class.return_value = mock_cropper
        mock_cropper.crop_video.return_value = "output_video.mp4"
        
        # Test parameters
        video_path = tmp_path / "test_video.mp4"
        video_path.touch()
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        # Call function
        result = process_single_file(
            video_path=video_path,
            output_dir=output_dir, 
            confidence=0.7,
            margin=50,
            smoothing=10,
            duration=None,
            verbose=True
        )
        
        # Verify results
        detection_file, tracking_file, output_path = result
        assert detection_file == "detection_file.json"
        assert tracking_file == "tracking_file.json"
        assert output_path == "output_video.mp4"
        
        # Verify function calls
        mock_detector_class.assert_called_once_with(
            model_name='yolov8l.pt', confidence=0.7, verbose=True
        )
        mock_detector.detect_people_in_video.assert_called_once_with(
            str(video_path), str(output_dir)
        )
        mock_tracker.select_person_to_track.assert_called_once_with("detection_file.json")
        mock_cropper.crop_video.assert_called_once_with(
            str(video_path), "tracking_file.json", output_dir=str(output_dir), duration_limit=None
        )


class TestProcessBatch:
    """Test cases for process_batch function."""
    
    @patch('auto_cropper.cli.process_single_file')
    @patch('auto_cropper.cli.click.echo')
    @patch('auto_cropper.cli.click.confirm')
    def test_process_batch_all_success(self, mock_confirm, mock_echo, mock_process_single, tmp_path):
        """Test batch processing when all files succeed."""
        # Setup
        video_files = [
            tmp_path / "video1.mp4",
            tmp_path / "video2.avi"
        ]
        for video_file in video_files:
            video_file.touch()
        
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        # Mock successful processing
        mock_process_single.return_value = ("det.json", "track.json", "out.mp4")
        mock_confirm.return_value = False
        
        # Call function
        process_batch(
            video_files=video_files,
            output_dir=output_dir,
            confidence=0.5,
            margin=50,
            smoothing=10,
            duration=None,
            verbose=False
        )
        
        # Verify process_single_file was called for each video
        assert mock_process_single.call_count == 2
        
        # Verify output directories were created
        assert (output_dir / "video1").exists()
        assert (output_dir / "video2").exists()
        
        # Verify success messages were echoed
        echo_calls = [call[0][0] for call in mock_echo.call_args_list]
        assert any("Starting batch processing of 2 video files" in call for call in echo_calls)
        assert any("✓ Successfully processed: video1.mp4" in call for call in echo_calls)
        assert any("✓ Successfully processed: video2.avi" in call for call in echo_calls)
        assert any("Total files: 2" in call for call in echo_calls)
        assert any("Successful: 2" in call for call in echo_calls)
        assert any("Failed: 0" in call for call in echo_calls)
    
    @patch('auto_cropper.cli.process_single_file')
    @patch('auto_cropper.cli.click.echo')
    @patch('auto_cropper.cli.click.confirm')
    def test_process_batch_partial_failure(self, mock_confirm, mock_echo, mock_process_single, tmp_path):
        """Test batch processing when some files fail."""
        # Setup
        video_files = [
            tmp_path / "video1.mp4",
            tmp_path / "video2.avi",
            tmp_path / "video3.mov"
        ]
        for video_file in video_files:
            video_file.touch()
        
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        # Mock processing - first succeeds, second fails, third succeeds
        def side_effect(video_path, *args, **kwargs):
            if "video2" in str(video_path):
                raise Exception("Processing failed for video2")
            return ("det.json", "track.json", "out.mp4")
        
        mock_process_single.side_effect = side_effect
        mock_confirm.return_value = False
        
        # Call function
        process_batch(
            video_files=video_files,
            output_dir=output_dir,
            confidence=0.5,
            margin=50,
            smoothing=10,
            duration=None,
            verbose=False
        )
        
        # Verify process_single_file was called for each video
        assert mock_process_single.call_count == 3
        
        # Verify output directories
        assert (output_dir / "video1").exists()
        assert (output_dir / "video2").exists()  # Directory created even if processing failed
        assert (output_dir / "video3").exists()
        
        # Verify summary shows correct counts
        echo_calls = [call[0][0] for call in mock_echo.call_args_list]
        assert any("Total files: 3" in call for call in echo_calls)
        assert any("Successful: 2" in call for call in echo_calls)
        assert any("Failed: 1" in call for call in echo_calls)
        assert any("✗ Failed to process video2.avi" in call for call in echo_calls)
    
    @patch('auto_cropper.cli.process_single_file')
    @patch('auto_cropper.cli.click.echo')
    @patch('auto_cropper.cli.click.confirm')
    def test_process_batch_cleanup(self, mock_confirm, mock_echo, mock_process_single, tmp_path):
        """Test batch processing cleanup functionality."""
        # Setup
        video_files = [tmp_path / "video1.mp4"]
        video_files[0].touch()
        
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        # Mock successful processing
        mock_process_single.return_value = ("det.json", "track.json", "out.mp4")
        mock_confirm.return_value = True  # User chooses to delete intermediate files
        
        # Create some JSON files to be cleaned up
        video_output_dir = output_dir / "video1"
        video_output_dir.mkdir()
        (video_output_dir / "video1_detections.json").touch()
        (video_output_dir / "video1_tracking.json").touch()
        (video_output_dir / "video1_cropped.mp4").touch()  # Should not be deleted
        
        # Call function
        process_batch(
            video_files=video_files,
            output_dir=output_dir,
            confidence=0.5,
            margin=50,
            smoothing=10,
            duration=None,
            verbose=False
        )
        
        # Verify cleanup was performed
        assert not (video_output_dir / "video1_detections.json").exists()
        assert not (video_output_dir / "video1_tracking.json").exists()
        assert (video_output_dir / "video1_cropped.mp4").exists()  # Video should remain
        
        # Verify cleanup message
        echo_calls = [call[0][0] for call in mock_echo.call_args_list]
        assert any("Deleted 2 intermediate files" in call for call in echo_calls)


class TestProcessCommandIntegration:
    """Integration tests for the process command with batch functionality."""
    
    @patch('auto_cropper.cli.process_batch')
    @patch('auto_cropper.cli.find_video_files')
    @patch('auto_cropper.cli.ensure_output_directory')
    def test_process_command_batch_mode(self, mock_ensure_output, mock_find_videos, mock_process_batch, tmp_path):
        """Test process command in batch mode."""
        from click.testing import CliRunner
        
        # Create a real input directory
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        
        # Setup mocks
        mock_ensure_output.return_value = Path("/output")
        mock_find_videos.return_value = [Path("video1.mp4"), Path("video2.mp4")]
        
        runner = CliRunner()
        result = runner.invoke(main, [
            'process', 
            '--input-directory', str(input_dir),
            '--confidence', '0.7',
            '--margin', '75'
        ])
        
        assert result.exit_code == 0
        mock_find_videos.assert_called_once_with(str(input_dir))
        mock_process_batch.assert_called_once()
        
        # Verify process_batch was called with correct parameters
        call_args = mock_process_batch.call_args
        args, kwargs = call_args
        assert len(args[0]) == 2  # Two video files
        assert args[2] == 0.7  # confidence
        assert args[3] == 75   # margin
    
    @patch('auto_cropper.cli.process_single_file')
    @patch('auto_cropper.cli.validate_input_file')
    @patch('auto_cropper.cli.ensure_output_directory')
    @patch('auto_cropper.cli.click.confirm')
    def test_process_command_single_mode(self, mock_confirm, mock_ensure_output, 
                                       mock_validate_input, mock_process_single):
        """Test process command in single file mode (regression test)."""
        from click.testing import CliRunner
        
        # Setup mocks
        mock_ensure_output.return_value = Path("/output")
        mock_validate_input.return_value = Path("video.mp4")
        mock_process_single.return_value = ("det.json", "track.json", "out.mp4")
        mock_confirm.return_value = False
        
        runner = CliRunner()
        result = runner.invoke(main, [
            'process', 
            'video.mp4',
            '--confidence', '0.6'
        ])
        
        assert result.exit_code == 0
        mock_validate_input.assert_called_once_with('video.mp4')
        mock_process_single.assert_called_once()
        
        # Verify process_single_file was called with correct parameters
        call_args = mock_process_single.call_args
        args, kwargs = call_args
        # Check positional arguments
        assert args[2] == 0.6  # confidence is the 3rd positional argument
    
    def test_process_command_no_input_error(self):
        """Test error when neither video_path nor input_directory is provided."""
        from click.testing import CliRunner
        
        runner = CliRunner()
        result = runner.invoke(main, ['process'])
        
        assert result.exit_code != 0
        assert "Must provide either VIDEO_PATH or --input-directory" in result.output
    
    def test_process_command_both_inputs_error(self):
        """Test error when both video_path and input_directory are provided."""
        from click.testing import CliRunner
        
        runner = CliRunner()
        result = runner.invoke(main, [
            'process', 
            'video.mp4',
            '--input-directory', '/input'
        ])
        
        assert result.exit_code != 0
        assert "Cannot specify both VIDEO_PATH and --input-directory" in result.output
    
    @patch('auto_cropper.cli.find_video_files')
    @patch('auto_cropper.cli.ensure_output_directory')
    def test_process_command_no_compatible_files_error(self, mock_ensure_output, mock_find_videos, tmp_path):
        """Test error when no compatible files found in directory."""
        from click.testing import CliRunner
        
        # Create a real input directory
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        
        # Setup mocks
        mock_ensure_output.return_value = Path("/output")
        mock_find_videos.return_value = []  # No compatible files
        
        runner = CliRunner()
        result = runner.invoke(main, [
            'process',
            '--input-directory', str(input_dir)
        ])
        
        assert result.exit_code != 0
        assert "No compatible video files found in directory" in result.output


class TestProcessCommandHelp:
    """Test help output for process command."""
    
    def test_process_command_help_shows_new_option(self):
        """Test that help output includes the new input-directory option."""
        from click.testing import CliRunner
        
        runner = CliRunner()
        result = runner.invoke(main, ['process', '--help'])
        
        assert result.exit_code == 0
        assert '--input-directory' in result.output
        assert '-id' in result.output
        assert 'Process all compatible video files in this' in result.output
        assert 'VIDEO_PATH: Path to the input video file (required unless --input-directory is' in result.output


if __name__ == '__main__':
    pytest.main([__file__])
