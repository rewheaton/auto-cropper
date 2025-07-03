"""Streaming JSON writer for large video processing."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Union

from auto_cropper.exceptions import StreamingWriterException


class StreamingJSONWriter:
    """Write JSON data incrementally to avoid memory issues with large files."""
    
    def __init__(self, filepath: Union[str, Path], video_info: Dict[str, Any]):
        """
        Initialize the streaming JSON writer.
        
        Args:
            filepath: Path to the output JSON file
            video_info: Video metadata to write in the header
            
        Raises:
            OSError: If file cannot be created
            StreamingWriterException: If initialization fails
        """
        self.filepath = Path(filepath)
        self.frame_count = 0
        self.is_closed = False
        self.logger = logging.getLogger(__name__)
        
        try:
            # Create parent directory if it doesn't exist
            self.filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Open file for writing
            self.file = open(self.filepath, 'w')
            
            # Write header with video info
            self.file.write('{"video_info": ')
            json.dump(video_info, self.file)
            self.file.write(', "frames": [')
            self.file.flush()
            
            self.logger.info(f"Streaming JSON writer initialized: {self.filepath}")
            
        except Exception as e:
            raise StreamingWriterException(f"Failed to initialize streaming writer: {e}") from e
    
    def write_frame(self, frame_data: Dict[str, Any]) -> None:
        """
        Write a single frame's data to the JSON file.
        
        Args:
            frame_data: Dictionary containing frame detection data
            
        Raises:
            ValueError: If writer is closed
            StreamingWriterException: If write operation fails
        """
        if self.is_closed:
            raise ValueError("Cannot write to closed writer")
        
        try:
            # Add comma separator for all frames except the first
            if self.frame_count > 0:
                self.file.write(',')
            
            # Write frame data
            json.dump(frame_data, self.file)
            self.file.flush()  # Ensure data is written to disk
            
            self.frame_count += 1
            
        except Exception as e:
            raise StreamingWriterException(f"Failed to write frame {self.frame_count}: {e}") from e
    
    def get_frame_count(self) -> int:
        """
        Get the number of frames written so far.
        
        Returns:
            Number of frames written
        """
        return self.frame_count
    
    def close(self) -> None:
        """
        Close the JSON file properly by writing the closing bracket.
        
        Safe to call multiple times.
        """
        if self.is_closed:
            return
        
        try:
            # Close the frames array and root object
            self.file.write(']}')
            self.file.close()
            self.is_closed = True
            
            self.logger.info(f"Streaming JSON writer closed: {self.frame_count} frames written to {self.filepath}")
            
        except Exception as e:
            self.logger.error(f"Error closing streaming writer: {e}")
            # Still mark as closed to prevent further writes
            self.is_closed = True
            if hasattr(self, 'file') and not self.file.closed:
                try:
                    self.file.close()
                except:
                    pass
    
    def __enter__(self):
        """Enter context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and ensure file is properly closed."""
        self.close()
        return False
    
    def __del__(self):
        """Ensure file is closed when object is garbage collected."""
        if not self.is_closed:
            try:
                self.close()
            except:
                pass  # Don't raise exceptions in destructor
