"""Memory monitoring utilities for large video processing."""

import logging
from typing import Dict, Optional

try:
    import psutil
except ImportError:
    psutil = None


class MemoryMonitor:
    """Monitor and manage memory usage during video processing."""
    
    def __init__(self, max_memory_mb: int = 2048, buffer_ratio: float = 0.8):
        """
        Initialize the memory monitor.
        
        Args:
            max_memory_mb: Maximum memory limit in MB
            buffer_ratio: Safety buffer ratio (0-1), leaves this much memory free
            
        Raises:
            ValueError: If parameters are invalid
        """
        if max_memory_mb <= 0:
            raise ValueError("Memory limit must be positive")
        
        if not 0 <= buffer_ratio <= 1:
            raise ValueError("Buffer ratio must be between 0 and 1")
        
        self.max_memory_mb = max_memory_mb
        self.buffer_ratio = buffer_ratio
        self.logger = logging.getLogger(__name__)
        
        # Warn if psutil is not available
        if psutil is None:
            self.logger.warning("psutil not available. Memory monitoring will be disabled.")
    
    def get_current_memory_usage(self) -> float:
        """
        Get current memory usage in MB.
        
        Returns:
            Current memory usage in MB, or 0.0 if unable to determine
        """
        if psutil is None:
            return 0.0
        
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / 1024 / 1024  # Convert bytes to MB
        except Exception as e:
            self.logger.warning(f"Unable to get memory usage: {e}")
            return 0.0
    
    def check_memory_usage(self) -> bool:
        """
        Check if current memory usage is within limits.
        
        Returns:
            True if memory usage is within limits, False otherwise
        """
        if psutil is None:
            return False  # Conservative approach when monitoring unavailable
        
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            current_usage = memory_info.rss / 1024 / 1024  # Convert bytes to MB
            return current_usage < self.max_memory_mb
        except Exception as e:
            self.logger.warning(f"Unable to check memory usage: {e}")
            return False  # Conservative approach when monitoring fails
    
    def get_available_memory(self) -> float:
        """
        Get available memory considering buffer ratio.
        
        Returns:
            Available memory in MB
        """
        return self.max_memory_mb * self.buffer_ratio
    
    def get_recommended_batch_size(self, frame_size_mb: float) -> int:
        """
        Calculate optimal batch size based on available memory.
        
        Args:
            frame_size_mb: Estimated size of a single frame in MB
            
        Returns:
            Recommended batch size (minimum 1)
            
        Raises:
            ValueError: If frame size is invalid
        """
        if frame_size_mb <= 0:
            raise ValueError("Frame size must be positive")
        
        available_mb = self.get_available_memory()
        batch_size = int(available_mb / frame_size_mb)
        
        return max(1, batch_size)
    
    def get_memory_status(self) -> Dict[str, float]:
        """
        Get comprehensive memory status information.
        
        Returns:
            Dictionary containing memory status information
        """
        current_usage = self.get_current_memory_usage()
        available = self.get_available_memory()
        usage_percentage = (current_usage / self.max_memory_mb * 100) if self.max_memory_mb > 0 else 0
        
        return {
            'current_usage_mb': current_usage,
            'max_memory_mb': self.max_memory_mb,
            'available_mb': available,
            'usage_percentage': usage_percentage,
            'within_limit': current_usage < self.max_memory_mb
        }
    
    def __enter__(self):
        """Enter context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        # Log final memory status
        if exc_type is None:
            status = self.get_memory_status()
            self.logger.info(f"Memory monitor exit: {status['current_usage_mb']:.1f}MB used "
                           f"({status['usage_percentage']:.1f}%)")
        return False
