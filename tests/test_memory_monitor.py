"""Tests for the MemoryMonitor class."""

import pytest
from unittest.mock import Mock, patch
from auto_cropper.memory_monitor import MemoryMonitor


class TestMemoryMonitor:
    """Test cases for MemoryMonitor class."""
    
    def test_init_default(self):
        """Test MemoryMonitor initialization with default values."""
        monitor = MemoryMonitor()
        assert monitor.max_memory_mb == 2048
        assert monitor.buffer_ratio == 0.8
    
    def test_init_custom(self):
        """Test MemoryMonitor initialization with custom values."""
        monitor = MemoryMonitor(max_memory_mb=4096, buffer_ratio=0.9)
        assert monitor.max_memory_mb == 4096
        assert monitor.buffer_ratio == 0.9
    
    def test_init_invalid_memory_limit(self):
        """Test MemoryMonitor with invalid memory limit."""
        with pytest.raises(ValueError, match="Memory limit must be positive"):
            MemoryMonitor(max_memory_mb=0)
        
        with pytest.raises(ValueError, match="Memory limit must be positive"):
            MemoryMonitor(max_memory_mb=-100)
    
    def test_init_invalid_buffer_ratio(self):
        """Test MemoryMonitor with invalid buffer ratio."""
        with pytest.raises(ValueError, match="Buffer ratio must be between 0 and 1"):
            MemoryMonitor(buffer_ratio=-0.1)
        
        with pytest.raises(ValueError, match="Buffer ratio must be between 0 and 1"):
            MemoryMonitor(buffer_ratio=1.1)
    
    @patch('psutil.Process')
    def test_get_current_memory_usage(self, mock_process):
        """Test getting current memory usage."""
        # Mock the memory info
        mock_memory_info = Mock()
        mock_memory_info.rss = 1024 * 1024 * 1024  # 1GB in bytes
        mock_process.return_value.memory_info.return_value = mock_memory_info
        
        monitor = MemoryMonitor()
        memory_mb = monitor.get_current_memory_usage()
        
        assert memory_mb == 1024.0  # 1GB in MB
        mock_process.assert_called_once()
    
    @patch('psutil.Process')
    def test_check_memory_usage_within_limit(self, mock_process):
        """Test memory usage check when within limit."""
        # Mock 1GB memory usage
        mock_memory_info = Mock()
        mock_memory_info.rss = 1024 * 1024 * 1024  # 1GB in bytes
        mock_process.return_value.memory_info.return_value = mock_memory_info
        
        monitor = MemoryMonitor(max_memory_mb=2048)  # 2GB limit
        
        assert monitor.check_memory_usage() is True
    
    @patch('psutil.Process')
    def test_check_memory_usage_exceeds_limit(self, mock_process):
        """Test memory usage check when exceeding limit."""
        # Mock 3GB memory usage
        mock_memory_info = Mock()
        mock_memory_info.rss = 3 * 1024 * 1024 * 1024  # 3GB in bytes
        mock_process.return_value.memory_info.return_value = mock_memory_info
        
        monitor = MemoryMonitor(max_memory_mb=2048)  # 2GB limit
        
        assert monitor.check_memory_usage() is False
    
    def test_get_recommended_batch_size(self):
        """Test batch size calculation."""
        monitor = MemoryMonitor(max_memory_mb=2048, buffer_ratio=0.8)
        
        # Available memory: 2048 * 0.8 = 1638.4 MB
        # Frame size: 100 MB
        # Expected batch size: 1638.4 / 100 = 16.384 -> 16
        batch_size = monitor.get_recommended_batch_size(frame_size_mb=100)
        assert batch_size == 16
    
    def test_get_recommended_batch_size_minimum(self):
        """Test batch size calculation with minimum of 1."""
        monitor = MemoryMonitor(max_memory_mb=100, buffer_ratio=0.8)
        
        # Available memory: 100 * 0.8 = 80 MB
        # Frame size: 200 MB (larger than available)
        # Expected batch size: max(1, 80 / 200) = 1
        batch_size = monitor.get_recommended_batch_size(frame_size_mb=200)
        assert batch_size == 1
    
    def test_get_recommended_batch_size_zero_frame_size(self):
        """Test batch size calculation with zero frame size."""
        monitor = MemoryMonitor()
        
        with pytest.raises(ValueError, match="Frame size must be positive"):
            monitor.get_recommended_batch_size(frame_size_mb=0)
    
    def test_get_recommended_batch_size_negative_frame_size(self):
        """Test batch size calculation with negative frame size."""
        monitor = MemoryMonitor()
        
        with pytest.raises(ValueError, match="Frame size must be positive"):
            monitor.get_recommended_batch_size(frame_size_mb=-10)
    
    def test_get_available_memory(self):
        """Test available memory calculation."""
        monitor = MemoryMonitor(max_memory_mb=2048, buffer_ratio=0.8)
        
        available = monitor.get_available_memory()
        assert available == 1638.4  # 2048 * 0.8
    
    @patch('psutil.Process')
    def test_get_memory_status(self, mock_process):
        """Test getting comprehensive memory status."""
        # Mock 1GB memory usage
        mock_memory_info = Mock()
        mock_memory_info.rss = 1024 * 1024 * 1024  # 1GB in bytes
        mock_process.return_value.memory_info.return_value = mock_memory_info
        
        monitor = MemoryMonitor(max_memory_mb=2048)
        status = monitor.get_memory_status()
        
        assert status['current_usage_mb'] == 1024.0
        assert status['max_memory_mb'] == 2048
        assert status['available_mb'] == 1638.4  # 2048 * 0.8
        assert status['usage_percentage'] == 50.0  # 1024 / 2048 * 100
        assert status['within_limit'] is True
    
    @patch('psutil.Process')
    def test_memory_monitor_context_manager(self, mock_process):
        """Test MemoryMonitor as context manager."""
        # Mock memory info
        mock_memory_info = Mock()
        mock_memory_info.rss = 1024 * 1024 * 1024  # 1GB in bytes
        mock_process.return_value.memory_info.return_value = mock_memory_info
        
        monitor = MemoryMonitor(max_memory_mb=2048)
        
        with monitor:
            # Test that context manager works
            assert monitor.check_memory_usage() is True
    
    @patch('psutil.Process')
    def test_memory_monitor_exception_handling(self, mock_process):
        """Test MemoryMonitor handles psutil exceptions gracefully."""
        # Mock psutil exception
        mock_process.side_effect = Exception("psutil error")
        
        monitor = MemoryMonitor()
        
        # Should not raise exception but return safe defaults
        assert monitor.get_current_memory_usage() == 0.0
        assert monitor.check_memory_usage() is False
