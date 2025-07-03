"""Exception classes for large video file processing."""


class LargeFileException(Exception):
    """Base exception for large file processing issues."""
    pass


class MemoryLimitException(LargeFileException):
    """Raised when memory limit is exceeded during processing."""
    
    def __init__(self, current_usage_mb: float, max_memory_mb: float, message: str = None):
        self.current_usage_mb = current_usage_mb
        self.max_memory_mb = max_memory_mb
        
        if message is None:
            message = (f"Memory limit exceeded: {current_usage_mb:.1f}MB used, "
                      f"limit is {max_memory_mb:.1f}MB")
        
        super().__init__(message)


class ProcessingTimeoutException(LargeFileException):
    """Raised when processing takes too long."""
    
    def __init__(self, timeout_seconds: float, message: str = None):
        self.timeout_seconds = timeout_seconds
        
        if message is None:
            message = f"Processing timeout exceeded: {timeout_seconds:.1f} seconds"
        
        super().__init__(message)


class ChunkProcessingException(LargeFileException):
    """Raised when chunk processing fails."""
    
    def __init__(self, chunk_start: int, chunk_end: int, original_error: Exception, message: str = None):
        self.chunk_start = chunk_start
        self.chunk_end = chunk_end
        self.original_error = original_error
        
        if message is None:
            message = (f"Failed to process chunk {chunk_start}-{chunk_end}: "
                      f"{type(original_error).__name__}: {original_error}")
        
        super().__init__(message)


class StreamingWriterException(LargeFileException):
    """Raised when streaming writer encounters an error."""
    pass
