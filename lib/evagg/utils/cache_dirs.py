"""Cache directory management utilities."""
import os
from typing import Optional


class CacheDirectoryManager:
    """Manages cache directories for various components."""
    
    _DEFAULT_CACHE_DIR = ".cache"
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize cache directory manager.
        
        Args:
            cache_dir: Root cache directory. Defaults to .cache
        """
        self._cache_dir = cache_dir or self._DEFAULT_CACHE_DIR
    
    def get_cache_dir(self, subdirectory: str) -> str:
        """Get or create a cache subdirectory.
        
        Args:
            subdirectory: Name of the subdirectory within cache
            
        Returns:
            Full path to the cache subdirectory
        """
        cache_path = os.path.join(self._cache_dir, subdirectory)
        if not os.path.exists(cache_path):
            os.makedirs(cache_path, exist_ok=True)
        return cache_path