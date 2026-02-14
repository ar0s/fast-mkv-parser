"""fast-mkv-parser: Sparse MKV track extraction library.

Reads only block headers and desired track payloads from MKV files,
skipping video data via lseek for ~10-15x I/O reduction on large files.
"""

from .extractor import MkvParser
from .matroska import TrackInfo

__all__ = ["MkvParser", "TrackInfo"]
__version__ = "0.1.0"
