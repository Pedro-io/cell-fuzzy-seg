"""Logger setup module.

Provides a shared `logger` instance (from Loguru) to be imported across jobs
for consistent logging. Additional sinks or formatting can be configured
centrally here in the future if needed.
"""

from loguru import logger

__all__ = ["logger"]
