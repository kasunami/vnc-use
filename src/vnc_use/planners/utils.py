"""Shared utilities for VNC planners."""

import io
import logging

from PIL import Image

logger = logging.getLogger(__name__)


def compress_screenshot(png_bytes: bytes, max_width: int = 512) -> bytes:
    """Compress screenshot to reduce token count.

    Args:
        png_bytes: Original PNG bytes
        max_width: Maximum width in pixels

    Returns:
        Compressed PNG bytes
    """
    img: Image.Image = Image.open(io.BytesIO(png_bytes))

    # Resize if too large
    if img.width > max_width:
        ratio = max_width / img.width
        new_size = (max_width, int(img.height * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        logger.debug(f"Resized screenshot to {new_size}")

    # Compress
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True, compress_level=9)
    compressed = buf.getvalue()

    logger.debug(f"Compressed screenshot: {len(png_bytes)} -> {len(compressed)} bytes")
    return compressed
