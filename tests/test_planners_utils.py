"""Tests for planners/utils.py module."""

import io

from PIL import Image

from vnc_use.planners.utils import compress_screenshot


def create_test_png(width: int = 100, height: int = 100, color: str = "red") -> bytes:
    """Create a minimal valid PNG image for testing."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestCompressScreenshot:
    """Tests for compress_screenshot function in utils.py."""

    def test_compress_small_image_unchanged_dimensions(self):
        """Should not resize image smaller than max_width."""
        png_bytes = create_test_png(100, 75)
        result = compress_screenshot(png_bytes, max_width=512)

        assert isinstance(result, bytes)
        img = Image.open(io.BytesIO(result))
        assert img.width == 100
        assert img.height == 75

    def test_compress_large_image_resizes(self):
        """Should resize image larger than max_width."""
        png_bytes = create_test_png(1024, 768)
        result = compress_screenshot(png_bytes, max_width=512)

        img = Image.open(io.BytesIO(result))
        assert img.width == 512
        # Height should maintain aspect ratio
        expected_height = int(768 * (512 / 1024))
        assert img.height == expected_height

    def test_compress_exact_max_width_unchanged(self):
        """Should not resize image exactly at max_width."""
        png_bytes = create_test_png(512, 384)
        result = compress_screenshot(png_bytes, max_width=512)

        img = Image.open(io.BytesIO(result))
        assert img.width == 512

    def test_compress_custom_max_width(self):
        """Should use custom max_width parameter."""
        png_bytes = create_test_png(1000, 500)
        result = compress_screenshot(png_bytes, max_width=256)

        img = Image.open(io.BytesIO(result))
        assert img.width == 256
        assert img.height == 128  # Maintains aspect ratio

    def test_compress_default_max_width(self):
        """Should use default max_width=512 when not specified."""
        png_bytes = create_test_png(1024, 512)
        result = compress_screenshot(png_bytes)  # No max_width specified

        img = Image.open(io.BytesIO(result))
        assert img.width == 512

    def test_compress_maintains_aspect_ratio(self):
        """Should maintain aspect ratio when resizing."""
        # Wide image
        png_bytes = create_test_png(2000, 500)
        result = compress_screenshot(png_bytes, max_width=500)

        img = Image.open(io.BytesIO(result))
        assert img.width == 500
        assert img.height == 125  # 500 * (500/2000) = 125

    def test_compress_returns_valid_png(self):
        """Should return valid PNG bytes."""
        png_bytes = create_test_png(800, 600)
        result = compress_screenshot(png_bytes, max_width=400)

        # Should be able to open as PNG
        img = Image.open(io.BytesIO(result))
        assert img.format == "PNG"

    def test_compress_optimizes_output(self):
        """Should produce optimized PNG output."""
        # Create a large image that can be compressed
        png_bytes = create_test_png(800, 600)
        result = compress_screenshot(png_bytes, max_width=400)

        # Result should be different from input (resized)
        assert result != png_bytes

    def test_compress_logs_resize_message(self, caplog):
        """Should log debug message when resizing."""
        png_bytes = create_test_png(1024, 768)

        import logging

        caplog.set_level(logging.DEBUG)

        compress_screenshot(png_bytes, max_width=512)

        assert "Resized screenshot to" in caplog.text

    def test_compress_logs_compression_message(self, caplog):
        """Should log debug message with compression details."""
        png_bytes = create_test_png(500, 375)

        import logging

        caplog.set_level(logging.DEBUG)

        compress_screenshot(png_bytes, max_width=512)

        assert "Compressed screenshot:" in caplog.text

    def test_compress_tall_image(self):
        """Should handle tall images (height > width) correctly."""
        # Tall image that needs resizing
        png_bytes = create_test_png(1024, 2048)
        result = compress_screenshot(png_bytes, max_width=512)

        img = Image.open(io.BytesIO(result))
        assert img.width == 512
        assert img.height == 1024  # Maintains 1:2 ratio

    def test_compress_square_image(self):
        """Should handle square images correctly."""
        png_bytes = create_test_png(1024, 1024)
        result = compress_screenshot(png_bytes, max_width=512)

        img = Image.open(io.BytesIO(result))
        assert img.width == 512
        assert img.height == 512  # Still square

    def test_compress_various_sizes(self):
        """Should handle various image sizes."""
        test_cases = [
            (100, 100),  # Small
            (512, 512),  # Exactly max
            (513, 513),  # Just over max
            (1920, 1080),  # HD
            (3840, 2160),  # 4K
        ]

        for width, height in test_cases:
            png_bytes = create_test_png(width, height)
            result = compress_screenshot(png_bytes, max_width=512)

            img = Image.open(io.BytesIO(result))
            assert img.width <= 512, f"Failed for {width}x{height}"
