"""Unit tests for OptixLog SDK validators."""

import math
import tempfile
from pathlib import Path
import pytest

from optixlog.validators import (
    ValidationError,
    validate_metrics,
    validate_file_path,
    validate_image,
    validate_step,
    validate_key,
    sanitize_metrics,
    guess_content_type,
    validate_api_key,
    validate_batch_size,
)


class TestValidateMetrics:
    """Test metric validation."""

    def test_valid_metrics(self):
        """Test that valid metrics pass."""
        metrics = {"loss": 0.5, "accuracy": 0.9, "step": 100}
        valid, error = validate_metrics(metrics)
        assert valid is True
        assert error is None

    def test_empty_metrics(self):
        """Test that empty metrics fail."""
        valid, error = validate_metrics({})
        assert valid is False
        assert "empty" in error.lower()

    def test_non_dict_metrics(self):
        """Test that non-dict fails."""
        valid, error = validate_metrics([1, 2, 3])
        assert valid is False
        assert "dictionary" in error.lower()

    def test_nan_value(self):
        """Test that NaN values fail."""
        metrics = {"loss": float('nan')}
        valid, error = validate_metrics(metrics)
        assert valid is False
        assert "nan" in error.lower()

    def test_inf_value(self):
        """Test that Inf values fail."""
        metrics = {"loss": float('inf')}
        valid, error = validate_metrics(metrics)
        assert valid is False
        assert "inf" in error.lower()

    def test_none_value(self):
        """Test that None values fail."""
        metrics = {"loss": None}
        valid, error = validate_metrics(metrics)
        assert valid is False
        assert "none" in error.lower()

    def test_non_string_key(self):
        """Test that non-string keys fail."""
        metrics = {123: 0.5}
        valid, error = validate_metrics(metrics)
        assert valid is False
        assert "string" in error.lower()

    def test_mixed_valid_types(self):
        """Test that various valid types pass."""
        metrics = {
            "loss": 0.5,
            "accuracy": 0.9,
            "step": 100,
            "name": "test",
            "success": True,
            "tags": ["a", "b"],
            "config": {"lr": 0.001}
        }
        valid, error = validate_metrics(metrics)
        assert valid is True
        assert error is None


class TestValidateFilePath:
    """Test file path validation."""

    def test_valid_file(self):
        """Test that valid file passes."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"test content")
            tmp.flush()

            valid, error = validate_file_path(tmp.name)
            assert valid is True
            assert error is None

            # Cleanup
            Path(tmp.name).unlink()

    def test_nonexistent_file(self):
        """Test that nonexistent file fails."""
        valid, error = validate_file_path("/nonexistent/file.txt")
        assert valid is False
        assert "not found" in error.lower()

    def test_directory_path(self):
        """Test that directory fails."""
        valid, error = validate_file_path("/tmp")
        assert valid is False
        assert "not a file" in error.lower()

    def test_non_string_path(self):
        """Test that non-string fails."""
        valid, error = validate_file_path(123)
        assert valid is False
        assert "string" in error.lower()

    def test_large_file(self):
        """Test that very large file fails."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            # Write > 100MB of data
            chunk = b"x" * (10 * 1024 * 1024)  # 10MB chunks
            for _ in range(11):  # Write 110MB
                tmp.write(chunk)
            tmp.flush()

            valid, error = validate_file_path(tmp.name)
            assert valid is False
            assert "large" in error.lower()

            # Cleanup
            Path(tmp.name).unlink()


class TestValidateImage:
    """Test image validation."""

    def test_valid_image(self):
        """Test that valid PIL image passes."""
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("PIL not available")

        img = Image.new('RGB', (100, 100))
        valid, error = validate_image(img)
        assert valid is True
        assert error is None

    def test_non_image_object(self):
        """Test that non-image fails."""
        valid, error = validate_image("not an image")
        assert valid is False

    def test_zero_dimension_image(self):
        """Test that zero-dimension image fails."""
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("PIL not available")

        img = Image.new('RGB', (0, 100))
        valid, error = validate_image(img)
        assert valid is False
        assert "zero" in error.lower()

    def test_too_large_image(self):
        """Test that too large image fails."""
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("PIL not available")

        img = Image.new('RGB', (15000, 15000))
        valid, error = validate_image(img)
        assert valid is False
        assert "large" in error.lower()


class TestValidateStep:
    """Test step validation."""

    def test_valid_step(self):
        """Test that valid step passes."""
        valid, error = validate_step(100)
        assert valid is True
        assert error is None

    def test_zero_step(self):
        """Test that zero step passes."""
        valid, error = validate_step(0)
        assert valid is True
        assert error is None

    def test_negative_step(self):
        """Test that negative step fails."""
        valid, error = validate_step(-1)
        assert valid is False
        assert "negative" in error.lower()

    def test_non_integer_step(self):
        """Test that non-integer fails."""
        valid, error = validate_step(1.5)
        assert valid is False
        assert "integer" in error.lower()


class TestValidateKey:
    """Test key validation."""

    def test_valid_key(self):
        """Test that valid key passes."""
        valid, error = validate_key("my_metric")
        assert valid is True
        assert error is None

    def test_empty_key(self):
        """Test that empty key fails."""
        valid, error = validate_key("")
        assert valid is False
        assert "empty" in error.lower()

    def test_whitespace_key(self):
        """Test that whitespace-only key fails."""
        valid, error = validate_key("   ")
        assert valid is False
        assert "empty" in error.lower()

    def test_too_long_key(self):
        """Test that too long key fails."""
        long_key = "x" * 201
        valid, error = validate_key(long_key)
        assert valid is False
        assert "long" in error.lower()

    def test_non_string_key(self):
        """Test that non-string fails."""
        valid, error = validate_key(123)
        assert valid is False
        assert "string" in error.lower()


class TestSanitizeMetrics:
    """Test metric sanitization."""

    def test_replace_nan(self):
        """Test that NaN is replaced."""
        metrics = {"loss": float('nan'), "accuracy": 0.9}
        sanitized = sanitize_metrics(metrics, nan_replacement=0.0)

        assert sanitized["loss"] == 0.0
        assert sanitized["accuracy"] == 0.9

    def test_replace_inf(self):
        """Test that Inf is replaced."""
        metrics = {"loss": float('inf'), "accuracy": 0.9}
        sanitized = sanitize_metrics(metrics, inf_replacement=1e6)

        assert sanitized["loss"] == 1e6
        assert sanitized["accuracy"] == 0.9

    def test_remove_invalid(self):
        """Test that invalid values are removed."""
        metrics = {"loss": float('nan'), "accuracy": 0.9, "error": float('inf')}
        sanitized = sanitize_metrics(metrics, remove_invalid=True)

        assert "loss" not in sanitized
        assert "error" not in sanitized
        assert sanitized["accuracy"] == 0.9

    def test_preserve_valid(self):
        """Test that valid values are preserved."""
        metrics = {"loss": 0.5, "accuracy": 0.9, "step": 100}
        sanitized = sanitize_metrics(metrics)

        assert sanitized == metrics


class TestGuessContentType:
    """Test content type guessing."""

    def test_image_types(self):
        """Test image content types."""
        assert guess_content_type("image.png") == "image/png"
        assert guess_content_type("photo.jpg") == "image/jpeg"
        assert guess_content_type("photo.jpeg") == "image/jpeg"

    def test_text_types(self):
        """Test text content types."""
        assert guess_content_type("data.csv") == "text/csv"
        assert guess_content_type("readme.txt") == "text/plain"
        assert guess_content_type("config.json") == "application/json"

    def test_archive_types(self):
        """Test archive content types."""
        assert guess_content_type("data.zip") == "application/zip"
        assert guess_content_type("backup.tar") == "application/x-tar"
        assert guess_content_type("compressed.gz") == "application/gzip"

    def test_hdf5_types(self):
        """Test HDF5 content types."""
        assert guess_content_type("data.h5") == "application/x-hdf5"
        assert guess_content_type("results.hdf5") == "application/x-hdf5"

    def test_unknown_type(self):
        """Test unknown content type."""
        assert guess_content_type("file.unknown") == "application/octet-stream"

    def test_case_insensitive(self):
        """Test that extension matching is case-insensitive."""
        assert guess_content_type("IMAGE.PNG") == "image/png"
        assert guess_content_type("Data.CSV") == "text/csv"


class TestValidateApiKey:
    """Test API key validation."""

    def test_valid_api_key(self):
        """Test that valid API key passes."""
        valid, error = validate_api_key("sk_test_1234567890")
        assert valid is True
        assert error is None

    def test_empty_api_key(self):
        """Test that empty API key fails."""
        valid, error = validate_api_key("")
        assert valid is False
        assert "required" in error.lower()

    def test_none_api_key(self):
        """Test that None API key fails."""
        valid, error = validate_api_key(None)
        assert valid is False
        assert "required" in error.lower()

    def test_short_api_key(self):
        """Test that short API key fails."""
        valid, error = validate_api_key("short")
        assert valid is False
        assert "short" in error.lower()

    def test_non_string_api_key(self):
        """Test that non-string API key fails."""
        valid, error = validate_api_key(12345)
        assert valid is False
        assert "string" in error.lower()


class TestValidateBatchSize:
    """Test batch size validation."""

    def test_valid_batch_size(self):
        """Test that valid batch size passes."""
        valid, error = validate_batch_size(100)
        assert valid is True
        assert error is None

    def test_zero_batch_size(self):
        """Test that zero batch size fails."""
        valid, error = validate_batch_size(0)
        assert valid is False
        assert "positive" in error.lower()

    def test_negative_batch_size(self):
        """Test that negative batch size fails."""
        valid, error = validate_batch_size(-10)
        assert valid is False
        assert "positive" in error.lower()

    def test_too_large_batch_size(self):
        """Test that too large batch size fails."""
        valid, error = validate_batch_size(2000, max_size=1000)
        assert valid is False
        assert "large" in error.lower()

    def test_non_integer_batch_size(self):
        """Test that non-integer fails."""
        valid, error = validate_batch_size(10.5)
        assert valid is False
        assert "integer" in error.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
