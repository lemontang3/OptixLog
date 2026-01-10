"""Unit tests for OptixLog SDK result types."""

from datetime import datetime
import pytest

from optixlog.result_types import (
    MetricResult,
    MediaResult,
    BatchResult,
    RunInfo,
    ArtifactInfo,
    ProjectInfo,
    ComparisonResult,
)


class TestMetricResult:
    """Test MetricResult dataclass."""

    def test_successful_result(self):
        """Test successful metric result."""
        result = MetricResult(
            step=10,
            metrics={"loss": 0.5, "accuracy": 0.9},
            success=True,
            timestamp=datetime.now()
        )

        assert result.step == 10
        assert result.success is True
        assert result.error is None
        assert bool(result) is True

    def test_failed_result(self):
        """Test failed metric result."""
        result = MetricResult(
            step=10,
            metrics={"loss": float('nan')},
            success=False,
            error="NaN detected"
        )

        assert result.success is False
        assert result.error == "NaN detected"
        assert bool(result) is False

    def test_repr_success(self):
        """Test repr for successful result."""
        result = MetricResult(
            step=10,
            metrics={"loss": 0.5, "accuracy": 0.9},
            success=True
        )

        repr_str = repr(result)
        assert "step=10" in repr_str
        assert "2 values" in repr_str
        assert "✓" in repr_str

    def test_repr_failure(self):
        """Test repr for failed result."""
        result = MetricResult(
            step=10,
            metrics={"loss": 0.5},
            success=False,
            error="Server error"
        )

        repr_str = repr(result)
        assert "step=10" in repr_str
        assert "Server error" in repr_str
        assert "✗" in repr_str


class TestMediaResult:
    """Test MediaResult dataclass."""

    def test_successful_image_upload(self):
        """Test successful image upload result."""
        result = MediaResult(
            key="test_image",
            success=True,
            media_id="media123",
            url="https://optixlog.com/media/media123",
            file_size=12345,
            content_type="image/png"
        )

        assert result.key == "test_image"
        assert result.success is True
        assert result.media_id == "media123"
        assert result.url is not None
        assert bool(result) is True

    def test_failed_upload(self):
        """Test failed upload result."""
        result = MediaResult(
            key="test_image",
            success=False,
            error="File too large"
        )

        assert result.success is False
        assert result.error == "File too large"
        assert bool(result) is False

    def test_repr_success(self):
        """Test repr for successful result."""
        result = MediaResult(
            key="test_image",
            success=True,
            url="https://optixlog.com/media/123",
            file_size=12345
        )

        repr_str = repr(result)
        assert "test_image" in repr_str
        assert "https://optixlog.com/media/123" in repr_str
        assert "12345 bytes" in repr_str

    def test_repr_failure(self):
        """Test repr for failed result."""
        result = MediaResult(
            key="test_image",
            success=False,
            error="Upload failed"
        )

        repr_str = repr(result)
        assert "test_image" in repr_str
        assert "Upload failed" in repr_str


class TestBatchResult:
    """Test BatchResult dataclass."""

    def test_fully_successful_batch(self):
        """Test batch where all operations succeeded."""
        result = BatchResult(
            total=10,
            successful=10,
            failed=0
        )

        assert result.total == 10
        assert result.successful == 10
        assert result.failed == 0
        assert result.success_rate == 100.0
        assert bool(result) is True

    def test_partially_successful_batch(self):
        """Test batch with some failures."""
        result = BatchResult(
            total=10,
            successful=7,
            failed=3,
            errors=["Error 1", "Error 2", "Error 3"]
        )

        assert result.success_rate == 70.0
        assert bool(result) is False  # Not all succeeded
        assert len(result.errors) == 3

    def test_completely_failed_batch(self):
        """Test batch where all operations failed."""
        result = BatchResult(
            total=5,
            successful=0,
            failed=5
        )

        assert result.success_rate == 0.0
        assert bool(result) is False

    def test_empty_batch(self):
        """Test empty batch."""
        result = BatchResult(
            total=0,
            successful=0,
            failed=0
        )

        assert result.success_rate == 0.0

    def test_repr(self):
        """Test repr."""
        result = BatchResult(
            total=10,
            successful=7,
            failed=3
        )

        repr_str = repr(result)
        assert "7/10" in repr_str
        assert "70.0%" in repr_str


class TestRunInfo:
    """Test RunInfo dataclass."""

    def test_run_info(self):
        """Test run info creation."""
        run_info = RunInfo(
            run_id="run123",
            name="Test Run",
            project_id="proj1",
            project_name="Test Project",
            config={"lr": 0.001},
            created_at="2024-01-01T00:00:00Z",
            status="completed"
        )

        assert run_info.run_id == "run123"
        assert run_info.name == "Test Run"
        assert run_info.status == "completed"
        assert run_info.config["lr"] == 0.001

    def test_repr(self):
        """Test repr."""
        run_info = RunInfo(
            run_id="run123",
            name="Test Run",
            project_id="proj1",
            project_name="Test Project",
            config={},
            created_at="2024-01-01T00:00:00Z"
        )

        repr_str = repr(run_info)
        assert "run123" in repr_str
        assert "Test Run" in repr_str
        assert "running" in repr_str  # Default status


class TestArtifactInfo:
    """Test ArtifactInfo dataclass."""

    def test_artifact_info(self):
        """Test artifact info creation."""
        artifact = ArtifactInfo(
            media_id="media123",
            key="results",
            kind="file",
            url="https://optixlog.com/media/media123",
            content_type="application/json",
            file_size=1024,
            meta={"description": "Test results"}
        )

        assert artifact.media_id == "media123"
        assert artifact.key == "results"
        assert artifact.kind == "file"
        assert artifact.meta["description"] == "Test results"

    def test_repr(self):
        """Test repr."""
        artifact = ArtifactInfo(
            media_id="media123",
            key="results",
            kind="file",
            url="https://optixlog.com/media/media123",
            content_type="application/json"
        )

        repr_str = repr(artifact)
        assert "results" in repr_str
        assert "file" in repr_str
        assert "https://optixlog.com/media/media123" in repr_str


class TestProjectInfo:
    """Test ProjectInfo dataclass."""

    def test_project_info(self):
        """Test project info creation."""
        project = ProjectInfo(
            project_id="proj123",
            name="Test Project",
            created_at="2024-01-01T00:00:00Z",
            run_count=42
        )

        assert project.project_id == "proj123"
        assert project.name == "Test Project"
        assert project.run_count == 42

    def test_repr_with_run_count(self):
        """Test repr with run count."""
        project = ProjectInfo(
            project_id="proj123",
            name="Test Project",
            created_at="2024-01-01T00:00:00Z",
            run_count=42
        )

        repr_str = repr(project)
        assert "proj123" in repr_str
        assert "Test Project" in repr_str
        assert "42 runs" in repr_str

    def test_repr_without_run_count(self):
        """Test repr without run count."""
        project = ProjectInfo(
            project_id="proj123",
            name="Test Project",
            created_at="2024-01-01T00:00:00Z"
        )

        repr_str = repr(project)
        assert "proj123" in repr_str
        assert "Test Project" in repr_str
        assert "runs" not in repr_str


class TestComparisonResult:
    """Test ComparisonResult dataclass."""

    def test_comparison_result(self):
        """Test comparison result creation."""
        run1 = RunInfo(
            run_id="run1",
            name="Run 1",
            project_id="proj1",
            project_name="Test",
            config={},
            created_at="2024-01-01T00:00:00Z"
        )

        run2 = RunInfo(
            run_id="run2",
            name="Run 2",
            project_id="proj1",
            project_name="Test",
            config={},
            created_at="2024-01-01T00:00:00Z"
        )

        comparison = ComparisonResult(
            runs=[run1, run2],
            common_metrics=["loss", "accuracy"],
            metrics_data={
                "loss": {
                    "run1": [0.5, 0.4, 0.3],
                    "run2": [0.6, 0.5, 0.4]
                },
                "accuracy": {
                    "run1": [0.8, 0.85, 0.9],
                    "run2": [0.75, 0.8, 0.85]
                }
            }
        )

        assert len(comparison.runs) == 2
        assert len(comparison.common_metrics) == 2
        assert "loss" in comparison.metrics_data
        assert "accuracy" in comparison.metrics_data

    def test_repr(self):
        """Test repr."""
        run1 = RunInfo(
            run_id="run1",
            name="Run 1",
            project_id="proj1",
            project_name="Test",
            config={},
            created_at="2024-01-01T00:00:00Z"
        )

        comparison = ComparisonResult(
            runs=[run1],
            common_metrics=["loss", "accuracy"],
            metrics_data={}
        )

        repr_str = repr(comparison)
        assert "1 runs" in repr_str
        assert "2 common metrics" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
