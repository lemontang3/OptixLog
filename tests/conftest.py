"""Pytest configuration and fixtures for OptixLog SDK tests."""

import os
import sys
from pathlib import Path
import pytest

# Add SDK to Python path
sdk_path = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(sdk_path))


@pytest.fixture
def mock_api_key():
    """Provide a mock API key for testing."""
    return "test_api_key_12345678"


@pytest.fixture
def mock_api_url():
    """Provide a mock API URL for testing."""
    return "https://test.optixlog.com"


@pytest.fixture
def sample_metrics():
    """Provide sample metrics for testing."""
    return {
        "loss": 0.5,
        "accuracy": 0.9,
        "step": 100,
        "learning_rate": 0.001
    }


@pytest.fixture
def sample_config():
    """Provide sample config for testing."""
    return {
        "model": "resnet50",
        "optimizer": "adam",
        "lr": 0.001,
        "epochs": 100
    }


@pytest.fixture(autouse=True)
def clean_env():
    """Clean environment variables before each test."""
    # Store original env
    original_env = os.environ.copy()

    # Remove OptixLog env vars
    for key in list(os.environ.keys()):
        if key.startswith("OPTIX") or key.startswith("OPTIXLOG"):
            del os.environ[key]

    yield

    # Restore original env
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def temp_image():
    """Create a temporary PIL image for testing."""
    try:
        from PIL import Image
        img = Image.new('RGB', (100, 100), color='red')
        return img
    except ImportError:
        pytest.skip("PIL not available")


@pytest.fixture
def temp_matplotlib_figure():
    """Create a temporary matplotlib figure for testing."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        ax.set_title("Test Plot")

        yield fig

        plt.close(fig)
    except ImportError:
        pytest.skip("Matplotlib not available")


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "tidy3d: mark test as tidy3d integration test"
    )
