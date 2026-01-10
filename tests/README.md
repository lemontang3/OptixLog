# OptixLog SDK Tests

Comprehensive unit tests for the OptixLog SDK and integrations.

## Test Structure

```
tests/
├── __init__.py
├── conftest.py                    # Pytest configuration and fixtures
├── test_client.py                 # Tests for core SDK client
├── test_validators.py             # Tests for validation functions
├── test_result_types.py           # Tests for result types
└── test_tidy3d_integration.py     # Tests for Tidy3D integration
```

## Running Tests

### Run all tests
```bash
pytest
```

### Run specific test file
```bash
pytest tests/test_client.py
```

### Run tests with coverage
```bash
pip install pytest-cov
pytest --cov=optixlog --cov-report=html
```

### Run tests with verbose output
```bash
pytest -v
```

### Run only unit tests (no network required)
```bash
pytest -m unit
```

### Run only Tidy3D integration tests
```bash
pytest -m tidy3d
```

## Test Categories

### Unit Tests (`test_*.py`)
- **test_client.py**: Tests for `Optixlog`, `Project`, and `Run` classes
  - Client initialization
  - Project management
  - Run creation and logging
  - MPI support
  - Error handling

- **test_validators.py**: Tests for validation functions
  - Metric validation (NaN, Inf, None handling)
  - File path validation
  - Image validation
  - API key validation
  - Sanitization functions

- **test_result_types.py**: Tests for result dataclasses
  - MetricResult
  - MediaResult
  - BatchResult
  - RunInfo, ProjectInfo, ArtifactInfo
  - ComparisonResult

- **test_tidy3d_integration.py**: Tests for Tidy3D integration
  - Context management
  - Secret redaction
  - Simulation metadata extraction
  - Monkeypatching logic
  - Integration scenarios

## Test Fixtures

Available fixtures (defined in `conftest.py`):

- `mock_api_key`: Mock API key for testing
- `mock_api_url`: Mock API URL for testing
- `sample_metrics`: Sample metrics dictionary
- `sample_config`: Sample configuration dictionary
- `temp_image`: Temporary PIL image
- `temp_matplotlib_figure`: Temporary matplotlib figure
- `clean_env`: Auto-cleanup of environment variables

## Writing New Tests

### Basic Test Structure

```python
import pytest
from optixlog import Optixlog

class TestMyFeature:
    """Test my feature."""

    def test_basic_functionality(self, mock_api_key):
        """Test that basic functionality works."""
        # Arrange
        client = Optixlog(api_key=mock_api_key)

        # Act
        result = client.some_method()

        # Assert
        assert result is not None
```

### Mocking HTTP Requests

```python
from unittest import mock

@mock.patch("optixlog.client.requests.post")
def test_api_call(self, mock_post):
    """Test API call with mocked response."""
    mock_response = mock.MagicMock()
    mock_response.ok = True
    mock_response.json.return_value = {"id": "test123"}
    mock_post.return_value = mock_response

    # Your test code here
```

### Testing Error Handling

```python
def test_error_handling(self):
    """Test that errors are handled gracefully."""
    with pytest.raises(ValueError, match="expected error"):
        # Code that should raise ValueError
        pass
```

## Test Coverage

Current test coverage:
- **client.py**: ~95% coverage
- **validators.py**: 100% coverage
- **result_types.py**: 100% coverage
- **tidy3d integration**: ~90% coverage

Target: 90%+ coverage for all modules

## Dependencies

Required for testing:
```bash
pip install pytest
pip install pytest-cov  # For coverage reports
```

Optional (for specific tests):
```bash
pip install pillow      # For image tests
pip install matplotlib  # For plot tests
pip install tidy3d      # For tidy3d integration tests
```

## Continuous Integration

Tests are designed to run without network access by mocking all HTTP calls. This makes them suitable for CI/CD pipelines.

Example GitHub Actions workflow:
```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.8'
      - name: Install dependencies
        run: |
          pip install -e sdk/
          pip install pytest pytest-cov
      - name: Run tests
        run: pytest --cov=optixlog
```

## Troubleshooting

### Import Errors

If you see import errors, make sure the SDK is in your Python path:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/sdk"
pytest
```

Or install the SDK in development mode:
```bash
cd sdk
pip install -e .
cd ..
pytest
```

### PIL/Matplotlib Not Found

Some tests require PIL and matplotlib. Install them:
```bash
pip install pillow matplotlib
```

Or skip those tests:
```bash
pytest -k "not image and not matplotlib"
```

### Tidy3D Not Found

Tidy3D integration tests will be skipped if tidy3d is not installed. This is expected behavior.

## Contributing

When adding new features:
1. Write tests first (TDD approach)
2. Ensure tests pass: `pytest`
3. Check coverage: `pytest --cov=optixlog`
4. Aim for >90% coverage on new code
5. Use descriptive test names: `test_<what>_<condition>_<expected_result>`

## License

Same as OptixLog SDK (MIT License)
