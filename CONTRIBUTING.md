# Contributing to OptixLog

Thank you for your interest in contributing to OptixLog! We welcome contributions from the community and are excited to work with you.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Community](#community)

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic understanding of photonic simulations (helpful but not required)
- Familiarity with pytest for testing

### Finding Issues to Work On

- Check our [GitHub Issues](https://github.com/fluxboard/Optixlog/issues) page
- Look for issues labeled `good first issue` for beginner-friendly tasks
- Issues labeled `help wanted` are actively seeking contributors
- Feel free to propose new features by opening an issue first

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/Optixlog.git
cd Optixlog
```

### 2. Create a Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install the SDK in editable mode with development dependencies
cd sdk
pip install -e .

# Install test dependencies
pip install pytest pytest-cov

# Install optional dependencies for full functionality
pip install mpi4py meep tidy3d rich matplotlib pillow numpy
```

### 4. Verify Installation

```bash
# Run tests to ensure everything is set up correctly
pytest
```

## How to Contribute

### Types of Contributions

We welcome many types of contributions:

- **Bug fixes** - Fix issues or incorrect behavior
- **New features** - Add new logging capabilities, integrations, or utilities
- **Documentation** - Improve README, docstrings, or add examples
- **Tests** - Add test coverage or improve existing tests
- **Performance** - Optimize slow operations
- **Integrations** - Add support for new simulation frameworks
- **Examples** - Add example scripts demonstrating usage

### Before You Start

1. **Check existing issues and PRs** - Someone might already be working on it
2. **Open an issue first** for significant changes to discuss the approach
3. **Keep changes focused** - One PR should address one issue or feature
4. **Write tests** - All new code should include tests

## Pull Request Process

### 1. Create a Branch

```bash
# Create a descriptive branch name
git checkout -b feature/add-lumerical-support
# or
git checkout -b fix/nan-validation-edge-case
```

### 2. Make Your Changes

- Write clean, readable code following our [coding standards](#coding-standards)
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass

### 3. Commit Your Changes

We follow conventional commit messages:

```bash
# Format: <type>(<scope>): <description>

git commit -m "feat(client): add support for Lumerical simulations"
git commit -m "fix(validators): handle edge case in NaN detection"
git commit -m "docs(readme): add example for batch logging"
git commit -m "test(query): add tests for compare_runs function"
```

**Types:**
- `feat` - New feature
- `fix` - Bug fix
- `docs` - Documentation changes
- `test` - Adding or updating tests
- `refactor` - Code refactoring
- `perf` - Performance improvements
- `chore` - Maintenance tasks

### 4. Push and Create Pull Request

```bash
git push origin your-branch-name
```

Then create a pull request on GitHub with:

- **Clear title** - Descriptive and concise
- **Description** - What changes were made and why
- **Linked issues** - Reference related issues (e.g., "Fixes #123")
- **Testing** - Describe how you tested the changes
- **Screenshots** - If applicable (UI changes, visualizations)

### 5. Code Review

- Respond to feedback promptly
- Make requested changes in new commits (don't force push during review)
- Once approved, a maintainer will merge your PR

## Coding Standards

### Python Style

- Follow [PEP 8](https://pep8.org/) style guide
- Use meaningful variable and function names
- Maximum line length: 100 characters
- Use type hints where appropriate

```python
# Good
def log_metric(self, step: int, metric_name: str, value: float) -> MetricResult:
    """Log a single metric value at a given step."""
    if math.isnan(value):
        raise ValidationError(f"Metric '{metric_name}' contains NaN value")
    return self._send_metric(step, metric_name, value)

# Avoid
def lm(s, mn, v):
    if math.isnan(v):
        raise ValidationError("nan")
    return self._sm(s, mn, v)
```

### Code Organization

- **Keep functions focused** - One function, one purpose
- **Avoid deep nesting** - Extract nested logic into helper functions
- **Use early returns** - Reduce nesting with guard clauses
- **DRY principle** - Don't repeat yourself

### Documentation

- Add docstrings to all public functions and classes
- Use Google-style docstrings

```python
def compare_runs(run_ids: List[str], api_key: str) -> ComparisonResult:
    """Compare metrics across multiple runs.

    Args:
        run_ids: List of run IDs to compare
        api_key: API key for authentication

    Returns:
        ComparisonResult containing aggregated metrics and statistics

    Raises:
        ValidationError: If run_ids is empty or contains invalid IDs
        APIError: If the API request fails

    Example:
        >>> comparison = compare_runs(['run_1', 'run_2'], api_key='key')
        >>> print(comparison.best_run_id)
    """
    # Implementation...
```

## Testing Guidelines

### Writing Tests

- Place tests in the `tests/` directory
- Test files should mirror the structure of the SDK
- Name test files `test_*.py`
- Name test functions `test_*`

### Test Categories

Use pytest markers to categorize tests:

```python
import pytest

@pytest.mark.unit
def test_validation_rejects_nan():
    """Unit test for NaN validation."""
    with pytest.raises(ValidationError):
        validate_metric(float('nan'))

@pytest.mark.integration
def test_end_to_end_logging():
    """Integration test for complete logging workflow."""
    client = Optixlog(api_key="test_key")
    # ...

@pytest.mark.slow
def test_large_batch_upload():
    """Test uploading 10,000 metrics."""
    # ...
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific category
pytest -m unit
pytest -m integration

# Run with coverage
pytest --cov=optixlog --cov-report=html

# Run specific test file
pytest tests/test_validators.py

# Run specific test
pytest tests/test_validators.py::test_validation_rejects_nan
```

### Test Coverage

- Aim for >80% code coverage
- All new features must include tests
- Bug fixes should include regression tests

## Documentation

### Code Documentation

- Add docstrings to all public APIs
- Include examples in docstrings where helpful
- Keep docstrings up to date with code changes

### README and Guides

- Update README.md if you add new features
- Add examples to demonstrate new functionality
- Update API reference if signatures change

### Inline Comments

Use comments to explain **why**, not **what**:

```python
# Good
# Skip logging on non-master ranks to avoid duplicate API calls
if self.rank != 0:
    return None

# Avoid (comment explains obvious code)
# Check if rank is not 0
if self.rank != 0:
    return None
```

## Project-Specific Guidelines

### MPI Compatibility

When adding features, ensure they work correctly in MPI environments:

```python
# Always check rank before API calls
if self.rank == 0:
    self._make_api_call()

# Non-master ranks should return early or return None
if self.rank != 0:
    return None
```

### Validation

Add validation for user inputs:

```python
from optixlog.validators import validate_metric, ValidationError

def log(self, step: int, **metrics):
    for key, value in metrics.items():
        validate_metric(value, metric_name=key)  # Raises ValidationError if invalid
```

### Error Handling

- Use custom exceptions from `validators.py` (e.g., `ValidationError`)
- Provide helpful error messages with context
- Don't silence exceptions without good reason

### Backward Compatibility

- Maintain backward compatibility when possible
- Deprecate features gracefully before removing
- Document breaking changes clearly in PR description

## Release Process

(For maintainers)

1. Update version in `pyproject.toml` and `setup.py`
2. Update CHANGELOG.md with release notes
3. Create a git tag: `git tag -a v0.3.0 -m "Release v0.3.0"`
4. Push tag: `git push origin v0.3.0`
5. Build and publish to PyPI: `python -m build && twine upload dist/*`

## Community

### Communication Channels

- **GitHub Issues** - Bug reports, feature requests
- **GitHub Discussions** - Questions, ideas, general discussion
- **Pull Requests** - Code review and collaboration

### Getting Help

If you need help:

1. Check the [README](README.md) and documentation
2. Search existing issues and discussions
3. Ask in [GitHub Discussions](https://github.com/fluxboard/Optixlog/discussions)
4. Open a new issue if you've found a bug

### Recognition

Contributors will be:
- Listed in release notes
- Credited in relevant documentation
- Added to CONTRIBUTORS.md (if we create one)

## License

By contributing to OptixLog, you agree that your contributions will be licensed under the GNU Lesser General Public License v2.1. See [LICENSE](LICENSE) for details.

This means:
- Your code will be open source under LGPL v2.1
- Others can use your contributions in both open and proprietary projects
- Modifications to your code must remain open source

## Questions?

Don't hesitate to ask questions! We're here to help:

- Open an issue labeled `question`
- Start a discussion in GitHub Discussions
- Comment on relevant issues or PRs

Thank you for contributing to OptixLog!

---

**Happy Coding!** We appreciate your time and effort in making OptixLog better for the photonics simulation community.
