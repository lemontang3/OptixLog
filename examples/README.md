# OptixLog Examples

This directory contains example scripts demonstrating various features of OptixLog.

## Prerequisites

```bash
# Install OptixLog with optional dependencies
pip install optixlog[all]

# Or install specific dependencies
pip install optixlog[mpi]  # For MPI examples
pip install optixlog[meep]  # For MEEP simulations
```

## Examples

### 1. Basic Usage (`basic_usage.py`)

Demonstrates fundamental OptixLog features:
- Client initialization
- Creating projects and runs
- Logging scalar metrics
- Logging matplotlib figures
- Logging numpy arrays as images

```bash
python basic_usage.py
```

### 2. MPI-Aware Logging (`mpi_example.py`)

Shows how OptixLog automatically handles MPI environments:
- Only master rank (rank 0) performs logging
- Safe to call logging methods from all ranks
- Getting MPI information

```bash
# Run with MPI
mpirun -n 4 python mpi_example.py

# Or run without MPI (works normally)
python mpi_example.py
```

### 3. Photonic Simulation (`photonic_simulation.py`)

Example of parameter sweep for waveguide optimization:
- Running multiple simulation configurations
- Logging simulation results
- Creating summary visualizations
- Tracking best results

```bash
python photonic_simulation.py
```

## Configuration

All examples require an OptixLog API key. You can provide it in two ways:

### Option 1: Environment Variable (Recommended)

```bash
export OPTIXLOG_API_KEY="your_api_key_here"
python basic_usage.py
```

### Option 2: Direct Parameter

Edit the example script and replace `"your_api_key_here"` with your actual API key:

```python
client = Optixlog(api_key="your_actual_key")
```

## Getting an API Key

1. Sign up at [optixlog.com](https://optixlog.com)
2. Navigate to Settings → API Keys
3. Generate a new API key
4. Copy and use it in your scripts

## Next Steps

After running these examples, check out:

- [Full Documentation](https://optixlog.com/docs)
- [API Reference](https://github.com/fluxboard/Optixlog#api-reference)
- [CONTRIBUTING.md](../CONTRIBUTING.md) to contribute your own examples

## Support

If you encounter any issues:

- Check the [GitHub Issues](https://github.com/fluxboard/Optixlog/issues)
- Join [GitHub Discussions](https://github.com/fluxboard/Optixlog/discussions)
- Read the [README](../README.md)
