# OptixLog SDK

[![License: LGPL v2.1](https://img.shields.io/badge/License-LGPL%20v2.1-blue.svg)](https://www.gnu.org/licenses/lgpl-2.1)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-0.2.0-green.svg)](https://github.com/fluxboard/Optixlog)

A Python SDK for logging and tracking photonic simulation experiments. OptixLog provides a fluent, intuitive API for managing experiment runs, logging metrics, uploading visualizations, and tracking simulation parameters with automatic MPI support for distributed computing environments.

## Features

- **Fluent API** - Chainable, intuitive interface for experiment tracking
- **Rich Logging** - Log metrics, images, plots, histograms, arrays, and arbitrary files
- **MPI Support** - Automatic detection and handling of distributed computing environments (OpenMPI, Intel MPI, MPICH, mpi4py)
- **Photonic Simulation Integration** - Built-in support for MEEP, Tidy3D, and other FDTD frameworks
- **Batch Operations** - Efficient bulk logging with automatic retry logic
- **Type-Safe** - Comprehensive input validation with helpful error messages
- **Zero Configuration** - Works out of the box with sensible defaults

## Installation

### Basic Installation

```bash
pip install optixlog
```

### Development Installation

```bash
git clone https://github.com/fluxboard/Optixlog.git
cd Optixlog/sdk
pip install -e .
```

### Optional Dependencies

For full functionality, install optional dependencies:

```bash
# For MPI support
pip install mpi4py

# For photonic simulation frameworks
pip install meep
pip install tidy3d

# For enhanced console output
pip install rich
```

## Quick Start

### 1. Initialize the Client

```python
from optixlog import Optixlog

# Initialize with API key
client = Optixlog(api_key="your_api_key_here")

# Or use environment variable OPTIXLOG_API_KEY
client = Optixlog()
```

### 2. Create a Project and Run

```python
# Get or create a project
project = client.project(name="photonic_waveguide_optimization")

# Start a new run
run = project.run(
    name="sweep_width_500nm",
    config={
        "waveguide_width": 500,
        "wavelength": 1550,
        "simulation_time": 1000,
        "resolution": 20
    }
)
```

### 3. Log Metrics

```python
# Log scalar metrics
run.log(step=0, transmission=0.85, reflection=0.12, loss=0.03)
run.log(step=1, transmission=0.88, reflection=0.10, loss=0.02)

# Log multiple metrics at once
run.log(
    step=100,
    transmission=0.92,
    reflection=0.06,
    loss=0.02,
    convergence=0.001
)
```

### 4. Log Visualizations

```python
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Log matplotlib figures
fig, ax = plt.subplots()
ax.plot(wavelengths, transmission_spectrum)
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Transmission")
run.log_matplotlib("transmission_spectrum", fig)

# Log images
image = Image.open("simulation_field.png")
run.log_image("electric_field", image)

# Log numpy arrays as heatmaps
field_data = np.random.rand(100, 100)
run.log_array_as_image("field_distribution", field_data)

# Create plots directly
run.log_plot(
    "optimization_curve",
    x=[0, 1, 2, 3, 4],
    y=[0.5, 0.7, 0.85, 0.90, 0.92],
    xlabel="Iteration",
    ylabel="Figure of Merit",
    title="Optimization Progress"
)
```

### 5. Log Files and Artifacts

```python
# Log arbitrary files
run.log_file("simulation_script", "simulation.py")
run.log_file("results_csv", "results.csv")

# Log configuration updates
run.add_config({"optimizer": "adam", "learning_rate": 0.001})
```

## MPI-Aware Logging

OptixLog automatically detects MPI environments and ensures only the master rank (rank 0) performs logging operations:

```python
from mpi4py import MPI
from optixlog import Optixlog

# This works seamlessly in MPI environments
client = Optixlog()
project = client.project("distributed_simulation")
run = project.run("mpi_run")

# Only rank 0 will actually log (other ranks return None)
for step in range(1000):
    # Your simulation code here
    transmission = simulate_step(step)

    # Safe to call from all ranks
    run.log(step=step, transmission=transmission)
```

Supported MPI implementations:
- OpenMPI
- Intel MPI
- MPICH
- mpi4py

## Advanced Usage

### Context Manager

```python
with project.run("experiment_1", config={"param": 1.0}) as run:
    for step in range(100):
        metric = compute_metric(step)
        run.log(step=step, metric=metric)
    # Run automatically finalized
```

### Batch Logging

```python
# Efficient bulk operations
batch_data = [
    {"step": 0, "loss": 0.5},
    {"step": 1, "loss": 0.4},
    {"step": 2, "loss": 0.3},
]
result = run.log_batch(batch_data)
print(f"Success rate: {result.success_rate}")
```

### Query and Compare Runs

```python
from optixlog.query import list_runs, get_run, compare_runs, get_metrics

# List all runs in a project
runs = list_runs(project_id="proj_123", api_key="your_key")

# Get specific run details
run_info = get_run(run_id="run_456", api_key="your_key")

# Compare multiple runs
comparison = compare_runs(
    run_ids=["run_1", "run_2", "run_3"],
    api_key="your_key"
)

# Get metrics for a run
metrics = get_metrics(run_id="run_456", api_key="your_key")
```

### Download Artifacts

```python
from optixlog.query import download_artifact

# Download logged files
download_artifact(
    run_id="run_456",
    artifact_key="simulation_results",
    save_path="./downloads/results.csv",
    api_key="your_key"
)
```

## API Reference

### Core Classes

#### `Optixlog`
Main client for interacting with OptixLog API.

**Methods:**
- `project(name: str, project_id: str = None) -> Project` - Get or create a project

#### `Project`
Represents a project container for organizing runs.

**Methods:**
- `run(name: str = None, config: dict = None, run_id: str = None) -> Run` - Create or get a run

#### `Run`
Individual experiment run for logging metrics and artifacts.

**Logging Methods:**
- `log(step: int, **metrics) -> MetricResult` - Log scalar metrics
- `log_config(config: dict) -> None` - Set run configuration
- `add_config(config: dict) -> None` - Add to existing configuration
- `log_image(key: str, image: PIL.Image) -> MediaResult` - Log PIL image
- `log_file(key: str, file_path: str) -> MediaResult` - Log arbitrary file
- `log_matplotlib(key: str, figure: matplotlib.Figure) -> MediaResult` - Log matplotlib figure
- `log_plot(key: str, x: list, y: list, **kwargs) -> MediaResult` - Create and log plot
- `log_array_as_image(key: str, array: np.ndarray, **kwargs) -> MediaResult` - Log numpy array as heatmap
- `log_histogram(key: str, data: list, **kwargs) -> MediaResult` - Log histogram
- `log_scatter(key: str, x: list, y: list, **kwargs) -> MediaResult` - Log scatter plot
- `log_batch(operations: list) -> BatchResult` - Batch logging operations

### Query Functions

All query functions are available in `optixlog.query`:

- `list_projects(api_key: str) -> List[ProjectInfo]`
- `list_runs(project_id: str, api_key: str) -> List[RunInfo]`
- `get_run(run_id: str, api_key: str) -> RunInfo`
- `get_artifacts(run_id: str, api_key: str) -> List[ArtifactInfo]`
- `download_artifact(run_id: str, artifact_key: str, save_path: str, api_key: str) -> str`
- `get_metrics(run_id: str, api_key: str) -> List[dict]`
- `compare_runs(run_ids: List[str], api_key: str) -> ComparisonResult`

## Configuration

### Environment Variables

- `OPTIXLOG_API_KEY` - Your API key (alternative to passing directly)
- `OPTIXLOG_API_URL` - Custom API endpoint (default: https://optixlog.com)

### Custom API Endpoint

```python
client = Optixlog(
    api_key="your_key",
    api_url="https://your-custom-endpoint.com"
)
```

## Error Handling

OptixLog provides comprehensive validation and helpful error messages:

```python
from optixlog.validators import ValidationError

try:
    run.log(step=0, metric=float('nan'))  # NaN detection
except ValidationError as e:
    print(f"Validation error: {e}")
```

## Examples

### FDTD Simulation with MEEP

```python
import meep as mp
from optixlog import Optixlog

client = Optixlog()
project = client.project("meep_photonic_crystal")
run = project.run("defect_cavity", config={
    "resolution": 20,
    "dimensions": "2D",
    "pml_layers": 1.0
})

# Run MEEP simulation
sim = mp.Simulation(...)
for step in range(100):
    sim.run(until=1)

    # Log field data
    ez_data = sim.get_array(component=mp.Ez)
    run.log_array_as_image(f"Ez_field_step_{step}", ez_data)

    # Log metrics
    energy = sim.electric_energy_in_box(...)
    run.log(step=step, energy=energy)
```

### Tidy3D Integration

```python
import tidy3d as td
from optixlog import Optixlog

client = Optixlog()
project = client.project("tidy3d_optimization")
run = project.run("waveguide_bend", config={
    "bend_radius": 5.0,
    "wavelength": 1.55
})

# Run Tidy3D simulation
sim = td.Simulation(...)
sim_data = td.web.run(sim, task_name="bend_optimization")

# Log results
transmission = sim_data["transmission"].values
run.log(step=0, transmission=float(transmission))
run.log_file("simulation_data", "sim_data.hdf5")
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=optixlog --cov-report=html

# Run specific test categories
pytest -m unit
pytest -m integration
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the GNU Lesser General Public License v2.1 - see the [LICENSE](LICENSE) file for details.

This license allows:
- Using OptixLog in both open source and proprietary projects
- Linking to OptixLog without open sourcing your entire application
- Modifying OptixLog (modifications must remain open source under LGPL)

## Roadmap

- [ ] Support for additional photonic simulation frameworks (Lumerical, COMSOL)
- [ ] Real-time collaboration features
- [ ] Advanced visualization dashboard
- [ ] Automatic hyperparameter optimization integration
- [ ] Cloud storage integration (S3, GCS)
- [ ] Jupyter notebook extensions

## Related Projects

- [Tidy3D](https://github.com/flexcompute/tidy3d) - GPU-accelerated electromagnetic simulation (LGPL v2.1)
- [MEEP](https://github.com/NanoComp/meep) - Free finite-difference time-domain simulation software
- [MLflow](https://github.com/mlflow/mlflow) - Machine learning experiment tracking

## Support

- Documentation: [https://optixlog.com/docs](https://optixlog.com/docs)
- Issues: [GitHub Issues](https://github.com/fluxboard/Optixlog/issues)
- Community: [Discussions](https://github.com/fluxboard/Optixlog/discussions)

## Citation

If you use OptixLog in your research, please cite:

```bibtex
@software{optixlog2025,
  title = {OptixLog: Experiment Tracking for Photonic Simulations},
  author = {FluxBoard Team},
  year = {2025},
  url = {https://github.com/fluxboard/Optixlog}
}
```

## Acknowledgments

OptixLog is inspired by and builds upon ideas from:
- Tidy3D's approach to photonic simulation
- MLflow's experiment tracking paradigm
- Weights & Biases' logging interface

---

Made with ⚡ by the Coupler team
