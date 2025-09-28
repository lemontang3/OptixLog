# OptixLog Examples

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Meep](https://img.shields.io/badge/Meep-FDTD-green.svg)](https://meep.readthedocs.io)
[![OptixLog](https://img.shields.io/badge/OptixLog-AI%20Powered-orange.svg)](https://optixlog.com)

> **Comprehensive simulation examples for OptixLog** - AI-powered experiment tracking for photonic simulations

This repository contains detailed, production-ready examples demonstrating how to integrate OptixLog with various photonic simulation frameworks. Track your simulations, log metrics, and visualize results with ease.

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/lemontang3/OptixLog.git
cd OptixLog

# Install OptixLog
pip install http://optixlog.com/optixlog-0.0.1-py3-none-any.whl

# Set your API key
export OPTIX_API_KEY="proj_your_api_key_here"

# Run your first example
python examples/01_quick_start.py
```

## 📋 Table of Contents

- [Installation](#installation)
- [Examples Overview](#examples-overview)
- [Framework Support](#framework-support)
- [Getting Started](#getting-started)
- [Contributing](#contributing)
- [Support](#support)

## 🔧 Installation

### Prerequisites

- Python 3.8+
- Meep (for FDTD simulations)
- NumPy, Matplotlib (for data analysis)
- OptixLog (for experiment tracking)

### Setup

```bash
# Install Meep (choose your platform)
# macOS with Homebrew
brew install meep

# Ubuntu/Debian
sudo apt-get install meep meep-dev

# Install Python dependencies
pip install numpy matplotlib scipy

# Install OptixLog
pip install http://optixlog.com/optixlog-0.0.1-py3-none-any.whl
```

## 🎯 Examples Overview

### 📊 **Quick Start Examples**
- [`01_quick_start.py`](examples/01_quick_start.py) - Basic OptixLog integration
- [`02_simple_metrics.py`](examples/02_simple_metrics.py) - Logging scalar metrics
- [`03_artifact_upload.py`](examples/03_artifact_upload.py) - Uploading plots and data

### 🌊 **1D Simulations**
- [`04_absorbed_1d.py`](examples/04_absorbed_1d.py) - 1D absorption in aluminum
- [`05_absorbed_power_density.py`](examples/05_absorbed_power_density.py) - Power density analysis
- [`06_waveguide_mode.py`](examples/06_waveguide_mode.py) - Waveguide mode analysis

### 🔬 **2D Simulations**
- [`07_binary_grating_phasemap.py`](examples/07_binary_grating_phasemap.py) - Binary grating phase mapping
- [`08_photonic_crystal.py`](examples/08_photonic_crystal.py) - Photonic crystal band structure
- [`09_mach_zehnder.py`](examples/09_mach_zehnder.py) - Mach-Zehnder interferometer

### 🌐 **3D Simulations**
- [`10_mie_sphere_3d.py`](examples/10_mie_sphere_3d.py) - Mie scattering from nanospheres
- [`11_ring_resonator.py`](examples/11_ring_resonator.py) - Ring resonator analysis
- [`12_metasurface.py`](examples/12_metasurface.py) - Metasurface design optimization

### 📈 **Advanced Examples**
- [`13_parameter_sweep.py`](examples/13_parameter_sweep.py) - Automated parameter sweeps
- [`14_optimization.py`](examples/14_optimization.py) - Design optimization with OptixLog
- [`15_multi_project.py`](examples/15_multi_project.py) - Managing multiple projects

## 🛠 Framework Support

| Framework | Examples | Status |
|-----------|----------|--------|
| **Meep** | 1D, 2D, 3D FDTD | ✅ Complete |
| **Tidy3D** | Coming Soon | 🚧 Planned |
| **Lumerical** | Coming Soon | 🚧 Planned |
| **COMSOL** | Coming Soon | 🚧 Planned |

## 🎮 Getting Started

### 1. **Basic Integration**

```python
import optixlog
import meep as mp
import numpy as np

# Initialize OptixLog
client = optixlog.init(
    run_name="my_first_simulation",
    config={"resolution": 30, "material": "silicon"}
)

# Your simulation code
sim = mp.Simulation(...)

# Log metrics during simulation
for step in range(100):
    sim.run(until=1)
    field = sim.get_array(...)
    power = float(np.mean(np.abs(field)**2))
    client.log(step=step, power=power)

# Upload results
client.log_file("results.png", "field_plot.png", "image/png")
```

### 2. **Parameter Sweeps**

```python
# Sweep over multiple parameters
wavelengths = np.linspace(0.4, 0.7, 20)
for wavelength in wavelengths:
    client = optixlog.init(
        run_name=f"sweep_{wavelength:.2f}",
        config={"wavelength": wavelength}
    )
    
    # Run simulation for this wavelength
    # ... simulation code ...
    
    client.log(transmission=transmission, reflection=reflection)
```

### 3. **Artifact Management**

```python
# Save and upload plots
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(frequencies, transmission)
plt.xlabel('Frequency')
plt.ylabel('Transmission')
plt.savefig('transmission_spectrum.png')

# Upload to OptixLog
client.log_file(
    "transmission_spectrum.png", 
    "results/transmission_spectrum.png", 
    "image/png"
)

# Upload data files
np.savetxt('data.csv', data, delimiter=',')
client.log_file("data.csv", "results/data.csv", "text/csv")
```

## 📊 Key Features

- **🔍 Real-time Monitoring**: Watch your simulations progress live
- **📈 Automated Logging**: Metrics, parameters, and artifacts automatically tracked
- **🎨 Rich Visualizations**: Interactive plots and data exploration
- **🔗 Project Organization**: Group related simulations and compare results
- **🤖 AI Insights**: Get suggestions for better designs and parameters
- **📱 Web Dashboard**: Access your results from anywhere

## 🏗 Project Structure

```
OptixLog/
├── examples/                 # All simulation examples
│   ├── 01_quick_start.py
│   ├── 02_simple_metrics.py
│   ├── 03_artifact_upload.py
│   ├── 04_absorbed_1d.py
│   └── ...
├── data/                     # Sample data files
├── docs/                     # Additional documentation
│   ├── api_reference.md
│   ├── best_practices.md
│   └── troubleshooting.md
├── tests/                    # Test suite
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## 🔑 Configuration

### Environment Variables

```bash
# Required
export OPTIX_API_KEY="proj_your_api_key_here"
export OPTIX_API_URL="https://coupler.onrender.com"

# Optional
export OPTIX_PROJECT="my_project_name"
```

### Configuration Files

Create a `.env` file in your project root:

```env
OPTIX_API_KEY=proj_your_api_key_here
OPTIX_API_URL=https://coupler.onrender.com
OPTIX_PROJECT=my_project_name
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Adding New Examples

1. **Fork the repository**
2. **Create a new example** following the naming convention: `##_descriptive_name.py`
3. **Add comprehensive documentation** with:
   - Clear description of the physics
   - Parameter explanations
   - Expected results
4. **Test thoroughly** with different parameter values
5. **Submit a pull request**

### Example Template

```python
"""
[Example Name] with OptixLog Integration

Description: Brief description of what this simulation does
Physics: Explanation of the underlying physics
Usage: How to run and what parameters to adjust

Author: Your Name
Date: YYYY-MM-DD
"""

import optixlog
import meep as mp
import numpy as np

def main():
    # Initialize OptixLog
    client = optixlog.init(
        run_name="example_name",
        config={
            "parameter1": value1,
            "parameter2": value2
        }
    )
    
    # Your simulation code here
    # ...
    
    # Log results
    client.log(metric1=value1, metric2=value2)

if __name__ == "__main__":
    main()
```

## 🐛 Troubleshooting

### Common Issues

**Q: Getting "Invalid URL" error?**
A: Make sure your API key is set correctly: `export OPTIX_API_KEY="proj_your_key"`

**Q: 500 Internal Server Error?**
A: Check that your metrics don't contain NaN or Inf values

**Q: Can't upload files?**
A: Ensure file paths are correct and files exist before uploading

### Getting Help

- 📧 **Email**: tanmayg@gatech.edu
- 💬 **Issues**: [GitHub Issues](https://github.com/lemontang3/OptixLog/issues)
- 📚 **Docs**: [OptixLog Documentation](https://optixlog.com/docs)

## 📄 License

This repository contains examples and educational content. Please check individual files for any licensing requirements.

## 🙏 Acknowledgments

- **Meep Team** for the excellent FDTD simulation framework
- **Supabase** for the robust backend infrastructure
- **Photonic Community** for inspiration and feedback

---

<div align="center">

**Ready to supercharge your photonic simulations?** 🚀

[Get Started](https://optixlog.com) • [Documentation](https://optixlog.com/docs) • [Examples](examples/)

</div>
