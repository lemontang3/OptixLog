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

# Install OptixLog SDK v0.0.4
pip install optixlog
# Or install with rich for colored output (recommended):
pip install optixlog[full]

# Set your API key
export OPTIX_API_KEY="proj_your_api_key_here"

# Run your first example
python demo.py
```

**New in v0.0.4:** 80% less boilerplate code with context managers and convenience helpers!

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

This repository contains **85+ comprehensive examples** covering a wide range of photonic simulations with OptixLog integration.

[![GitHub](https://img.shields.io/badge/GitHub-View%20Examples-blue?logo=github)](https://github.com/lemontang3/OptixLog)
[![GitHub Stars](https://img.shields.io/github/stars/lemontang3/OptixLog?style=social)](https://github.com/lemontang3/OptixLog/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/lemontang3/OptixLog?style=social)](https://github.com/lemontang3/OptixLog/network/members)

### 🚀 **Quick Start Examples**
- [`quick_start.py`](examples/quick_start.py) - Basic OptixLog integration and logging

### 🔬 **Basic Meep Simulations**
- [`straight_waveguide.py`](examples/straight_waveguide.py) - Simple straight waveguide with field visualization
- [`absorbed_1d.py`](examples/absorbed_1d.py) - 1D absorption simulation in aluminum
- [`bend_flux.py`](examples/bend_flux.py) - 90-degree waveguide bend transmission analysis
- [`binary_grating_phasemap.py`](examples/binary_grating_phasemap.py) - Binary grating phase map generation
- [`binary_grating_analysis.py`](examples/binary_grating_analysis.py) - Advanced binary grating diffraction analysis
- [`ring_resonator.py`](examples/ring_resonator.py) - Ring resonator mode calculation and analysis

### 🌐 **Waveguide & Transmission Examples**
- [`waveguide_crossing.py`](examples/waveguide_crossing.py) - Waveguide crossing analysis
- [`coupler.py`](examples/coupler.py) - Optical coupler simulation
- [`bent_waveguide.py`](examples/bent_waveguide.py) - Bent waveguide analysis
- [`ring_cyl.py`](examples/ring_cyl.py) - Cylindrical ring resonator
- [`ring_gds.py`](examples/ring_gds.py) - Ring resonator with GDS integration

### 📊 **Grating & Diffraction Examples**
- [`zone_plate.py`](examples/zone_plate.py) - Zone plate focusing analysis
- [`polarization_grating.py`](examples/polarization_grating.py) - Polarization grating simulation
- [`binary_grating_oblique.py`](examples/binary_grating_oblique.py) - Oblique binary grating
- [`binary_grating_n2f.py`](examples/binary_grating_n2f.py) - Near-to-far field analysis
- [`grating2d_triangular_lattice.py`](examples/grating2d_triangular_lattice.py) - 2D triangular lattice grating
- [`binary_grating_phasemap_advanced.py`](examples/binary_grating_phasemap_advanced.py) - Advanced binary grating phase map analysis

### 🔬 **Scattering & Radiation Examples**
- [`cherenkov_radiation.py`](examples/cherenkov_radiation.py) - Cherenkov radiation analysis
- [`dipole_in_vacuum_cyl_off_axis.py`](examples/dipole_in_vacuum_cyl_off_axis.py) - Off-axis dipole radiation
- [`differential_cross_section.py`](examples/differential_cross_section.py) - Differential scattering cross-section
- [`antenna_radiation.py`](examples/antenna_radiation.py) - Antenna radiation pattern
- [`point_dipole_cyl.py`](examples/point_dipole_cyl.py) - Point dipole in cylindrical geometry
- [`cylinder_cross_section.py`](examples/cylinder_cross_section.py) - Cylinder scattering cross-section
- [`mie_scattering.py`](examples/mie_scattering.py) - Mie scattering analysis

### 🏗️ **MPB (Eigenmode) Examples**
- [`mpb_hole_slab.py`](examples/mpb_hole_slab.py) - Holey slab eigenmode analysis
- [`mpb_tri_holes.py`](examples/mpb_tri_holes.py) - Triangular hole photonic crystal
- [`mpb_bragg.py`](examples/mpb_bragg.py) - Bragg reflector analysis
- [`mpb_strip.py`](examples/mpb_strip.py) - Strip waveguide eigenmodes
- [`mpb_sq_rods.py`](examples/mpb_sq_rods.py) - Square rod photonic crystal
- [`mpb_tri_rods.py`](examples/mpb_tri_rods.py) - Triangular rod photonic crystal
- [`mpb_line_defect.py`](examples/mpb_line_defect.py) - Line defect in photonic crystal
- [`mpb_tutorial.py`](examples/mpb_tutorial.py) - MPB tutorial examples
- [`mpb_diamond.py`](examples/mpb_diamond.py) - Diamond structure analysis
- [`mpb_honey_rods.py`](examples/mpb_honey_rods.py) - Honeycomb rod structure
- [`parallel_wvgs_mpb.py`](examples/parallel_wvgs_mpb.py) - Parallel waveguides MPB analysis
- [`mpb_bragg_sine.py`](examples/mpb_bragg_sine.py) - Sine-modulated Bragg structure
- [`mpb_data_analysis.py`](examples/mpb_data_analysis.py) - MPB data analysis techniques

### 🎯 **Cavity & Resonator Examples**
- [`holey_wvg_cavity.py`](examples/holey_wvg_cavity.py) - Holey waveguide cavity
- [`cavity_farfield.py`](examples/cavity_farfield.py) - Cavity far-field analysis
- [`planar_cavity_ldos.py`](examples/planar_cavity_ldos.py) - Planar cavity LDOS analysis
- [`cavity_arrayslice.py`](examples/cavity_arrayslice.py) - Cavity array slice analysis
- [`ring_mode_overlap.py`](examples/ring_mode_overlap.py) - Ring resonator mode overlap
- [`metal_cavity_ldos.py`](examples/metal_cavity_ldos.py) - Metal cavity LDOS analysis

### 🎛️ **Adjoint Optimization Examples**
- [`mode_converter.py`](examples/mode_converter.py) - Mode converter optimization
- [`binary_grating_levelset.py`](examples/binary_grating_levelset.py) - Level set binary grating optimization
- [`multilayer_opt.py`](examples/multilayer_opt.py) - Multilayer optimization

### 🔧 **Advanced & Specialized Examples**
- [`perturbation_theory.py`](examples/perturbation_theory.py) - Perturbation theory analysis
- [`eps_fit_lorentzian.py`](examples/eps_fit_lorentzian.py) - Lorentzian material fitting
- [`holey_wvg_bands.py`](examples/holey_wvg_bands.py) - Holey waveguide band structure
- [`gaussian_beam.py`](examples/gaussian_beam.py) - Gaussian beam propagation
- [`parallel_wvgs_force.py`](examples/parallel_wvgs_force.py) - Parallel waveguide force analysis
- [`3rd_harm_1d.py`](examples/3rd_harm_1d.py) - Third harmonic generation
- [`extraction_eff_ldos.py`](examples/extraction_eff_ldos.py) - Extraction efficiency and LDOS
- [`mode_coeff_phase.py`](examples/mode_coeff_phase.py) - Mode coefficient phase analysis
- [`absorbed_power_density.py`](examples/absorbed_power_density.py) - Absorbed power density analysis
- [`absorber_1d.py`](examples/absorber_1d.py) - 1D absorber analysis
- [`metasurface_lens.py`](examples/metasurface_lens.py) - Metasurface lens design
- [`chirped_pulse.py`](examples/chirped_pulse.py) - Chirped pulse propagation
- [`multilevel_atom.py`](examples/multilevel_atom.py) - Multilevel atom interaction
- [`disc_extraction_efficiency.py`](examples/disc_extraction_efficiency.py) - Disc extraction efficiency
- [`stochastic_emitter_line.py`](examples/stochastic_emitter_line.py) - Stochastic emitter line analysis
- [`oblique_planewave.py`](examples/oblique_planewave.py) - Oblique plane wave analysis
- [`stochastic_emitter_reciprocity.py`](examples/stochastic_emitter_reciprocity.py) - Stochastic emitter reciprocity
- [`mode_decomposition.py`](examples/mode_decomposition.py) - Mode decomposition analysis
- [`antenna_pec_ground_plane.py`](examples/antenna_pec_ground_plane.py) - PEC ground plane antenna
- [`finite_grating.py`](examples/finite_grating.py) - Finite grating analysis
- [`refl_quartz.py`](examples/refl_quartz.py) - Quartz reflection analysis
- [`solve_cw.py`](examples/solve_cw.py) - Continuous wave solver
- [`wvg_src.py`](examples/wvg_src.py) - Waveguide source analysis
- [`dipole_in_vacuum_1d.py`](examples/dipole_in_vacuum_1d.py) - 1D dipole in vacuum
- [`stochastic_emitter.py`](examples/stochastic_emitter.py) - Stochastic emitter analysis
- [`material_dispersion.py`](examples/material_dispersion.py) - Material dispersion analysis
- [`disc_radiation_pattern.py`](examples/disc_radiation_pattern.py) - Disc radiation pattern
- [`faraday_rotation.py`](examples/faraday_rotation.py) - Faraday rotation analysis
- [`phase_in_material.py`](examples/phase_in_material.py) - Phase analysis in materials
- [`diffracted_planewave.py`](examples/diffracted_planewave.py) - Diffracted plane wave
- [`perturbation_theory_2d.py`](examples/perturbation_theory_2d.py) - 2D perturbation theory

### 🔮 **Future Framework Examples**
- **Tidy3D Examples**: Coming soon - waveguide simulations, FOM tracking
- **Lumerical Examples**: Coming soon - FDTD/MODE integration  
- **COMSOL Examples**: Coming soon - data import and logging

## 🛠 Framework Support

| Framework | Examples | Status |
|-----------|----------|--------|
| **Meep** | 85+ examples (FDTD, MPB, Optimization) | ✅ Complete |
| **Tidy3D** | Coming Soon | 🚧 Planned |
| **Lumerical** | Coming Soon | 🚧 Planned |
| **COMSOL** | Coming Soon | 🚧 Planned |

### 📊 **Meep Coverage**
- **Basic Simulations**: Waveguides, bends, gratings, cavities
- **Advanced Physics**: Scattering, radiation, nonlinear effects
- **MPB Integration**: Photonic crystals, band structures, eigenmodes
- **Optimization**: Adjoint methods, level-set optimization
- **Specialized**: Stochastic emitters, material dispersion, perturbation theory

## 🎮 Getting Started

### 1. **Basic Integration** (SDK v0.0.4)

```python
import optixlog
import meep as mp
import numpy as np

# Use context manager (auto-cleanup!)
with optixlog.run("my_first_simulation", 
                   config={"resolution": 30, "material": "silicon"}) as client:
    
    # Your simulation code
    sim = mp.Simulation(...)
    
    # Log metrics during simulation
    for step in range(100):
        sim.run(until=1)
        field = sim.get_array(...)
        power = float(np.mean(np.abs(field)**2))
        
        # Get return values!
        result = client.log(step=step, power=power)
        if result and step % 10 == 0:
            print(f"✓ Step {step}: logged successfully")
    
    # Zero-boilerplate plot logging!
    client.log_array_as_image("field_plot", field, cmap='RdBu')
```

### 2. **Parameter Sweeps** (SDK v0.0.4)

```python
# Sweep over multiple parameters
wavelengths = np.linspace(0.4, 0.7, 20)
for wavelength in wavelengths:
    # Context manager per sweep
    with optixlog.run(f"sweep_{wavelength:.2f}",
                       config={"wavelength": wavelength}) as client:
        
        # Run simulation for this wavelength
        # ... simulation code ...
        
        # Log with return values
        result = client.log(transmission=transmission, reflection=reflection)
        print(f"✓ Logged sweep at λ={wavelength:.2f}: {result.url}")
        
        # Use convenience helpers
        client.log_plot("spectrum", frequencies, transmission, 
                        title=f"Transmission (λ={wavelength:.2f})")
```

### 3. **Artifact Management** (SDK v0.0.4 - 80% Less Code!)

```python
import matplotlib.pyplot as plt

with optixlog.run("artifact_demo") as client:
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(frequencies, transmission)
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Transmission')
    
    # ONE LINE! No manual save/upload/cleanup!
    result = client.log_matplotlib("transmission_spectrum", fig)
    print(f"✓ Plot uploaded: {result.url}")
    
    # Auto-detects content type!
    client.log_file("data", "results/data.csv")
    
    # Or use helper to create plot from data
    client.log_plot("quick_plot", x_data, y_data, title="My Plot")
```

**Boilerplate reduced by 80%!** No more manual `savefig()` → `PIL.Image` → `log_image()` → `os.remove()`.

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
