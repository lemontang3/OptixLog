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

This repository contains **85+ comprehensive examples** covering a wide range of photonic simulations with OptixLog integration.

### 🚀 **Quick Start Examples**
- [`01_quick_start.py`](examples/01_quick_start.py) - Basic OptixLog integration and logging

### 🔬 **Basic Meep Simulations**
- [`02_straight_waveguide.py`](examples/02_straight_waveguide.py) - Simple straight waveguide with field visualization
- [`04_absorbed_1d.py`](examples/04_absorbed_1d.py) - 1D absorption simulation in aluminum
- [`05_bend_flux.py`](examples/05_bend_flux.py) - 90-degree waveguide bend transmission analysis
- [`07_binary_grating_phasemap.py`](examples/07_binary_grating_phasemap.py) - Binary grating phase map generation
- [`08_binary_grating_analysis.py`](examples/08_binary_grating_analysis.py) - Advanced binary grating diffraction analysis
- [`10_mie_sphere_3d.py`](examples/10_mie_sphere_3d.py) - 3D Mie scattering from nanospheres
- [`12_ring_resonator.py`](examples/12_ring_resonator.py) - Ring resonator mode calculation and analysis

### 🌐 **Waveguide & Transmission Examples**
- [`39_waveguide_crossing.py`](examples/39_waveguide_crossing.py) - Waveguide crossing analysis
- [`67_coupler.py`](examples/67_coupler.py) - Optical coupler simulation
- [`86_bent_waveguide.py`](examples/86_bent_waveguide.py) - Bent waveguide analysis
- [`87_ring_cyl.py`](examples/87_ring_cyl.py) - Cylindrical ring resonator
- [`90_ring_gds.py`](examples/90_ring_gds.py) - Ring resonator with GDS integration

### 📊 **Grating & Diffraction Examples**
- [`26_zone_plate.py`](examples/26_zone_plate.py) - Zone plate focusing analysis
- [`36_polarization_grating.py`](examples/36_polarization_grating.py) - Polarization grating simulation
- [`37_binary_grating_oblique.py`](examples/37_binary_grating_oblique.py) - Oblique binary grating
- [`63_binary_grating_n2f.py`](examples/63_binary_grating_n2f.py) - Near-to-far field analysis
- [`77_grating2d_triangular_lattice.py`](examples/77_grating2d_triangular_lattice.py) - 2D triangular lattice grating
- [`82_binary_grating_phasemap.py`](examples/82_binary_grating_phasemap.py) - Binary grating phase map analysis

### 🔬 **Scattering & Radiation Examples**
- [`22_cherenkov_radiation.py`](examples/22_cherenkov_radiation.py) - Cherenkov radiation analysis
- [`23_dipole_in_vacuum_cyl_off_axis.py`](examples/23_dipole_in_vacuum_cyl_off_axis.py) - Off-axis dipole radiation
- [`32_differential_cross_section.py`](examples/32_differential_cross_section.py) - Differential scattering cross-section
- [`34_antenna_radiation.py`](examples/34_antenna_radiation.py) - Antenna radiation pattern
- [`38_point_dipole_cyl.py`](examples/38_point_dipole_cyl.py) - Point dipole in cylindrical geometry
- [`55_cylinder_cross_section.py`](examples/55_cylinder_cross_section.py) - Cylinder scattering cross-section
- [`81_mie_scattering.py`](examples/81_mie_scattering.py) - Mie scattering analysis

### 🏗️ **MPB (Eigenmode) Examples**
- [`21_mpb_hole_slab.py`](examples/21_mpb_hole_slab.py) - Holey slab eigenmode analysis
- [`25_mpb_tri_holes.py`](examples/25_mpb_tri_holes.py) - Triangular hole photonic crystal
- [`28_mpb_bragg.py`](examples/28_mpb_bragg.py) - Bragg reflector analysis
- [`30_mpb_strip.py`](examples/30_mpb_strip.py) - Strip waveguide eigenmodes
- [`46_mpb_sq_rods.py`](examples/46_mpb_sq_rods.py) - Square rod photonic crystal
- [`48_mpb_tri_rods.py`](examples/48_mpb_tri_rods.py) - Triangular rod photonic crystal
- [`53_mpb_line_defect.py`](examples/53_mpb_line_defect.py) - Line defect in photonic crystal
- [`69_mpb_tutorial.py`](examples/69_mpb_tutorial.py) - MPB tutorial examples
- [`70_mpb_diamond.py`](examples/70_mpb_diamond.py) - Diamond structure analysis
- [`74_mpb_honey_rods.py`](examples/74_mpb_honey_rods.py) - Honeycomb rod structure
- [`78_parallel_wvgs_mpb.py`](examples/78_parallel_wvgs_mpb.py) - Parallel waveguides MPB analysis
- [`92_mpb_bragg_sine.py`](examples/92_mpb_bragg_sine.py) - Sine-modulated Bragg structure
- [`94_mpb_data_analysis.py`](examples/94_mpb_data_analysis.py) - MPB data analysis techniques

### 🎯 **Cavity & Resonator Examples**
- [`45_holey_wvg_cavity.py`](examples/45_holey_wvg_cavity.py) - Holey waveguide cavity
- [`59_cavity_farfield.py`](examples/59_cavity_farfield.py) - Cavity far-field analysis
- [`60_planar_cavity_ldos.py`](examples/60_planar_cavity_ldos.py) - Planar cavity LDOS analysis
- [`65_cavity_arrayslice.py`](examples/65_cavity_arrayslice.py) - Cavity array slice analysis
- [`68_ring_mode_overlap.py`](examples/68_ring_mode_overlap.py) - Ring resonator mode overlap
- [`75_metal_cavity_ldos.py`](examples/75_metal_cavity_ldos.py) - Metal cavity LDOS analysis

### 🎛️ **Adjoint Optimization Examples**
- [`95_mode_converter.py`](examples/95_mode_converter.py) - Mode converter optimization
- [`96_binary_grating_levelset.py`](examples/96_binary_grating_levelset.py) - Level set binary grating optimization
- [`97_multilayer_opt.py`](examples/97_multilayer_opt.py) - Multilayer optimization

### 🔧 **Advanced & Specialized Examples**
- [`24_perturbation_theory.py`](examples/24_perturbation_theory.py) - Perturbation theory analysis
- [`27_eps_fit_lorentzian.py`](examples/27_eps_fit_lorentzian.py) - Lorentzian material fitting
- [`29_holey_wvg_bands.py`](examples/29_holey_wvg_bands.py) - Holey waveguide band structure
- [`31_gaussian_beam.py`](examples/31_gaussian_beam.py) - Gaussian beam propagation
- [`33_parallel_wvgs_force.py`](examples/33_parallel_wvgs_force.py) - Parallel waveguide force analysis
- [`40_3rd_harm_1d.py`](examples/40_3rd_harm_1d.py) - Third harmonic generation
- [`41_extraction_eff_ldos.py`](examples/41_extraction_eff_ldos.py) - Extraction efficiency and LDOS
- [`43_mode_coeff_phase.py`](examples/43_mode_coeff_phase.py) - Mode coefficient phase analysis
- [`44_absorbed_power_density.py`](examples/44_absorbed_power_density.py) - Absorbed power density analysis
- [`47_absorber_1d.py`](examples/47_absorber_1d.py) - 1D absorber analysis
- [`49_metasurface_lens.py`](examples/49_metasurface_lens.py) - Metasurface lens design
- [`51_chirped_pulse.py`](examples/51_chirped_pulse.py) - Chirped pulse propagation
- [`54_multilevel_atom.py`](examples/54_multilevel_atom.py) - Multilevel atom interaction
- [`57_disc_extraction_efficiency.py`](examples/57_disc_extraction_efficiency.py) - Disc extraction efficiency
- [`61_stochastic_emitter_line.py`](examples/61_stochastic_emitter_line.py) - Stochastic emitter line analysis
- [`62_oblique_planewave.py`](examples/62_oblique_planewave.py) - Oblique plane wave analysis
- [`64_stochastic_emitter_reciprocity.py`](examples/64_stochastic_emitter_reciprocity.py) - Stochastic emitter reciprocity
- [`66_mode_decomposition.py`](examples/66_mode_decomposition.py) - Mode decomposition analysis
- [`71_antenna_pec_ground_plane.py`](examples/71_antenna_pec_ground_plane.py) - PEC ground plane antenna
- [`72_finite_grating.py`](examples/72_finite_grating.py) - Finite grating analysis
- [`73_refl_quartz.py`](examples/73_refl_quartz.py) - Quartz reflection analysis
- [`76_solve_cw.py`](examples/76_solve_cw.py) - Continuous wave solver
- [`79_wvg_src.py`](examples/79_wvg_src.py) - Waveguide source analysis
- [`80_dipole_in_vacuum_1d.py`](examples/80_dipole_in_vacuum_1d.py) - 1D dipole in vacuum
- [`83_stochastic_emitter.py`](examples/83_stochastic_emitter.py) - Stochastic emitter analysis
- [`84_material_dispersion.py`](examples/84_material_dispersion.py) - Material dispersion analysis
- [`85_disc_radiation_pattern.py`](examples/85_disc_radiation_pattern.py) - Disc radiation pattern
- [`88_faraday_rotation.py`](examples/88_faraday_rotation.py) - Faraday rotation analysis
- [`89_phase_in_material.py`](examples/89_phase_in_material.py) - Phase analysis in materials
- [`91_diffracted_planewave.py`](examples/91_diffracted_planewave.py) - Diffracted plane wave
- [`93_perturbation_theory_2d.py`](examples/93_perturbation_theory_2d.py) - 2D perturbation theory

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
