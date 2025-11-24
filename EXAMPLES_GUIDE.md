# OptixLog Examples Guide

Complete reference for all OptixLog examples demonstrating photonic simulation tracking.

## 🚀 Quick Start

```bash
# Install OptixLog
pip install optixlog

# Set your API key
export OPTIX_API_KEY="proj_your_api_key_here"

# Run the demo
python demo.py
```

## 📚 Core Files

### `demo.py`
Main demonstration of OptixLog features:
- Parameter sweep simulations
- Context managers for automatic cleanup
- Convenience helpers (log_plot, log_matplotlib)
- Return values with URLs
- Colored console output

### `requirements.txt`
All dependencies needed to run the examples.

### `CONTRIBUTING.md`
Guidelines for creating new examples with OptixLog.

## 🎯 Key Examples

### 1. Simple Simulations

**`straight_waveguide.py`**
- Basic straight waveguide FDTD simulation
- Demonstrates field visualization with `log_array_as_image()`
- Shows `log_matplotlib()` for plots
- ~150 lines total

**`ring_resonator.py`**
- Ring resonator mode analysis
- Harminv resonance detection
- Multiple visualization types
- `log_plot()` for mode frequencies
- ~180 lines total

### 2. Parameter Sweeps

**`binary_grating_hyperparameter_sweep.py`**
- Complex multi-parameter optimization
- Demonstrates multiple runs with context managers
- Phase map and transmittance visualization
- ~350 lines total

## 🎨 OptixLog Features

### Context Managers
```python
with optixlog.run("experiment", config={...}) as client:
    # Your code here
    client.log(step=0, metric=value)
# Automatic cleanup happens here!
```

### Matplotlib Logging
```python
fig, ax = plt.subplots()
ax.plot(x, y)
client.log_matplotlib("my_plot", fig)  # One line!
```

### Array Visualization
```python
field_data = sim.get_array(...)
client.log_array_as_image("field", field_data, cmap='hot')
```

### Data Plots
```python
client.log_plot("spectrum", frequencies, transmission, 
                 title="Transmission Spectrum")
```

### Return Values
```python
result = client.log(step=0, loss=0.5)
if result and result.success:
    print(f"✓ Logged: {result.url}")
```

## 📊 Example Categories

### Meep Examples (85+ files)
Located in `Meep Examples/` directory:

- **Waveguides**: straight, bent, coupled
- **Resonators**: ring, cavity, photonic crystal
- **Gratings**: binary, chirped, blazed
- **Sources**: gaussian beam, dipole, plane wave
- **Materials**: dispersion, anisotropy, nonlinearity
- **Optimization**: adjoint, level-set, topology
- **MPB**: band structures, eigenmodes

## 🛠 Tools

### `modernize_examples.py`
Automated script to update example files:
```bash
# Preview changes
python modernize_examples.py "Meep Examples/" --dry-run

# Apply transformations
python modernize_examples.py "Meep Examples/"
```

## 📖 Documentation

- **README.md**: Overview and quick start
- **CONTRIBUTING.md**: Example creation guidelines
- **requirements.txt**: Dependencies

## 🎯 Best Practices

### ✅ DO
- Use context managers (`with optixlog.run()`)
- Use convenience helpers (`log_matplotlib()`, `log_array_as_image()`)
- Check return values for feedback
- Log configuration at the start
- Include metadata with uploads

### ❌ DON'T
- Manually save/upload/cleanup files
- Ignore return values
- Create unnecessary temporary files
- Hardcode API keys in code

## 💡 Tips

1. **Start Simple**: Begin with `straight_waveguide.py` or `demo.py`
2. **Use Helpers**: Leverage `log_matplotlib()` and `log_array_as_image()`
3. **Context Managers**: Always use `with optixlog.run()` for automatic cleanup
4. **Return Values**: Check result objects for URLs and success status
5. **Metadata**: Add descriptive metadata to your uploads

## 🔗 Resources

- **OptixLog Dashboard**: https://optixlog.com
- **API Documentation**: https://optixlog.com/docs
- **Meep Documentation**: https://meep.readthedocs.io

---

*All examples use OptixLog SDK for experiment tracking and visualization.*

