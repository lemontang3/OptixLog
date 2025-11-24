# Contributing to OptixLog Examples

Thank you for your interest in contributing to the OptixLog Examples repository! This document provides guidelines for contributing new examples, improvements, and bug fixes.

## 🤝 How to Contribute

### 1. **Fork and Clone**
```bash
git clone https://github.com/lemontang3/OptixLog.git
cd OptixLog
```

### 2. **Create a Branch**
```bash
git checkout -b feature/your-example-name
```

### 3. **Make Your Changes**
Follow the guidelines below for adding new examples or improving existing ones.

### 4. **Test Your Changes**
```bash
# Run your example to make sure it works
python examples/your_example.py

# Run the test suite (if available)
pytest tests/
```

### 5. **Submit a Pull Request**
Create a pull request with a clear description of your changes.

## 📝 Adding New Examples

### File Naming Convention
- Use numbered prefixes: `##_descriptive_name.py`
- Use underscores for spaces
- Be descriptive but concise

Examples:
- `08_photonic_crystal.py`
- `12_metasurface_design.py`
- `15_multi_parameter_sweep.py`

### Example Template

```python
"""
[Example Name] with OptixLog

Description: Brief description of what this simulation does
Physics: Explanation of the underlying physics

Features demonstrated:
- Context managers for automatic cleanup
- log_matplotlib() for one-line plotting  
- log_array_as_image() for direct array visualization
- Return values with URLs
- Colored console output

Author: Your Name
Date: YYYY-MM-DD
"""

import os
import optixlog
import meep as mp
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import matplotlib.pyplot as plt

def main():
    """Main simulation function with OptixLog tracking"""
    
    # Get API key
    api_key = os.getenv("OPTIX_API_KEY")
    if not api_key:
        print("❌ Error: OPTIX_API_KEY not set!")
        return
    
    # Use context manager for automatic cleanup
    with optixlog.run(
        run_name="example_name",
        project="MyProject",
        config={
            "parameter1": value1,
            "parameter2": value2,
            "description": "Brief description"
        },
        create_project_if_not_exists=True
    ) as client:
        
        print(f"✅ Run initialized: {client.run_id}")
        
        # Your simulation code here
        sim = mp.Simulation(...)
        sim.run(...)
        
        # Get return values for feedback
        result = client.log(step=0, metric1=value1, metric2=value2)
        if result and result.success:
            print(f"✓ Metrics logged: {result.url}")
        
        # Get field data
        field_data = sim.get_array(...)
        
        # Log arrays directly as images
        client.log_array_as_image("field_plot", field_data, cmap='hot')
        
        # Create matplotlib plot
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set_title("My Plot")
        
        # One line to log the plot!
        client.log_matplotlib("my_plot", fig)
        plt.close(fig)
        
    # Context manager handles all cleanup automatically!

if __name__ == "__main__":
    main()
```

## 🎨 OptixLog Best Practices

### ✅ DO Use These Features

**1. Context Managers (Required for all examples)**
```python
# ✅ CORRECT - Use context manager
with optixlog.run("experiment", config={...}) as client:
    # Your code here
    pass

# ❌ WRONG - Don't use init() without context manager
client = optixlog.init(...)  # Not recommended - use context manager
```

**2. Convenience Helpers (Eliminate boilerplate)**
```python
# ✅ CORRECT - One line!
client.log_matplotlib("plot", fig)

# ❌ WRONG - Manual save/upload/cleanup
plt.savefig("plot.png")
client.log_file("plot", "plot.png", "image/png")
os.remove("plot.png")  # Unnecessary boilerplate!
```

**3. Array Visualization**
```python
# ✅ CORRECT - Direct array logging
client.log_array_as_image("field", field_array, cmap='RdBu')

# ❌ WRONG - Manual conversion with temporary files
plt.imshow(field_array)
plt.savefig("field.png")
client.log_file("field", "field.png")
os.remove("field.png")
```

**4. Return Values**
```python
# ✅ CORRECT - Use return values
result = client.log(step=0, loss=0.5)
if result and result.success:
    print(f"✓ Logged: {result.url}")

# ❌ ACCEPTABLE but not ideal - Ignoring feedback
client.log(step=0, loss=0.5)  # Works, but you miss out on useful feedback
```

**5. Simple Data Plots**
```python
# ✅ CORRECT - Use helper
client.log_plot("spectrum", frequencies, transmission, 
                 title="Transmission Spectrum")

# ❌ WRONG - Manual matplotlib handling
fig, ax = plt.subplots()
ax.plot(frequencies, transmission)
plt.savefig("spectrum.png")
client.log_file("spectrum", "spectrum.png")
```

### ❌ DON'T Do These

1. **Don't use manual file cleanup** - Context managers handle it
2. **Don't specify api_url explicitly** - SDK defaults to correct URL
3. **Don't use old plt.savefig() → log_file() pattern** - Use log_matplotlib()
4. **Don't create temporary files** - Use helpers
5. **Don't ignore return values** - They provide useful feedback

### 📊 Example Comparison

**WITHOUT Convenience Helpers - 15 lines:**
```python
client = optixlog.init(api_key=..., project=...)

client.log(step=0, loss=0.5)

plt.figure()
plt.plot(x, y)
path = "plot.png"
plt.savefig(path)
plt.close()

from PIL import Image
img = Image.open(path)
client.log_image("plot", img)
os.remove(path)
```

**WITH Convenience Helpers - 7 lines:**
```python
with optixlog.run("experiment") as client:
    result = client.log(step=0, loss=0.5)
    
    plt.figure()
    plt.plot(x, y)
    client.log_matplotlib("plot", plt.gcf())
```

**Result: 53% less code + cleaner!**

### Documentation Requirements

Each example should include:

1. **Header Comment Block** with:
   - Descriptive title
   - Brief description of the simulation
   - Physics explanation
   - Usage instructions
   - Author and date

2. **Inline Comments** explaining:
   - Key parameters and their physical meaning
   - Important code sections
   - OptixLog integration points

3. **README Section** (add to main README.md):
   - Brief description
   - Key parameters
   - Expected results
   - Any special requirements

### Code Quality Standards

- **Python Style**: Follow PEP 8
- **Imports**: Group and sort imports properly
- **Error Handling**: Include try-catch blocks for OptixLog operations
- **Documentation**: Add docstrings for functions
- **Testing**: Test with different parameter values

### Example Categories

Organize examples by complexity and application:

1. **Quick Start** (01-03): Basic OptixLog integration
2. **1D Simulations** (04-06): One-dimensional problems
3. **2D Simulations** (07-09): Two-dimensional problems
4. **3D Simulations** (10-12): Three-dimensional problems
5. **Advanced** (13-15): Complex multi-parameter studies

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Example file** that's causing the issue
2. **Error message** (full traceback)
3. **Environment details**:
   - Python version
   - Meep version
   - Operating system
4. **Steps to reproduce**
5. **Expected vs actual behavior**

## 💡 Feature Requests

For new features or example requests:

1. **Check existing issues** first
2. **Describe the use case** clearly
3. **Provide example code** if possible
4. **Explain the expected benefit**

## 📋 Review Process

All contributions will be reviewed for:

- **Code quality** and style
- **Documentation** completeness
- **Functionality** and correctness
- **OptixLog integration** best practices
- **Educational value** for users

## 🎯 Best Practices

### OptixLog Integration

1. **Always use try-catch** around OptixLog operations
2. **Provide fallback behavior** if OptixLog is unavailable
3. **Log meaningful metrics** with descriptive names
4. **Upload relevant artifacts** (plots, data files)
5. **Use descriptive run names** and configuration

### Simulation Best Practices

1. **Use appropriate resolution** for the problem
2. **Include convergence checks** where applicable
3. **Provide parameter explanations** in comments
4. **Include visualization** of results
5. **Make examples reproducible** with fixed seeds where needed

### Documentation

1. **Write clear docstrings** for all functions
2. **Include physics explanations** for complex simulations
3. **Provide usage examples** in comments
4. **Update README** when adding new examples
5. **Include expected results** and interpretation

## 🚀 Getting Help

If you need help:

- **Open an issue** for questions
- **Join discussions** in GitHub Discussions
- **Email**: tanmayg@gatech.edu
- **Check documentation**: https://optixlog.com/docs

## 📄 License

By contributing, you agree that your contributions will be available for educational and example purposes.

---

Thank you for contributing to OptixLog Examples! 🎉
