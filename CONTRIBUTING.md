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
import matplotlib.pyplot as plt

def main():
    """Main simulation function"""
    
    # Initialize OptixLog
    client = optixlog.init(
        run_name="example_name",
        config={
            "parameter1": value1,
            "parameter2": value2,
            "description": "Brief description"
        },
        create_project_if_not_exists=True
    )
    
    # Your simulation code here
    # ...
    
    # Log results
    client.log(metric1=value1, metric2=value2)
    
    # Upload artifacts if applicable
    # client.log_file("plot.png", "results/plot.png", "image/png")

if __name__ == "__main__":
    main()
```

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
