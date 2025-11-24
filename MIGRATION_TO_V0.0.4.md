# Migration Guide: OptixLog SDK v0.0.3 → v0.0.4

## 🎉 Welcome to SDK v0.0.4!

This guide helps you migrate your existing OptixLog code to take advantage of the new v0.0.4 features that reduce boilerplate by **80%**.

## 📋 What's New in v0.0.4

1. **✓ Colored Console Output** - Beautiful terminal feedback
2. **✓ Input Validation** - Catches NaN/Inf before API calls
3. **✓ Return Values** - Get URLs and metadata from operations
4. **✓ Context Managers** - Auto-cleanup with `with` statements
5. **✓ Convenience Helpers** - 80% less boilerplate!
6. **✓ Query Capabilities** - List runs, download artifacts
7. **✓ Batch Operations** - Parallel logging
8. **✓ Type Hints** - Full IDE support

## 🔄 Migration Patterns

### Pattern 1: Context Managers

**BEFORE (v0.0.3):**
```python
client = optixlog.init(
    api_key=api_key,
    api_url=api_url,
    project=project_name,
    run_name="my_experiment",
    config={...}
)

# Your code here
client.log(step=0, loss=0.5)

# No automatic cleanup
```

**AFTER (v0.0.4):**
```python
# Context manager handles everything!
with optixlog.run(
    run_name="my_experiment",
    config={...}
) as client:
    # Your code here
    client.log(step=0, loss=0.5)
    
# Auto-cleanup happens here!
```

**Benefits:**
- Automatic cleanup
- Cleaner code structure
- Better error handling
- Pythonic pattern

---

### Pattern 2: Matplotlib Logging (HUGE TIME SAVER!)

**BEFORE (v0.0.3) - 15 lines:**
```python
import io
from PIL import Image

plt.figure()
plt.plot(x, y)
plt.title("My Plot")

# Manual save to buffer
buf = io.BytesIO()
plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
buf.seek(0)

# Convert to PIL Image
pil_image = Image.open(buf)

# Upload
client.log_image("my_plot", pil_image)

# Cleanup
buf.close()
plt.close()
```

**AFTER (v0.0.4) - 1 line:**
```python
plt.figure()
plt.plot(x, y)
plt.title("My Plot")

# ONE LINE! 🎉
client.log_matplotlib("my_plot", plt.gcf())
```

**Boilerplate reduction: 93%!**

---

### Pattern 3: Array Visualization

**BEFORE (v0.0.3) - 10+ lines:**
```python
plt.figure()
plt.imshow(field_array, cmap='hot')
plt.colorbar()
plt.title("Field Data")

plot_path = "field_plot.png"
plt.savefig(plot_path, dpi=150)
plt.close()

client.log_file("field_plot", plot_path, "image/png")

# Manual cleanup
if os.path.exists(plot_path):
    os.remove(plot_path)
```

**AFTER (v0.0.4) - 1 line:**
```python
# Directly log numpy array as heatmap!
client.log_array_as_image("field_plot", field_array, cmap='hot', title="Field Data")
```

**Boilerplate reduction: 90%!**

---

### Pattern 4: Simple Plots from Data

**BEFORE (v0.0.3):**
```python
plt.figure(figsize=(10, 6))
plt.plot(x_data, y_data)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('My Data')
plt.grid(True)

plot_path = "data_plot.png"
plt.savefig(plot_path)
plt.close()

client.log_file("data_plot", plot_path, "image/png")
os.remove(plot_path)
```

**AFTER (v0.0.4):**
```python
# Create and log in one line!
client.log_plot("data_plot", x_data, y_data, 
                 title="My Data", xlabel="X", ylabel="Y")
```

---

### Pattern 5: Return Values

**BEFORE (v0.0.3):**
```python
client.log(step=0, loss=0.5)
# Returns None - no feedback

client.log_image("plot", img)
# Returns None - no URL
```

**AFTER (v0.0.4):**
```python
# Get feedback!
result = client.log(step=0, loss=0.5)
if result.success:
    print(f"✓ Logged successfully")

# Get URL directly!
media = client.log_image("plot", img)
print(f"View at: {media.url}")
```

---

### Pattern 6: File Upload

**BEFORE (v0.0.3):**
```python
# Manual content-type specification
client.log_file("data", "results.csv", "text/csv")
client.log_file("plot", "figure.png", "image/png")
```

**AFTER (v0.0.4):**
```python
# Auto-detects content-type!
client.log_file("data", "results.csv")
client.log_file("plot", "figure.png")
```

---

### Pattern 7: Batch Operations

**BEFORE (v0.0.3):**
```python
# Sequential logging
for step in range(100):
    client.log(step=step, loss=losses[step], acc=accs[step])
```

**AFTER (v0.0.4):**
```python
# Parallel logging!
metrics_batch = [
    {"step": i, "loss": losses[i], "acc": accs[i]} 
    for i in range(100)
]
result = client.log_batch(metrics_batch)
print(f"Success rate: {result.success_rate:.1f}%")
```

---

### Pattern 8: Multiple Plots Comparison

**NEW in v0.0.4 - Not possible before!**

```python
# Compare multiple series in one plot
client.log_multiple_plots("comparison", [
    (x, train_loss, "Train"),
    (x, val_loss, "Validation"),
], title="Loss Curves")
```

---

## 🚀 Complete Example Migration

### BEFORE (v0.0.3) - 45 lines:

```python
import os
import io
from PIL import Image
import optixlog
import numpy as np
import matplotlib.pyplot as plt

# Manual initialization
api_key = os.getenv("OPTIX_API_KEY")
client = optixlog.init(
    api_key=api_key,
    api_url="https://backend.optixlog.com",
    project="MyProject",
    run_name="experiment_1",
    config={"lr": 0.001}
)

# Logging
client.log(step=0, loss=0.5)

# Manual plot handling
plt.figure()
plt.plot([1,2,3], [1,4,2])
plot_path = "plot.png"
plt.savefig(plot_path)
plt.close()

# Manual upload
client.log_file("plot", plot_path, "image/png")

# Manual cleanup
if os.path.exists(plot_path):
    os.remove(plot_path)

# Field visualization
field = np.random.rand(100, 100)
plt.figure()
plt.imshow(field, cmap='hot')
plt.colorbar()
field_path = "field.png"
plt.savefig(field_path)
plt.close()

client.log_file("field", field_path, "image/png")
os.remove(field_path)

print("Done!")
```

### AFTER (v0.0.4) - 15 lines:

```python
import optixlog
import numpy as np
import matplotlib.pyplot as plt

# Context manager + auto-cleanup
with optixlog.run("experiment_1", config={"lr": 0.001}) as client:
    
    # Return values
    result = client.log(step=0, loss=0.5)
    
    # One-line plotting
    plt.figure()
    plt.plot([1,2,3], [1,4,2])
    client.log_matplotlib("plot", plt.gcf())
    
    # One-line array visualization
    field = np.random.rand(100, 100)
    client.log_array_as_image("field", field, cmap='hot')

print("Done!")
```

**Result: 67% less code, much cleaner!**

---

## 🛡️ Breaking Changes

### 1. Return Values Changed

**Impact:** Methods now return result objects instead of None

**Migration:**
```python
# Old: Ignore return value
client.log(step=0, loss=0.5)

# New: Optionally use return value
result = client.log(step=0, loss=0.5)
if result.success:
    print(f"Logged: {result}")
```

**Compatibility:** Code still works if you ignore return values

### 2. New Dependency: rich

**Impact:** SDK now requires `rich>=13.0.0`

**Migration:**
```bash
pip install optixlog  # Automatically installs rich
```

**Compatibility:** If rich fails to install, SDK falls back to plain output

### 3. Helper Methods Added

**Impact:** Client instances now have additional methods

**Migration:** No action needed - new methods are additions, not replacements

---

## ✅ Migration Checklist

For each file using OptixLog:

- [ ] Wrap main code in `with optixlog.run()` context manager
- [ ] Replace `plt.savefig()` → `log_image()` with `log_matplotlib()`
- [ ] Replace `plt.imshow(array)` → `savefig()` with `log_array_as_image()`
- [ ] Remove manual file cleanup (`os.remove()`)
- [ ] Use return values for validation (optional)
- [ ] Update API URL references
- [ ] Update requirements.txt to v0.0.4
- [ ] Test with colored output

---

## 🎯 Quick Wins

**Highest Impact Changes:**

1. **Replace manual matplotlib handling** → Use `log_matplotlib()`
   - **Time saved:** ~10 lines per plot
   
2. **Add context managers** → Use `with optixlog.run()`
   - **Benefit:** Automatic cleanup, better structure

3. **Use array helpers** → Use `log_array_as_image()`
   - **Time saved:** ~8 lines per array visualization

---

## 📚 Additional Resources

- **SDK README**: See `/Coupler/sdk/README.md` for full documentation
- **Demo**: Run `python demo.py` to see all features
- **Examples**: Check `/Meep Examples/` for updated examples

---

## 🆘 Need Help?

**Common Issues:**

**Q: Getting import errors?**
```bash
pip install --upgrade optixlog
pip install rich>=13.0.0
```

**Q: Context manager not working?**
```python
# Make sure you're using optixlog.run(), not optixlog.init()
with optixlog.run("experiment") as client:
    # Your code
```

**Q: Helpers not available?**
```python
# Helpers are auto-added when using optixlog.run() or optixlog.init()
client = optixlog.init(...)  # Helpers automatically added!
client.log_matplotlib(...)  # Now available
```

---

**Happy migrating! You'll love the 80% reduction in boilerplate code! 🎉**

