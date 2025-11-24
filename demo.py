#!/usr/bin/env python3
"""
OptixLog Demo

This demo showcases OptixLog features for experiment tracking:
- Context managers for automatic cleanup
- Convenience helpers (log_plot, log_matplotlib)
- Return values with URLs
- Colored console output
- Input validation (NaN/Inf detection)
- Auto content-type detection

Demonstrates a parameter sweep simulation with comprehensive logging.
"""

import math
import os
import numpy as np
import matplotlib.pyplot as plt
import optixlog

def simulate_series(a, b, c, y0=1.0, steps=60):
    """
    Simulate a 1D driven linear dynamical system.
    y_{t+1} = a * y_t + b * sin(c * t)
    
    Returns list of (step, value) tuples.
    """
    series = []
    y = y0
    for t in range(steps):
        series.append((t, y))
        y = a * y + b * math.sin(c * t)
    return series

def main(steps=60):
    """
    Run parameter sweep simulation with OptixLog
    """
    # Check API key
    api_key = os.getenv("OPTIX_API_KEY")
    if not api_key:
        print("⚠️  Warning: OPTIX_API_KEY environment variable not set!")
        print("   Set it with: export OPTIX_API_KEY='your-api-key'")
        print("   Get your API key from: https://optixlog.com")
        return
    
    # Base parameters
    base_a = 0.92
    base_b = 0.50
    base_c = 0.15
    
    # Create configurations
    configs = []
    for i in range(7):
        a = base_a + 0.01 * i          # slightly more persistent
        b = base_b * (1 + 0.10 * i)    # stronger forcing
        c = base_c * (1 + 0.05 * i)    # slightly higher frequency
        configs.append((f"config_{i+1}", a, b, c))
    
    print("=" * 70)
    print("OptixLog Demo - Parameter Sweep")
    print("=" * 70)
    print(f"\nRunning {len(configs)} configurations with OptixLog features:")
    print("  ✓ Context managers (auto-cleanup)")
    print("  ✓ Convenience helpers (log_plot)")
    print("  ✓ Return values with URLs")
    print("  ✓ Colored output")
    print("=" * 70)
    
    # Store results for final comparison plot
    all_series = []
    
    # Run each configuration
    for name, a, b, c in configs:
        print(f"\n{'='*70}")
        print(f"Running {name}: a={a:.3f}, b={b:.3f}, c={c:.3f}")
        print(f"{'='*70}")
        
        # Use context manager - auto-cleanup on exit
        with optixlog.run(
            run_name=name,
            project="Examples",
            config={
                "parameter_a": a,
                "parameter_b": b,
                "parameter_c": c,
                "initial_y": 1.0,
                "system_type": "discrete_linear_driven",
                "equation": "y_{t+1} = a * y_t + b * sin(c * t)"
            },
            create_project_if_not_exists=True
        ) as client:
            
            # Simulate system
            series = simulate_series(a, b, c, steps=steps)
            all_series.append((name, series))
            
            steps_data = [step for step, _ in series]
            values_data = [value for _, value in series]
            
            # Log metrics step by step
            y = 1.0
            for t in range(steps):
                # Returns MetricResult with success status
                result = client.log(
                    step=t,
                    y=y,
                    forcing_term=b * math.sin(c * t),
                    linear_term=a * y
                )
                
                # Update state
                y = a * y + b * math.sin(c * t)
                
                if t % 10 == 0 and result:
                    print(f"  Step {t:3d}: y={y:.4f} (logged ✓)")
            
            # Use convenience helper for plotting
            plot_result = client.log_plot(
                "timeseries",
                steps_data,
                values_data,
                title=f"{name}: System Evolution",
                xlabel="Step",
                ylabel="Value y(t)"
            )
            
            if plot_result and plot_result.success:
                print(f"  ✓ Plot uploaded: {plot_result.url}")
            
            # Log summary statistics
            max_val = max(values_data)
            min_val = min(values_data)
            mean_val = sum(values_data) / len(values_data)
            
            client.log(
                step=steps,
                summary=True,
                max_value=max_val,
                min_value=min_val,
                mean_value=mean_val
            )
            
            print(f"  ✓ Completed {name} - logged {steps} steps")
        
        # Context manager automatically handles cleanup!
    
    # Create final comparison plot
    print(f"\n{'='*70}")
    print("Creating comparison plot of all configurations...")
    print(f"{'='*70}")
    
    # Use context manager for final summary run
    with optixlog.run(
        run_name="comparison_summary",
        project="Examples",
        config={
            "type": "comparison",
            "num_configs": len(configs)
        },
        create_project_if_not_exists=True
    ) as client:
        
        # Create comparison plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for name, series in all_series:
            steps_data = [step for step, _ in series]
            values_data = [value for _, value in series]
            ax.plot(steps_data, values_data, label=name, linewidth=2)
        
        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel('Value y(t)', fontsize=12)
        ax.set_title('Parameter Sweep Comparison: All Configurations', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # log_matplotlib() - no manual save/upload/cleanup!
        comparison_result = client.log_matplotlib("comparison_all_configs", fig)
        
        if comparison_result and comparison_result.success:
            print(f"  ✓ Comparison plot uploaded")
            print(f"  URL: {comparison_result.url}")
        
        plt.close(fig)
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"✓ All {len(configs)} configurations completed!")
    print(f"{'='*70}")
    print(f"\n💡 View your runs at: https://optixlog.com")
    print(f"   Project: Examples")
    print(f"\n🎉 OptixLog Features Demonstrated:")
    print(f"   ✓ Context managers - auto-cleanup")
    print(f"   ✓ log_plot() - one-line plotting")
    print(f"   ✓ log_matplotlib() - zero boilerplate")
    print(f"   ✓ Return values - immediate feedback")
    print(f"   ✓ Colored output - beautiful terminal")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
