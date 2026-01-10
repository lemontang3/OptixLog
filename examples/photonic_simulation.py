#!/usr/bin/env python3
# Copyright (C) 2025 FluxBoard Team
# Licensed under LGPL v2.1

"""
Photonic Simulation Example

This script demonstrates using OptixLog with photonic simulations.
It simulates a waveguide parameter sweep and logs results.
"""

import numpy as np
import matplotlib.pyplot as plt
from optixlog import Optixlog

def simulate_waveguide(width_nm, wavelength_nm):
    """
    Simulate a waveguide and return transmission/reflection.
    This is a placeholder - replace with actual MEEP or Tidy3D simulation.
    """
    # Placeholder simulation
    transmission = 0.8 + 0.15 * np.sin(width_nm / 100) * np.random.uniform(0.9, 1.1)
    reflection = 1.0 - transmission - 0.05 * np.random.rand()

    # Simulate field profile
    field_profile = np.exp(-((np.linspace(-width_nm, width_nm, 100) / width_nm) ** 2))
    field_2d = field_profile[:, np.newaxis] * field_profile[np.newaxis, :]

    return transmission, reflection, field_2d

def main():
    # Initialize OptixLog
    client = Optixlog(api_key="your_api_key_here")
    project = client.project("waveguide_optimization")

    # Parameter sweep
    wavelength = 1550  # nm
    widths = np.linspace(300, 700, 20)  # nm

    # Track best result
    best_transmission = 0
    best_width = None

    print("Starting waveguide parameter sweep...")

    for idx, width in enumerate(widths):
        # Create a run for each width
        run = project.run(
            name=f"width_{int(width)}nm",
            config={
                "waveguide_width_nm": float(width),
                "wavelength_nm": wavelength,
                "resolution": 20,
                "simulation_time": 1000
            }
        )

        print(f"\nSimulating width = {width:.1f} nm...")

        # Run simulation
        transmission, reflection, field = simulate_waveguide(width, wavelength)

        # Log metrics
        run.log(
            step=0,
            transmission=float(transmission),
            reflection=float(reflection),
            loss=float(1.0 - transmission - reflection)
        )

        # Log field profile
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(field, extent=[-width, width, -width, width], cmap='RdBu')
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("y (nm)")
        ax.set_title(f"Field Profile - Width {width:.0f} nm")
        plt.colorbar(im, ax=ax, label="Field Intensity")

        run.log_matplotlib("field_profile", fig)
        plt.close(fig)

        # Log the 2D field as an image
        run.log_array_as_image("field_2d", field, cmap="RdBu")

        # Track best result
        if transmission > best_transmission:
            best_transmission = transmission
            best_width = width

        print(f"  Transmission: {transmission:.4f}")
        print(f"  Reflection: {reflection:.4f}")

    # Create summary run
    summary_run = project.run(
        name="sweep_summary",
        config={
            "sweep_range": f"{widths[0]}-{widths[-1]} nm",
            "num_points": len(widths),
            "best_width_nm": float(best_width),
            "best_transmission": float(best_transmission)
        }
    )

    # Create summary plot
    transmissions = []
    for width in widths:
        t, _, _ = simulate_waveguide(width, wavelength)
        transmissions.append(t)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(widths, transmissions, 'o-', linewidth=2, markersize=8)
    ax.axhline(y=best_transmission, color='r', linestyle='--',
               label=f'Best: {best_transmission:.4f} @ {best_width:.0f} nm')
    ax.set_xlabel("Waveguide Width (nm)", fontsize=12)
    ax.set_ylabel("Transmission", fontsize=12)
    ax.set_title("Waveguide Width Optimization", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    summary_run.log_matplotlib("optimization_curve", fig)

    print(f"\n{'='*60}")
    print(f"Optimization Complete!")
    print(f"Best width: {best_width:.1f} nm")
    print(f"Best transmission: {best_transmission:.4f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
