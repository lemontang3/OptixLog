#!/usr/bin/env python3
# Copyright (C) 2025 FluxBoard Team
# Licensed under LGPL v2.1

"""
Basic OptixLog Usage Example

This script demonstrates the fundamental features of OptixLog:
- Initializing the client
- Creating projects and runs
- Logging metrics
- Logging visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
from optixlog import Optixlog

def main():
    # Initialize client (use OPTIXLOG_API_KEY environment variable or pass directly)
    client = Optixlog(api_key="your_api_key_here")

    # Create or get a project
    project = client.project(name="basic_example")
    print(f"Using project: {project.name}")

    # Create a new run
    run = project.run(
        name="example_run_1",
        config={
            "learning_rate": 0.001,
            "batch_size": 32,
            "optimizer": "adam"
        }
    )
    print(f"Created run: {run.name}")

    # Simulate a training loop and log metrics
    print("\nLogging metrics...")
    for step in range(100):
        # Simulate some metric values
        loss = 1.0 / (step + 1) + np.random.normal(0, 0.01)
        accuracy = 1.0 - loss + np.random.normal(0, 0.01)

        # Log metrics
        run.log(
            step=step,
            loss=float(loss),
            accuracy=float(accuracy)
        )

        if step % 20 == 0:
            print(f"Step {step}: loss={loss:.4f}, accuracy={accuracy:.4f}")

    # Create and log a matplotlib figure
    print("\nCreating visualization...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot 1: Loss curve
    steps = np.arange(100)
    losses = 1.0 / (steps + 1) + np.random.normal(0, 0.01, 100)
    ax1.plot(steps, losses)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Accuracy curve
    accuracies = 1.0 - losses + np.random.normal(0, 0.01, 100)
    ax2.plot(steps, accuracies)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training Accuracy")
    ax2.grid(True, alpha=0.3)

    # Log the figure
    run.log_matplotlib("training_curves", fig)
    print("Logged training curves visualization")

    # Log a numpy array as an image
    field_data = np.random.rand(50, 50)
    run.log_array_as_image("field_pattern", field_data, cmap="viridis")
    print("Logged field pattern visualization")

    print("\nExample complete!")

if __name__ == "__main__":
    main()
