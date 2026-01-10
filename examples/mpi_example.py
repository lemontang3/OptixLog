#!/usr/bin/env python3
# Copyright (C) 2025 FluxBoard Team
# Licensed under LGPL v2.1

"""
MPI-Aware Logging Example

This script demonstrates how OptixLog automatically handles MPI environments.
Only the master rank (rank 0) will perform actual logging operations.

Run with: mpirun -n 4 python mpi_example.py
"""

import numpy as np
from optixlog import Optixlog, get_mpi_info

def main():
    # Get MPI information
    mpi_info = get_mpi_info()
    print(f"Rank {mpi_info['rank']}/{mpi_info['size']}: Starting simulation")

    # Initialize client - works on all ranks, but only rank 0 logs
    client = Optixlog(api_key="your_api_key_here")
    project = client.project("mpi_simulation")
    run = project.run(
        name=f"parallel_run_{np.random.randint(1000)}",
        config={
            "mpi_size": mpi_info["size"],
            "simulation_type": "distributed_fdtd"
        }
    )

    # Simulate distributed computation
    for step in range(100):
        # Each rank does its own computation
        local_result = np.random.rand() * (mpi_info['rank'] + 1)

        # Simulate some work
        computation = np.sum(np.random.rand(1000, 1000))

        # Safe to call from all ranks - only rank 0 actually logs
        # Other ranks return None silently
        run.log(
            step=step,
            local_result=float(local_result),
            computation_sum=float(computation)
        )

        if mpi_info['is_master'] and step % 20 == 0:
            print(f"Step {step}: Master rank logged metrics")

    if mpi_info['is_master']:
        print("\nSimulation complete! All metrics logged by master rank.")
    else:
        print(f"\nRank {mpi_info['rank']}: Computation complete.")

if __name__ == "__main__":
    main()
