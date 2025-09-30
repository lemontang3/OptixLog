"""
1D Absorption Simulation with OptixLog Integration

This script simulates electromagnetic field absorption in a 1D aluminum medium using Meep
and logs the results to OptixLog for visualization and tracking.

Usage:
    # Set your OptixLog API key
    export OPTIX_API_KEY="proj_your_api_key_here"
    
    # Run the simulation
    python absorbed_1d.py                    # Without PML (uses absorber)
    python absorbed_1d.py --pml             # With PML boundary layers

The script will:
1. Initialize a new OptixLog run in the "examples" project
2. Log simulation parameters (resolution, boundary conditions, etc.)
3. Run the 1D Meep simulation with field monitoring
4. Log field evolution data over time
5. Generate and log visualization plots
6. Provide a link to view results in OptixLog dashboard

If OptixLog is not available, the script will run normally without logging.
"""

import argparse
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

# Add the Coupler SDK to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Coupler', 'sdk'))

from meep.materials import Al
import meep as mp

# Import OptixLog SDK
import optixlog


def main(args):
    # Initialize OptixLog client
    try:
        config = {
            "simulation_type": "absorbed_1d",
            "material": "Aluminum",
            "dimensions": 1,
            "resolution": 40,
            "use_pml": args.pml,
            "wavelength": 0.803,
            "cell_size_z": 10
        }

        client = optixlog.init(
            project="MeepExamples",
            run_name=f"absorbed_1d_al_{'pml' if args.pml else 'absorber'}",
            config=config,
            create_project_if_not_exists=True
        )

        print(f"[OptixLog] Started run: absorbed_1d_al_{'pml' if args.pml else 'absorber'}")
        print(f"[OptixLog] Project: Examples")
        print(f"[OptixLog] Run ID: {client.run_id}")
        optixlog_enabled = True
    except Exception as e:
        print(f"[Warning] OptixLog initialization failed: {e}")
        print("[Info] Continuing without OptixLog...")
        client = None
        optixlog_enabled = False

    resolution = 40
    cell_size = mp.Vector3(z=10)

    boundary_layers = [
        mp.PML(1, direction=mp.Z) if args.pml else mp.Absorber(1, direction=mp.Z)
    ]

    wavelength = 0.803
    frequency = 1 / wavelength
    
    sources = [
        mp.Source(
            src=mp.GaussianSource(frequency, fwidth=0.1),
            center=mp.Vector3(),
            component=mp.Ex,
        )
    ]

    # Log simulation parameters
    if optixlog_enabled:
        client.log(0,
            resolution=resolution,
            cell_size_z=10,
            wavelength=wavelength,
            frequency=frequency,
            use_pml=args.pml,
            boundary_type="PML" if args.pml else "Absorber",
            material="Aluminum",
            source_component="Ex"
        )

    # Store field data for logging and visualization
    field_data = []
    time_data = []

    def print_stuff(sim):
        p = sim.get_field_point(mp.Ex, mp.Vector3())
        current_time = sim.meep_time()
        
        # Store data for logging
        field_data.append(p.real)
        time_data.append(current_time)
        
        print(f"ex:, {current_time}, {p.real}")
        
        # Log field data periodically (every 20 steps to avoid too much data)
        if optixlog_enabled and len(field_data) % 20 == 0:
            client.log(len(field_data) // 20,
                field_ex=p.real,
                time=current_time,
                field_magnitude=abs(p.real)
            )

    sim = mp.Simulation(
        cell_size=cell_size,
        resolution=resolution,
        dimensions=1,
        default_material=Al,
        boundary_layers=boundary_layers,
        sources=sources,
    )

    print(f"[Simulation] Starting 1D absorption simulation...")
    print(f"[Simulation] Boundary: {'PML' if args.pml else 'Absorber'}")
    print(f"[Simulation] Wavelength: {wavelength} μm")
    
    sim.run(
        mp.at_every(10, print_stuff),
        until_after_sources=mp.stop_when_fields_decayed(50, mp.Ex, mp.Vector3(), 1e-6),
    )

    # Analyze and log final results
    if field_data and optixlog_enabled:
        # Calculate absorption statistics
        max_field = max(field_data)
        min_field = min(field_data)
        final_field = field_data[-1] if field_data else 0
        field_decay = abs(final_field / max_field) if max_field != 0 else 0
        
        # Log final results
        client.log(len(field_data) // 20 + 1,
            max_field=max_field,
            min_field=min_field,
            final_field=final_field,
            field_decay_ratio=field_decay,
            total_time_steps=len(field_data),
            simulation_completed=True
        )

        print(f"[OptixLog] Logged simulation results to run {client.run_id}")
        
        # Create and log field evolution plot
        create_field_plot(field_data, time_data, wavelength, args.pml, client)
        
        print(f"[OptixLog] Logged visualization to run {client.run_id}")
        print(f"[OptixLog] View results at: https://optixlog.com/runs/{client.run_id}")
    else:
        print("[Info] Simulation completed successfully (without OptixLog)")


def create_field_plot(field_data, time_data, wavelength, use_pml, client):
    """Create and log a field evolution plot"""
    try:
        from PIL import Image
        
        plt.figure(figsize=(12, 8))
        
        # Plot field evolution
        plt.subplot(2, 1, 1)
        plt.plot(time_data, field_data, 'b-', linewidth=1.5, alpha=0.8)
        plt.xlabel('Time')
        plt.ylabel('Ex Field (real part)')
        plt.title(f'Field Evolution in 1D Aluminum Medium\nλ={wavelength} μm, Boundary: {"PML" if use_pml else "Absorber"}')
        plt.grid(True, alpha=0.3)
        
        # Plot field magnitude
        plt.subplot(2, 1, 2)
        field_magnitude = [abs(f) for f in field_data]
        plt.semilogy(time_data, field_magnitude, 'r-', linewidth=1.5, alpha=0.8)
        plt.xlabel('Time')
        plt.ylabel('|Ex Field|')
        plt.title('Field Magnitude (Log Scale)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("field_evolution_1d.png", dpi=150, bbox_inches="tight")
        
        # Log the plot to OptixLog
        field_img = Image.open("field_evolution_1d.png")
        client.log_image("field_evolution", field_img,
            meta={
                "stage": "results",
                "type": "field_evolution",
                "wavelength": wavelength,
                "boundary_type": "PML" if use_pml else "Absorber",
                "material": "Aluminum",
                "max_field": float(max(field_data)),
                "final_field": float(field_data[-1]) if field_data else 0
            })
        
        # Clean up temporary file
        os.remove("field_evolution_1d.png")
        
    except Exception as e:
        print(f"[Warning] Failed to create field plot: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="1D Electromagnetic Absorption Simulation with OptixLog Integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python absorbed_1d.py              # Run with absorber boundary
  python absorbed_1d.py --pml        # Run with PML boundary

Environment Variables:
  OPTIX_API_KEY                      # Your OptixLog API key for logging results
        """
    )
    parser.add_argument(
        "-pml", 
        action="store_true", 
        default=False, 
        help="Use PML (Perfectly Matched Layer) as boundary condition instead of absorber"
    )
    args = parser.parse_args()
    
    print(f"[1D Absorption] Starting simulation with {'PML' if args.pml else 'Absorber'} boundary...")
    main(args)