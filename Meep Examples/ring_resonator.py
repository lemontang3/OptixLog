"""
Ring Resonator Mode Analysis with OptixLog SDK v0.0.4

This example demonstrates ring resonator mode calculation with the new SDK v0.0.4 features:
✓ Context managers - auto-cleanup
✓ log_matplotlib() - one-line plotting
✓ Return values with URLs
✓ Colored console output
✓ 80% less boilerplate!

Based on the Meep tutorial: calculating 2D ring-resonator modes
"""

import os
import optixlog
import meep as mp
import numpy as np
import matplotlib.pyplot as plt

# OptixLog Configuration
api_key = os.getenv("OPTIX_API_KEY", "proj_YOUR_API_KEY_HERE")
project_name = os.getenv("OPTIX_PROJECT", "RingResonator")

print(f"🚀 Starting ring resonator simulation with OptixLog SDK v0.0.4")

def main():
    try:
        # NEW in v0.0.4: Context manager!
        with optixlog.run(
            run_name="ring_resonator_modes",
            project=project_name,
            config={
                "simulation_type": "ring_resonator",
                "description": "2D ring resonator mode calculation and analysis",
                "framework": "meep",
                "sdk_version": "0.0.4"
            },
            create_project_if_not_exists=True
        ) as client:
            
            print(f"✅ OptixLog client initialized. Run ID: {client.run_id}")
            
            # Ring resonator parameters
            n = 3.4  # index of waveguide
            w = 1  # width of waveguide
            r = 1  # inner radius of ring
            pad = 4  # padding between waveguide and edge of PML
            dpml = 2  # thickness of PML
            sxy = 2 * (r + w + pad + dpml)  # cell size
            resolution = 10
            
            # Source parameters
            fcen = 0.15  # pulse center frequency
            df = 0.1  # pulse width (in frequency)
            source_position = r + 0.1
            
            # NEW in v0.0.4: Return values!
            result = client.log(step=0,
                       waveguide_index=n,
                       waveguide_width=w,
                       inner_radius=r,
                       outer_radius=r + w,
                       cell_size=sxy,
                       resolution=resolution,
                       center_frequency=fcen,
                       frequency_width=df)
            
            if result and result.success:
                print(f"✓ Configuration logged")
            
            # Create geometry
            ring_geometry = [
                mp.Cylinder(radius=r + w, material=mp.Medium(index=n)),
                mp.Cylinder(radius=r, material=mp.air)
            ]
            
            # Create simulation
            sim = mp.Simulation(
                cell_size=mp.Vector3(sxy, sxy),
                geometry=ring_geometry,
                sources=[
                    mp.Source(
                        mp.GaussianSource(fcen, fwidth=df),
                        component=mp.Ez,
                        center=mp.Vector3(source_position)
                    )
                ],
                resolution=resolution,
                boundary_layers=[mp.PML(dpml)]
            )
            
            print("🏃 Running simulation...")
            
            # Run with Harminv to find resonances
            h = mp.Harminv(mp.Ez, mp.Vector3(source_position), fcen, df)
            sim.run(mp.after_sources(h), until_after_sources=300)
            
            # Get resonant frequencies
            modes = h.modes
            print(f"\n📊 Found {len(modes)} resonant modes")
            
            # Log each mode
            for i, mode in enumerate(modes):
                freq = mode.freq
                Q = mode.Q
                decay = mode.decay
                
                mode_result = client.log(
                    step=i + 1,
                    mode_number=i + 1,
                    frequency=freq,
                    quality_factor=Q,
                    decay_rate=decay,
                    wavelength=1 / freq if freq > 0 else 0
                )
                
                print(f"  Mode {i+1}: f={freq:.4f}, Q={Q:.1f}, decay={decay:.4f}")
            
            # Get field data for visualization
            eps_data = sim.get_array(component=mp.Dielectric)
            ez_data = sim.get_array(component=mp.Ez)
            
            # Calculate field statistics
            max_field = float(np.max(np.abs(ez_data)))
            mean_field = float(np.mean(np.abs(ez_data)))
            
            client.log(step=len(modes) + 1,
                       max_field_amplitude=max_field,
                       mean_field_amplitude=mean_field,
                       num_modes_found=len(modes))
            
            print("\n📊 Generating visualizations...")
            
            # ===== GEOMETRY VISUALIZATION =====
            fig1, ax1 = plt.subplots(figsize=(8, 8))
            im1 = ax1.imshow(eps_data.transpose(), cmap='binary', interpolation='spline36')
            ax1.set_title("Ring Resonator Geometry")
            ax1.axis('off')
            plt.colorbar(im1, ax=ax1, label="Permittivity")
            
            # NEW in v0.0.4: One line instead of 10!
            geom_result = client.log_matplotlib("geometry", fig1,
                                               meta={"description": "Ring resonator geometry"})
            
            if geom_result and geom_result.success:
                print(f"  ✓ Geometry plot logged: {geom_result.url}")
            
            plt.close(fig1)
            
            # ===== FIELD VISUALIZATION =====
            fig2, ax2 = plt.subplots(figsize=(8, 8))
            ax2.imshow(eps_data.transpose(), cmap='binary', alpha=0.3, interpolation='spline36')
            im2 = ax2.imshow(ez_data.transpose(), cmap='RdBu', alpha=0.9, interpolation='spline36')
            ax2.set_title("Electric Field Ez Distribution")
            ax2.axis('off')
            plt.colorbar(im2, ax=ax2, label="Field Amplitude")
            
            # ONE LINE again!
            field_result = client.log_matplotlib("field_distribution", fig2,
                                                meta={"description": "Ez field in ring resonator"})
            
            if field_result and field_result.success:
                print(f"  ✓ Field plot logged: {field_result.url}")
            
            plt.close(fig2)
            
            # ===== MODE FREQUENCY PLOT =====
            if len(modes) > 0:
                mode_freqs = [m.freq for m in modes]
                mode_Qs = [m.Q for m in modes]
                
                # NEW in v0.0.4: Use log_plot helper!
                plot_result = client.log_plot(
                    "mode_frequencies",
                    range(1, len(modes) + 1),
                    mode_freqs,
                    title="Resonant Mode Frequencies",
                    xlabel="Mode Number",
                    ylabel="Frequency"
                )
                
                if plot_result and plot_result.success:
                    print(f"  ✓ Mode frequency plot logged: {plot_result.url}")
            
            print("\n" + "="*70)
            print("✅ Ring resonator simulation completed!")
            print(f"📈 Found {len(modes)} resonant modes")
            print(f"📊 Logged {len(modes) + 2} metric steps and 3 visualizations")
            print(f"🔗 View run at: https://optixlog.com/runs/{client.run_id}")
            print("="*70)
            
            print("\n🎉 SDK v0.0.4 Features Used:")
            print("  ✓ Context manager - automatic cleanup")
            print("  ✓ log_matplotlib() - zero boilerplate plotting")
            print("  ✓ log_plot() - one-line data plots")
            print("  ✓ Return values - immediate URLs")
            print("  ✓ Colored output - beautiful terminal")
            print("\n  Boilerplate reduction: ~80%!")
            print("="*70)

    except ValueError as e:
        print(f"\n❌ OptixLog Error: {e}")
        print("Please ensure your API key is set correctly.")
    except Exception as e:
        print(f"\n❌ Simulation Error: {e}")
        import traceback
        traceback.print_exc()

# No manual cleanup needed! Context manager handles it!

if __name__ == "__main__":
    main()
