"""
Multilevel Atom.Py with OptixLog Integration

Electromagnetic simulation

Based on the Meep tutorial: multilevel-atom.py
"""

import os
import optixlog
import meep as mp
import matplotlib.pyplot as plt
import numpy as np

# OptixLog Configuration
api_key = os.getenv("OPTIX_API_KEY", "proj_YOUR_API_KEY_HERE")
api_url = os.getenv("OPTIX_API_URL", "https://coupler.onrender.com")
project_name = os.getenv("OPTIX_PROJECT", "MeepExamples")

print(f"🚀 Initializing OptixLog client for project: {project_name}")

try:
    # Initialize OptixLog client
    client = optixlog.init(
        api_key=api_key,
        api_url=api_url,
        project=project_name,
        run_name="multilevel-atom_simulation",
        config={
            "simulation_type": "general",
            "description": "Electromagnetic simulation",
            "framework": "meep",
            "original_file": "multilevel-atom.py"
        },
        create_project_if_not_exists=True
    )
    print(f"✅ OptixLog client initialized. Run ID: {client.run_id}")
    print(f"🔗 View run at: https://optixlog.com/runs/{client.run_id}")


    # Log simulation parameters
    client.log(step=0,
        resolution=400,
        r=1,
        pad=1,
        dpml=1
    )

resolution = 400
ncav = 1.5  # cavity refractive index
Lcav = 1  # cavity length
dpad = 1  # padding thickness
dpml = 1  # PML thickness
sz = Lcav + dpad + dpml
cell_size = mp.Vector3(z=sz)
dimensions = 1
pml_layers = [mp.PML(dpml, side=mp.High)]

# For defining laser properties in MEEP, the transition rates / frequencies are specified in units of 2*pi*a/c.
# gamma_21 in MEEP is the Full-Width Half-Max, as opposed to gamma_perp, which is the HWHM in SALT.
# Additionally, the classical coupling element sigma = 2*theta^2*omega_a/hbar, where
# theta is the off-diagonal dipole matrix element.

# These different conventions can cause a bit of confusion when comparing against SALT, so here we perform
# this transformation explicitly.

omega_a = 40  # omega_a in SALT
freq_21 = omega_a / (2 * math.pi)  # emission frequency  (units of 2πc/a)

gamma_perp = 4  # HWHM in angular frequency, SALT
gamma_21 = (2 * gamma_perp) / (
    2 * math.pi
)  # FWHM emission linewidth in sec^-1 (units of 2πc/a)
# Note that 2*pi*gamma_21 = 2*gamma_perp in SALT.

theta = 1  # theta, the off-diagonal dipole matrix element, in SALT
sigma_21 = 2 * theta * theta * omega_a  # dipole coupling strength (hbar = 1)

# The gain medium in MEEP is allowed to have an arbitrary number of levels, and is not
# restricted to a two-level gain medium, as it simulates the populations of every individual
# atomic energy level.

# If you are using a 2 level gain model, you can compare against
# results which only simulate the atomic inversion, using the definitions
# gamma_parallel = pumping_rate + rate_21
# D_0 = (pumping_rate - rate_21)/(pumping_rate + rate_21) * N0

# In fact, even if you arn't using a 2 level gain model, you can compare against an effective
# two level model using the formula provided in Cerjan et al., Opt. Express 20, 474 (2012).

# Here, D_0 as written above is not yet in "SALT" units. To make this conversion,
# D_0 (SALT) = theta^2/(hbar*gamma_perp) * D_0 (as written above)

# Finally, note the lack of 4*pi in the above conversion that is written in many published SALT papers.
# This 4*pi comes from using Gaussian units, in which the displacement field, D = E + 4*pi*P, whereas
# in SI units, D = eps0*E + P, which is what MEEP uses.

# Gain medium pump and decay rates are specified in units of c/a.

rate_21 = 0.005  # non-radiative rate  (units of c/a)
N0 = 37  # initial population density of ground state
Rp = 0.0051  # pumping rate of ground to excited state
# so for example, these parameters have D_0 (SALT) = 0.0693.

# Make the actual medium in MEEP:
transitions = [
    mp.Transition(
        1,
        2,
        pumping_rate=Rp,
        frequency=freq_21,
        gamma=gamma_21,
        sigma_diag=mp.Vector3(sigma_21, 0, 0),
    ),
    mp.Transition(2, 1, transition_rate=rate_21),
]
ml_atom = mp.MultilevelAtom(sigma=1, transitions=transitions, initial_populations=[N0])
two_level = mp.Medium(index=ncav, E_susceptibilities=[ml_atom])

# Specify the cavity geometry:
geometry = [
    mp.Block(
        center=mp.Vector3(z=-0.5 * sz + 0.5 * Lcav),
        size=mp.Vector3(mp.inf, mp.inf, Lcav),
        material=two_level,
    )
]

sim = mp.Simulation(
    cell_size=cell_size,
    resolution=resolution,
    boundary_layers=pml_layers,
    geometry=geometry,
    dimensions=dimensions,
)

sim.init_sim()


def field_func(p):
    return 1 if p.z == -0.5 * sz + 0.5 * Lcav else 0


sim.fields.initialize_field(mp.Ex, field_func)

# Specify the end time:
endt = 7000
# Note that the total number of time steps run is endt*resolution*2. This is the origin of the extra
# factor of 2 in the definition of dt in fieldfft_meep.m.


def print_field(sim):
    fp = sim.get_field_point(
        mp.Ex, mp.Vector3(z=(-0.5 * sz) + Lcav + (0.5 * dpad))
    ).real
    print(f"field:, {sim.meep_time()}, {fp}")


sim.run(mp.after_time(endt - 250, print_field)
    
    # Log simulation completion
    client.log(step=1, simulation_completed=True), until=endt)
except ValueError as e:
    print(f"\n❌ OptixLog Error: {{e}}")
    print("Please ensure your API key and URL are correct.")
except Exception as e:
    print(f"\n❌ Simulation Error: {{e}}")

finally:
    # Clean up generated files
    import glob
    for file_path in glob.glob("*.png") + glob.glob("*.csv") + glob.glob("*.txt"):
        if os.path.exists(file_path):
            os.remove(file_path)
