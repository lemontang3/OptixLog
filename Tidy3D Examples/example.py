import tidy3d as td
import photonforge as pf
import optixlog
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import time

# Initialize OptixLog
meida_key = os.getenv("OPTIX_API_KEY")
client = optixlog.init(
    api_key=api_key,
    api_url=os.getenv("OPTIX_API_URL", "https://coupler.onrender.com"),
    project="Tidy3D_Project",
    run_name="simulation_run",
    config={"simulation_type": "FDTD", "framework": "Tidy3D"}
)

# Setup simulation
# ... your Tidy3D setup code ...

# Create monitors
field_mon = td.FieldMonitor(...)
flux_mon = td.FluxMonitor(...)

# Run simulation
start_time = time.time()
sim_data = sim.run()
computation_time = time.time() - start_time

# Extract and log metrics
client.log(step=0,
    computation_time_seconds=computation_time,
    simulation_size=float(sim.size),
    num_frequencies=len(sim.freqs)
)

# Extract flux data
flux_values = sim_data['flux_monitor'].flux.values
frequencies = sim_data['flux_monitor'].freq.values

for step, (freq, flux) in enumerate(zip(frequencies, flux_values), 1):
    client.log(step=step,
        frequency_ghz=freq * 1e-9,
        transmission=float(flux)
    )

# Extract and log field visualization
fig, ax = plt.subplots(figsize=(10, 8))
sim_data['field_monitor'].plot_field("E", "abs", ax=ax)
plt.savefig("field.png", dpi=150)
with open("field.png", "rb") as f:
    img = Image.open(io.BytesIO(f.read()))
    client.log_image("electric_field", img)

# Export data as CSV if needed
import pandas as pd
df = pd.DataFrame({
    'frequency_GHz': frequencies * 1e-9,
    'transmission': flux_values
})
df.to_csv('results.csv', index=False)
client.log_file("transmission_data", "results.csv", content_type="text/csv")