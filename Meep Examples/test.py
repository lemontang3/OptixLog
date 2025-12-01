"""
OptixLog Connectivity & Logging Test

This script verifies that OptixLog is working correctly:
- Initializes a test project and run
- Logs a few synthetic metrics
- Uploads a small text artifact
- Confirms successful run completion
"""

import os
import optixlog
import time
import numpy as np

# --- Configuration ---
api_key = os.getenv("OPTIX_API_KEY", "proj_YOUR_API_KEY_HERE")
api_url = os.getenv("OPTIX_API_URL", "https://optixlog.com")
project_name = os.getenv("OPTIX_PROJECT", "OptixLog Demo Project")

print(f"🚀 Starting OptixLog connectivity test for project: {project_name}")

def main():
    try:
        # 1️⃣ Initialize OptixLog client
        client = optixlog.init(
            api_key=api_key,
            api_url=api_url,
            project=project_name,
            run_name="connectivity_test_run",
            config={
                "test_type": "connectivity_check",
                "description": "Quick test for OptixLog logging and artifact upload"
            },
            create_project_if_not_exists=True
        )
        print(f"✅ Initialized client. Run ID: {client.run_id}")
        print(f"🔗 View run at: https://optixlog.com/runs/{client.run_id}")

        # 2️⃣ Log some metrics
        for step in range(1000):
            dummy_loss = np.exp(-step) + np.random.uniform(0, 0.05)
            dummy_accuracy = 1 - dummy_loss
            client.log(step=step, loss=float(dummy_loss), accuracy=float(dummy_accuracy))
            print(f"📈 Logged step {step}: loss={dummy_loss:.4f}, acc={dummy_accuracy:.4f}")
            time.sleep(1)

        # 3️⃣ Log a small artifact (text file)
        artifact_path = "optixlog_test_artifact.txt"
        with open(artifact_path, "w") as f:
            f.write("This is a test artifact uploaded from optixlog_test.py\n")
            f.write(f"Run ID: {client.run_id}\n")

        client.log_file(
            "test_artifact",
            artifact_path,
            "text/plain",
            meta={"description": "Connectivity test artifact"}
        )
        print(f"📂 Uploaded artifact: {artifact_path}")

        # 4️⃣ Wrap up test
        client.log(step=99, test_status="completed", success=True)
        print("🏁 OptixLog test completed successfully!")

    except Exception as e:
        print(f"\n❌ Error during OptixLog test: {e}")

    finally:
        # Cleanup
        if os.path.exists("optixlog_test_artifact.txt"):
            os.remove("optixlog_test_artifact.txt")

if __name__ == "__main__":
    main()