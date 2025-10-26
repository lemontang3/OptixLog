#!/usr/bin/env python3
"""
Simple script to run the binary grating hyperparameter sweep
"""

import subprocess
import sys
import os

def main():
    print("🚀 Starting Binary Grating Hyperparameter Sweep")
    print("=" * 50)
    
    # Check if API key is set
    api_key = os.getenv("OPTIX_API_KEY")
    if not api_key:
        print("⚠️  Warning: OPTIX_API_KEY not set")
        print("   Set it with: export OPTIX_API_KEY='proj_your_key_here'")
        print("   Continuing anyway...")
    else:
        print(f"✅ API key found: {api_key[:20]}...")
    
    # Run the hyperparameter sweep
    try:
        script_path = os.path.join(os.path.dirname(__file__), "binary_grating_hyperparameter_sweep.py")
        result = subprocess.run([sys.executable, script_path], check=True)
        print("\n🎉 Hyperparameter sweep completed successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Hyperparameter sweep failed with exit code: {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️  Hyperparameter sweep interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
