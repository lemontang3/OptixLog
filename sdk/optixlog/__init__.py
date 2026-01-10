# Copyright (C) 2025 FluxBoard Team
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA

"""
OptixLog SDK - Experiment tracking for photonic simulations

Usage:
    from optixlog import Optixlog
    
    # Create client
    client = Optixlog(api_key)
    
    # Get project (creates if doesn't exist)
    project = client.project("my_project")
    
    # Create a run
    run = project.run("experiment_1")
    
    # Add configuration
    run.add_config({"lr": 0.001, "epochs": 100})
    
    # Log metrics
    for step in range(100):
        run.log(step=step, loss=0.5, accuracy=0.9)
    
    # Log matplotlib figures
    run.log_matplotlib("plot", fig)

Loop usage:
    for i in range(10):
        run = client.project("sweep").run(f"run_{i}")
        run.add_config({"param": i})
        run.log(step=0, result=i * 2)
"""

from .client import Optixlog, Project, Run, OxInvalidTaskError, _detect_mpi_environment
from .result_types import (
    MetricResult,
    MediaResult,
    BatchResult,
    RunInfo,
    ArtifactInfo,
    ProjectInfo,
    ComparisonResult
)
from .validators import ValidationError

__version__ = "0.2.0"
__all__ = [
    # Main classes
    "Optixlog",
    "Project",
    "Run",
    
    # Result types
    "MetricResult",
    "MediaResult",
    "BatchResult",
    "RunInfo",
    "ArtifactInfo",
    "ProjectInfo",
    "ComparisonResult",
    
    # Exceptions
    "ValidationError",
    "OxInvalidTaskError",
    
    # Query functions (lazy loaded)
    "list_runs",
    "get_run",
    "get_artifacts",
    "download_artifact",
    "get_metrics",
    "compare_runs",
    "list_projects",
    
    # Utility
    "get_mpi_info",
    "is_master_process",
]


def get_mpi_info():
    """
    Get current MPI environment information.
    
    Returns:
        Dictionary with is_master, rank, size, has_mpi
    """
    is_master, rank, size, mpi_comm = _detect_mpi_environment()
    return {
        "is_master": is_master,
        "rank": rank,
        "size": size,
        "has_mpi": mpi_comm is not None
    }


def is_master_process() -> bool:
    """
    Check if current process is the master process.
    
    Returns:
        True if master, False otherwise
    """
    is_master, _, _, _ = _detect_mpi_environment()
    return is_master


# Lazy load query functions to avoid import overhead
def __getattr__(name):
    """Lazy load query functions"""
    if name in ["list_runs", "get_run", "get_artifacts", "download_artifact",
                "get_metrics", "compare_runs", "list_projects"]:
        from . import query
        return getattr(query, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
