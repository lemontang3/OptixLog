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
OptixLog SDK - Fluent API for experiment tracking

Usage:
    from optixlog import Optixlog
    
    client = Optixlog(api_key="your_api_key")
    project = client.project(name="my_project")
    run = project.run(name="experiment_1", config={"lr": 0.001})
    
    run.log(step=0, loss=0.5, accuracy=0.8)
    run.log_matplotlib("plot", fig)
"""

import requests
import json
import io
import os
import base64
from typing import Optional, Dict, Any, List, Union, Tuple, TYPE_CHECKING
from PIL import Image
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from rich.console import Console
    console = Console()
except ImportError:
    console = None

from .validators import (
    validate_metrics,
    validate_file_path,
    validate_image,
    validate_step,
    validate_key,
    guess_content_type,
)
from .result_types import MetricResult, MediaResult, BatchResult

if TYPE_CHECKING:
    import numpy as np

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import matplotlib
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


DEFAULT_API_URL = "https://optixlog.com"


def _print(message: str, style: str = ""):
    """Print with rich colors if available, otherwise plain text"""
    if console:
        console.print(message, style=style)
    else:
        print(message)


def _detect_mpi_environment():
    """
    Detect MPI environment and return rank information.
    Returns: (is_master, rank, size, mpi_comm)
    """
    is_master = True
    rank = 0
    size = 1
    mpi_comm = None
    
    if 'OMPI_COMM_WORLD_RANK' in os.environ:
        rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        is_master = (rank == 0)
        return is_master, rank, size, None
    elif 'PMI_RANK' in os.environ:
        rank = int(os.environ['PMI_RANK'])
        size = int(os.environ['PMI_SIZE'])
        is_master = (rank == 0)
        return is_master, rank, size, None
    elif 'MPI_LOCALRANKID' in os.environ:
        rank = int(os.environ['MPI_LOCALRANKID'])
        size = int(os.environ['MPI_LOCALNRANKS'])
        is_master = (rank == 0)
        return is_master, rank, size, None
    
    try:
        import mpi4py.MPI as MPI
        try:
            mpi_comm = MPI.COMM_WORLD
            rank = mpi_comm.Get_rank()
            size = mpi_comm.Get_size()
            is_master = (rank == 0)
            return is_master, rank, size, mpi_comm
        except:
            if MPI.Is_initialized():
                mpi_comm = MPI.COMM_WORLD
                rank = mpi_comm.Get_rank()
                size = mpi_comm.Get_size()
                is_master = (rank == 0)
                return is_master, rank, size, mpi_comm
            else:
                return True, 0, 1, None
    except ImportError:
        pass
    
    try:
        import meep as mp
        is_master = mp.am_master()
        rank = 0 if is_master else 1
        size = 2 if not is_master else 1
        return is_master, rank, size, None
    except ImportError:
        pass
    
    return True, 0, 1, None


class OxInvalidTaskError(Exception):
    """Raised when a task_id is invalid or doesn't belong to the project"""
    pass


class Optixlog:
    """
    Main OptixLog client - entry point for the SDK.
    
    Example:
        client = Optixlog(api_key="your_key")
        project = client.project(name="my_project")
        run = project.run(name="experiment_1", config={"lr": 0.001})
        run.log(step=0, loss=0.5)
    """
    
    def __init__(self, *, api_key: str, api_url: Optional[str] = None):
        """
        Initialize OptixLog client.
        
        Args:
            api_key: Your OptixLog API key (required, keyword-only)
            api_url: API URL (defaults to https://optixlog.com)
        """
        self.api_key = api_key.strip() if api_key else None
        self.api_url = (api_url or os.getenv("OPTIX_API_URL", DEFAULT_API_URL)).rstrip("/")
        self.headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        
        self.is_master, self.rank, self.size, self.mpi_comm = _detect_mpi_environment()
        
        if not self.api_key:
            raise ValueError("Missing API key. Get your API key from: https://optixlog.com")
        
        self._projects_cache: Optional[List[Dict[str, Any]]] = None
    
    def _fetch_projects(self) -> List[Dict[str, Any]]:
        """Fetch available projects from the server"""
        if self._projects_cache is not None:
            return self._projects_cache
        
        try:
            response = requests.get(
                f"{self.api_url}/api/sdk/initialize-run-check",
                headers=self.headers
            )
            
            if response.status_code == 401:
                raise ValueError("Invalid API Key. Get your API key from https://optixlog.com")
            elif response.status_code == 403:
                raise ValueError("API Key access denied")
            elif not response.ok:
                raise ValueError(f"Server error ({response.status_code})")
            
            data = response.json()
            projects: List[Dict[str, Any]] = data.get("projects", [])
            self._projects_cache = projects
            return projects
            
        except requests.exceptions.ConnectionError:
            raise ValueError(f"Cannot connect to OptixLog server at {self.api_url}")
    
    def project(self, *, name: str, create_if_not_exists: bool = False) -> "Project":
        """
        Get a project by name or ID.
        
        Args:
            name: Project name or ID (required, keyword-only)
            create_if_not_exists: Create the project if it doesn't exist (default: False)
        
        Returns:
            Project instance
            
        Example:
            project = client.project(name="my_project")
        """
        if not self.is_master:
            return Project(self, name, name)
        
        projects = self._fetch_projects()
        
        project_id = None
        project_name = None
        for p in projects:
            if p["id"] == name or p["name"] == name:
                project_id = p["id"]
                project_name = p["name"]
                break
        
        if not project_id:
            if create_if_not_exists:
                project_id = self._create_project(name)
                project_name = name
                self._projects_cache = None
            else:
                available = [p["name"] for p in projects]
                raise ValueError(
                    f"Project '{name}' not found. "
                    f"Available: {', '.join(available) if available else 'none'}"
                )
        
        return Project(self, project_id, project_name or name)
    
    def _create_project(self, name: str) -> str:
        """Create a new project and return its ID"""
        try:
            response = requests.post(
                f"{self.api_url}/api/sdk/create-project",
                headers=self.headers,
                json={"name": name}
            )
            
            if response.status_code == 401:
                raise ValueError("Invalid API Key")
            elif not response.ok:
                raise ValueError(f"Failed to create project ({response.status_code})")
            
            result = response.json()
            project_id = result["id"]
            
            if result.get("created", True):
                _print(f"✓ Project '{name}' created", "green bold")
            else:
                _print(f"✓ Project '{name}' found", "green")
            
            return project_id
                
        except requests.exceptions.ConnectionError:
            raise ValueError(f"Cannot connect to server at {self.api_url}")
    
    def list_projects(self) -> List[Dict[str, Any]]:
        """List all available projects"""
        return self._fetch_projects()


class Project:
    """Represents an OptixLog project."""
    
    def __init__(self, client: Optixlog, project_id: str, project_name: str):
        self._client = client
        self.id = project_id
        self.name = project_name
    
    def run(self, *, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> "Run":
        """
        Create a new run in this project.
        
        Args:
            name: Optional name for the run (keyword-only)
            config: Optional configuration dictionary (keyword-only)
        
        Returns:
            Run instance
            
        Example:
            run = project.run(name="experiment_1", config={"lr": 0.001})
        """
        return Run(self._client, self, name, config)
    
    def __repr__(self) -> str:
        return f"Project(id='{self.id}', name='{self.name}')"


class Run:
    """Represents a single experiment run with all logging capabilities."""
    
    def __init__(self, client: Optixlog, project: Project, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        self._client = client
        self._project = project
        self.name = name
        self.run_id: Optional[str] = None
        self._config: Dict[str, Any] = config.copy() if config else {}
        self._initialized = False
        
        self.is_master = client.is_master
        self.rank = client.rank
        self.size = client.size
        self.mpi_comm = client.mpi_comm
        
        if self.is_master:
            self._initialize()
    
    def _initialize(self):
        """Initialize the run on the server"""
        if self._initialized:
            return
        
        try:
            response = requests.post(
                f"{self._client.api_url}/api/sdk/create-run",
                headers=self._client.headers,
                json={
                    "name": self.name,
                    "config": self._config,
                    "project_id": self._project.id
                }
            )
            
            if response.status_code == 401:
                raise ValueError("Invalid API Key")
            elif not response.ok:
                raise ValueError(f"Failed to create run ({response.status_code})")
            
            self.run_id = response.json()["id"]
            self._initialized = True
            _print(f"✓ Run initialized: {self.run_id}", "green bold")
            
        except requests.exceptions.ConnectionError:
            raise ValueError(f"Cannot connect to server at {self._client.api_url}")
    
    def set_config(self, config: Dict[str, Any]) -> "Run":
        """Add configuration to this run (chainable)."""
        self._config.update(config)
        
        if self._initialized and self.is_master:
            try:
                requests.post(
                    f"{self._client.api_url}/api/sdk/update-run-config",
                    headers=self._client.headers,
                    json={"run_id": self.run_id, "config": self._config}
                )
            except:
                pass
        
        return self
    
    def log(self, step: int, **kv) -> Optional[MetricResult]:
        """Log metrics for a specific step."""
        if not self.is_master:
            return None
        
        valid, error = validate_step(step)
        if not valid:
            _print(f"✗ Invalid step: {error}", "red")
            return MetricResult(step=step, metrics=kv, success=False, error=error)
        
        valid, error = validate_metrics(kv)
        if not valid:
            _print(f"✗ Invalid metrics: {error}", "red")
            return MetricResult(step=step, metrics=kv, success=False, error=error)
        
        try:
            payload = {"run_id": self.run_id, "step": step, "kv": kv}
            response = requests.post(
                f"{self._client.api_url}/api/sdk/log-metric",
                headers=self._client.headers,
                json=payload
            )
            
            if response.status_code == 401:
                error = "Invalid API Key"
                _print(f"✗ {error}", "red")
                return MetricResult(step=step, metrics=kv, success=False, error=error)
            elif not response.ok:
                error = f"Server error ({response.status_code})"
                _print(f"✗ {error}", "red")
                return MetricResult(step=step, metrics=kv, success=False, error=error)
            
            return MetricResult(
                step=step,
                metrics=kv,
                success=True,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            error = str(e)
            _print(f"✗ Failed to log metrics: {error}", "red")
            return MetricResult(step=step, metrics=kv, success=False, error=error)

    def log_image(self, key: str, pil_image: Image.Image, 
                   meta: Optional[Dict[str, Any]] = None) -> Optional[MediaResult]:
        """Log a PIL image."""
        if not self.is_master:
            return None
        
        valid, error = validate_key(key)
        if not valid:
            _print(f"✗ Invalid key: {error}", "red")
            return MediaResult(key=key, success=False, error=error)
        
        valid, error = validate_image(pil_image)
        if not valid:
            _print(f"✗ Invalid image: {error}", "red")
            return MediaResult(key=key, success=False, error=error)
        
        try:
            buf = io.BytesIO()
            pil_image.save(buf, format="PNG")
            image_bytes = buf.getvalue()
            file_size = len(image_bytes)
            
            blob_base64 = base64.b64encode(image_bytes).decode("utf-8")
            
            payload = {
                "run_id": self.run_id,
                "kind": "image",
                "key": key,
                "meta": json.dumps(meta or {}),
                "blob": blob_base64,
                "filename": "image.png",
                "content_type": "image/png",
            }
            
            response = requests.post(
                f"{self._client.api_url}/api/sdk/log-image",
                headers=self._client.headers,
                json=payload
            )
            
            if not response.ok:
                error = f"Upload failed ({response.status_code})"
                _print(f"✗ {error}", "red")
                return MediaResult(key=key, success=False, error=error)
            
            response_data = response.json()
            media_id = response_data.get("id", None)
            url = f"{self._client.api_url}/media/{media_id}" if media_id else None
            
            _print(f"✓ Image '{key}' uploaded", "green")
            return MediaResult(
                key=key,
                success=True,
                media_id=media_id,
                url=url,
                file_size=file_size,
                content_type="image/png"
            )
            
        except Exception as e:
            error = str(e)
            _print(f"✗ Failed to upload image: {error}", "red")
            return MediaResult(key=key, success=False, error=error)

    def log_file(self, key: str, path: str, 
                  content_type: Optional[str] = None,
                  meta: Optional[Dict[str, Any]] = None) -> Optional[MediaResult]:
        """Log a file."""
        if not self.is_master:
            return None
        
        valid, error = validate_key(key)
        if not valid:
            _print(f"✗ Invalid key: {error}", "red")
            return MediaResult(key=key, success=False, error=error)
        
        valid, error = validate_file_path(path)
        if not valid:
            _print(f"✗ Invalid file: {error}", "red")
            return MediaResult(key=key, success=False, error=error)
        
        if content_type is None:
            content_type = guess_content_type(path)
        
        try:
            with open(path, "rb") as f:
                file_bytes = f.read()
                file_size = len(file_bytes)
            
            blob_base64 = base64.b64encode(file_bytes).decode("utf-8")
            filename = os.path.basename(path)
            
            payload = {
                "run_id": self.run_id,
                "kind": "file",
                "key": key,
                "meta": json.dumps(meta or {}),
                "blob": blob_base64,
                "filename": filename,
                "content_type": content_type,
            }
            
            response = requests.post(
                f"{self._client.api_url}/api/sdk/log-image",
                headers=self._client.headers,
                json=payload
            )
            
            if not response.ok:
                error = f"Upload failed ({response.status_code})"
                _print(f"✗ {error}", "red")
                return MediaResult(key=key, success=False, error=error)
            
            response_data = response.json()
            media_id = response_data.get("id", None)
            url = f"{self._client.api_url}/media/{media_id}" if media_id else None
            
            _print(f"✓ File '{key}' uploaded ({file_size / 1024:.1f} KB)", "green")
            return MediaResult(
                key=key,
                success=True,
                media_id=media_id,
                url=url,
                file_size=file_size,
                content_type=content_type
            )
            
        except Exception as e:
            error = str(e)
            _print(f"✗ Failed to upload file: {error}", "red")
            return MediaResult(key=key, success=False, error=error)
    
    def log_matplotlib(self, key: str, fig,
                       meta: Optional[Dict[str, Any]] = None) -> Optional[MediaResult]:
        """Log a matplotlib figure directly."""
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib not installed. Install with: pip install matplotlib")
        
        if not hasattr(fig, 'savefig'):
            raise ValueError(f"Expected matplotlib figure, got {type(fig).__name__}")
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        pil_image = Image.open(buf)
        
        result = self.log_image(key, pil_image, meta)
        buf.close()
        
        return result
    
    def log_plot(self, key: str,
                 x_data: Union[List, "np.ndarray"],
                 y_data: Union[List, "np.ndarray"],
                 title: Optional[str] = None,
                 xlabel: Optional[str] = None,
                 ylabel: Optional[str] = None,
                 figsize: Tuple[int, int] = (8, 6),
                 style: str = '-',
                 meta: Optional[Dict[str, Any]] = None) -> Optional[MediaResult]:
        """Create and log a simple plot from data."""
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib not installed. Install with: pip install matplotlib")
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(x_data, y_data, style)
        
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        result = self.log_matplotlib(key, fig, meta)
        plt.close(fig)
        
        return result
    
    def log_array_as_image(self, key: str,
                           array: "np.ndarray",
                           cmap: str = 'viridis',
                           title: Optional[str] = None,
                           colorbar: bool = True,
                           figsize: Tuple[int, int] = (8, 6),
                           meta: Optional[Dict[str, Any]] = None) -> Optional[MediaResult]:
        """Convert numpy array to heatmap and log it."""
        if not HAS_NUMPY:
            raise ImportError("NumPy not installed. Install with: pip install numpy")
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib not installed. Install with: pip install matplotlib")
        
        if not isinstance(array, np.ndarray):
            raise ValueError(f"Expected numpy array, got {type(array).__name__}")
        if array.ndim != 2:
            raise ValueError(f"Expected 2D array, got {array.ndim}D array")
        
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(array, cmap=cmap, aspect='auto', interpolation='nearest')
        
        if title:
            ax.set_title(title)
        if colorbar:
            plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        
        result = self.log_matplotlib(key, fig, meta)
        plt.close(fig)
        
        return result
    
    def log_histogram(self, key: str,
                      data: Union[List, "np.ndarray"],
                      bins: int = 50,
                      title: Optional[str] = None,
                      xlabel: Optional[str] = None,
                      ylabel: str = "Count",
                      figsize: Tuple[int, int] = (8, 6),
                      meta: Optional[Dict[str, Any]] = None) -> Optional[MediaResult]:
        """Create and log a histogram."""
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib not installed. Install with: pip install matplotlib")
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.hist(data, bins=bins, alpha=0.7, edgecolor='black')
        
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        result = self.log_matplotlib(key, fig, meta)
        plt.close(fig)
        
        return result
    
    def log_scatter(self, key: str,
                    x_data: Union[List, "np.ndarray"],
                    y_data: Union[List, "np.ndarray"],
                    title: Optional[str] = None,
                    xlabel: Optional[str] = None,
                    ylabel: Optional[str] = None,
                    figsize: Tuple[int, int] = (8, 6),
                    s: int = 20,
                    alpha: float = 0.7,
                    meta: Optional[Dict[str, Any]] = None) -> Optional[MediaResult]:
        """Create and log a scatter plot."""
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib not installed. Install with: pip install matplotlib")
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(x_data, y_data, s=s, alpha=alpha)
        
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        result = self.log_matplotlib(key, fig, meta)
        plt.close(fig)
        
        return result
    
    def log_multiple_plots(self, key: str,
                           plots_data: List[Tuple[Union[List, "np.ndarray"], Union[List, "np.ndarray"], str]],
                           title: Optional[str] = None,
                           xlabel: Optional[str] = None,
                           ylabel: Optional[str] = None,
                           figsize: Tuple[int, int] = (10, 6),
                           meta: Optional[Dict[str, Any]] = None) -> Optional[MediaResult]:
        """Create and log multiple lines on the same plot."""
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib not installed. Install with: pip install matplotlib")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        for x_data, y_data, label in plots_data:
            ax.plot(x_data, y_data, label=label)
        
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        result = self.log_matplotlib(key, fig, meta)
        plt.close(fig)
        
        return result
    
    def log_batch(self, metrics_list: List[Dict[str, Any]],
                  max_workers: int = 4) -> Optional[BatchResult]:
        """Log multiple metrics in parallel."""
        if not self.is_master:
            return None
        
        total = len(metrics_list)
        successful = 0
        failed = 0
        results = []
        errors = []
        
        _print(f"Logging {total} metric batches...", "cyan")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for item in metrics_list:
                item_copy = item.copy()
                step = item_copy.pop("step", 0)
                future = executor.submit(self.log, step, **item_copy)
                futures[future] = step
            
            for future in as_completed(futures):
                result = future.result()
                if result and result.success:
                    successful += 1
                else:
                    failed += 1
                    if result and result.error:
                        errors.append(result.error)
                results.append(result)
        
        batch_result = BatchResult(
            total=total,
            successful=successful,
            failed=failed,
            results=results,
            errors=errors
        )
        
        if failed == 0:
            _print(f"✓ Batch complete: {successful}/{total} successful", "green bold")
        else:
            _print(f"⚠ Batch complete: {successful}/{total} successful, {failed} failed", "yellow")
        
        return batch_result
    
    def log_source_code(self, source: str, content: str, environment: str,
                        signature: Optional[str] = None) -> bool:
        """Log source code for this run."""
        if not self.is_master:
            return False
        
        if not content:
            _print("⚠ No source code content to log", "yellow")
            return False
        
        try:
            content_bytes = content.encode("utf-8")
            content_base64 = base64.b64encode(content_bytes).decode("utf-8")
            
            payload = {
                "run_id": self.run_id,
                "source": source or "unknown",
                "content": content_base64,
                "environment": environment if environment in [".ipynb", ".py", "colab", "interactive", "unknown"] else "unknown",
            }
            
            if signature:
                payload["signature"] = signature
            
            response = requests.post(
                f"{self._client.api_url}/api/sdk/log-source-code",
                headers=self._client.headers,
                json=payload
            )
            
            if not response.ok:
                _print(f"⚠ Failed to log source code ({response.status_code})", "yellow")
                return False
            
            response_data = response.json()
            if response_data.get("reused"):
                _print("✓ Source code linked (reused existing)", "green")
            else:
                _print("✓ Source code uploaded", "green")
            return True
            
        except Exception as e:
            _print(f"⚠ Failed to log source code: {e}", "yellow")
            return False
    
    def __enter__(self) -> "Run":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.is_master and exc_type is None:
            _print("✓ Run completed successfully", "green")
        elif self.is_master and exc_type is not None:
            _print(f"⚠ Run ended with error: {exc_val}", "yellow")
        return False
    
    def __repr__(self) -> str:
        return f"Run(id='{self.run_id}', name='{self.name}', project='{self._project.name}')"
