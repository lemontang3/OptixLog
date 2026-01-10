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
Query capabilities for OptixLog SDK

This module provides functions to list runs, get run details, and download artifacts.
"""

import requests
import os
from typing import Optional, List, Dict, Any
from pathlib import Path

try:
    from rich.console import Console
    from rich.table import Table
    console = Console()
except ImportError:
    console = None

from .result_types import RunInfo, ArtifactInfo, ProjectInfo, ComparisonResult


def _print(message: str, style: str = ""):
    """Print with rich colors if available"""
    if console:
        console.print(message, style=style)
    else:
        print(message)


def list_runs(api_url: str, api_key: str, 
              project: Optional[str] = None, 
              limit: int = 10) -> List[RunInfo]:
    """
    List recent runs, optionally filtered by project
    
    Args:
        api_url: API endpoint URL
        api_key: API key
        project: Optional project name to filter by
        limit: Maximum number of runs to return
    
    Returns:
        List of RunInfo objects
        
    Example:
        runs = optixlog.list_runs(api_url, api_key, project="MyProject", limit=5)
        for run in runs:
            print(f"{run.name}: {run.run_id}")
    
    Note:
        This function requires the /api/sdk/list-runs backend endpoint.
        Check if your OptixLog backend supports this endpoint.
    """
    api_url = api_url.rstrip("/")
    headers = {"Authorization": f"Bearer {api_key}"}
    
    try:
        # Get project ID if project name provided
        project_id = None
        if project:
            projects_response = requests.get(f"{api_url}/api/sdk/initialize-run-check", headers=headers)
            if not projects_response.ok:
                _print(f"✗ Failed to fetch projects", "red")
                return []
            
            data = projects_response.json()
            projects = data.get("projects", data) if isinstance(data, dict) else data
            for p in projects:
                if p["name"] == project:
                    project_id = p["id"]
                    break
            
            if not project_id:
                _print(f"✗ Project '{project}' not found", "red")
                return []
        
        # Get runs using SDK endpoint
        params = {"limit": limit, "offset": 0}
        if project_id:
            params["project_id"] = project_id
        
        response = requests.get(f"{api_url}/api/sdk/list-runs", headers=headers, params=params)
        
        if response.status_code == 404:
            _print(f"⚠ list_runs endpoint not available on this server", "yellow")
            return []
        elif not response.ok:
            _print(f"✗ Failed to fetch runs ({response.status_code})", "red")
            return []
        
        runs_data = response.json()
        if not isinstance(runs_data, list):
            runs_data = runs_data.get("runs", []) if isinstance(runs_data, dict) else []
        
        runs = []
        for run_data in runs_data:
            run_info = RunInfo(
                run_id=run_data.get("id", run_data.get("run_id", "")),
                name=run_data.get("name"),
                project_id=run_data.get("project_id", project_id or ""),
                project_name=project or run_data.get("project_name", "Unknown"),
                config=run_data.get("config", {}) or {},
                created_at=run_data.get("created_at", ""),
                status=run_data.get("status", "completed")
            )
            runs.append(run_info)
        
        _print(f"✓ Found {len(runs)} runs", "green")
        return runs
        
    except requests.exceptions.ConnectionError:
        _print("✗ Cannot connect to server", "red")
        return []
    except Exception as e:
        _print(f"✗ Error: {e}", "red")
        return []


def get_run(api_url: str, api_key: str, run_id: str) -> Optional[RunInfo]:
    """
    Get details about a specific run
    
    Args:
        api_url: API endpoint URL
        api_key: API key
        run_id: Run ID
    
    Returns:
        RunInfo object or None if not found
        
    Example:
        run = optixlog.get_run(api_url, api_key, "run_abc123")
        print(f"Config: {run.config}")
    """
    api_url = api_url.rstrip("/")
    headers = {"Authorization": f"Bearer {api_key}"}
    
    try:
        response = requests.get(f"{api_url}/api/sdk/get-run", headers=headers, params={"run_id": run_id})
        
        if response.status_code == 404:
            _print(f"✗ Run '{run_id}' not found", "red")
            return None
        elif not response.ok:
            _print(f"✗ Failed to fetch run ({response.status_code})", "red")
            return None
        
        run_data = response.json()
        if not run_data:
            return None
        
        run_info = RunInfo(
            run_id=run_data.get("id", ""),
            name=run_data.get("name"),
            project_id=run_data.get("project_id", ""),
            project_name=run_data.get("project_name", "Unknown"),
            config=run_data.get("config", {}) or {},
            created_at=run_data.get("created_at", ""),
            status=run_data.get("status", "completed")
        )
        
        _print(f"✓ Retrieved run '{run_id}'", "green")
        return run_info
        
    except Exception as e:
        _print(f"✗ Error: {e}", "red")
        return None


def get_artifacts(api_url: str, api_key: str, run_id: str) -> List[ArtifactInfo]:
    """
    Get list of artifacts (images, files) for a run
    
    Args:
        api_url: API endpoint URL
        api_key: API key
        run_id: Run ID
    
    Returns:
        List of ArtifactInfo objects
        
    Example:
        artifacts = optixlog.get_artifacts(api_url, api_key, "run_abc123")
        for artifact in artifacts:
            print(f"{artifact.key}: {artifact.url}")
    """
    api_url = api_url.rstrip("/")
    headers = {"Authorization": f"Bearer {api_key}"}
    
    try:
        response = requests.get(f"{api_url}/api/sdk/get-artifacts", headers=headers, params={"run_id": run_id})
        
        if response.status_code == 404:
            _print(f"✗ Run '{run_id}' not found", "red")
            return []
        elif not response.ok:
            _print(f"✗ Failed to fetch artifacts ({response.status_code})", "red")
            return []
        
        media_data = response.json()
        if not isinstance(media_data, list):
            media_data = []
        
        artifacts = []
        for item in media_data:
            artifact = ArtifactInfo(
                media_id=item.get("id", ""),
                key=item.get("key", ""),
                kind=item.get("kind", "file"),
                url=item.get("url", f"{api_url}/media/{item.get('id', '')}"),
                content_type=item.get("content_type", "application/octet-stream"),
                file_size=item.get("file_size"),
                created_at=item.get("created_at"),
                meta=item.get("meta", {}) or {}
            )
            artifacts.append(artifact)
        
        _print(f"✓ Found {len(artifacts)} artifacts", "green")
        return artifacts
        
    except Exception as e:
        _print(f"✗ Error: {e}", "red")
        return []


def download_artifact(api_url: str, api_key: str, 
                       media_id: str, 
                       output_path: str) -> bool:
    """
    Download an artifact to a local file
    
    Args:
        api_url: API endpoint URL
        api_key: API key
        media_id: Media/artifact ID
        output_path: Path to save the downloaded file
    
    Returns:
        True if successful, False otherwise
        
    Example:
        success = optixlog.download_artifact(
            api_url, api_key, "media_abc123", "output/plot.png"
        )
    """
    api_url = api_url.rstrip("/")
    headers = {"Authorization": f"Bearer {api_key}"}
    
    try:
        # For now, use the media URL directly (download endpoint may need to be implemented)
        response = requests.get(f"{api_url}/media/{media_id}", headers=headers, stream=True)
        
        if response.status_code == 404:
            _print(f"✗ Artifact '{media_id}' not found", "red")
            return False
        elif not response.ok:
            _print(f"✗ Download failed ({response.status_code})", "red")
            return False
        
        # Create output directory if needed
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Download file
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        file_size = os.path.getsize(output_path)
        _print(f"✓ Downloaded to '{output_path}' ({file_size / 1024:.1f} KB)", "green")
        return True
        
    except Exception as e:
        _print(f"✗ Error: {e}", "red")
        return False


def get_metrics(api_url: str, api_key: str, run_id: str) -> Dict[str, List[Any]]:
    """
    Get all metrics logged for a run
    
    Args:
        api_url: API endpoint URL
        api_key: API key
        run_id: Run ID
    
    Returns:
        Dictionary of metric_name -> list of (step, value) tuples
        
    Example:
        metrics = optixlog.get_metrics(api_url, api_key, "run_abc123")
        for name, values in metrics.items():
            print(f"{name}: {len(values)} data points")
    """
    api_url = api_url.rstrip("/")
    headers = {"Authorization": f"Bearer {api_key}"}
    
    try:
        response = requests.get(f"{api_url}/api/sdk/get-metrics", headers=headers, params={"run_id": run_id})
        
        if response.status_code == 404:
            _print(f"✗ Run '{run_id}' not found", "red")
            return {}
        elif not response.ok:
            _print(f"✗ Failed to fetch metrics ({response.status_code})", "red")
            return {}
        
        metrics_data = response.json()
        if not isinstance(metrics_data, list):
            metrics_data = []
        
        # Organize by metric name
        metrics_by_name: Dict[str, List[Any]] = {}
        for entry in metrics_data:
            step = entry.get("step", 0) or 0
            kv = entry.get("kv", {}) or {}
            
            for key, value in kv.items():
                if key not in metrics_by_name:
                    metrics_by_name[key] = []
                metrics_by_name[key].append((step, value))
        
        # Sort by step
        for key in metrics_by_name:
            metrics_by_name[key].sort(key=lambda x: x[0])
        
        _print(f"✓ Retrieved {len(metrics_by_name)} metrics", "green")
        return metrics_by_name
        
    except Exception as e:
        _print(f"✗ Error: {e}", "red")
        return {}


def compare_runs(api_url: str, api_key: str, run_ids: List[str]) -> Optional[ComparisonResult]:
    """
    Compare metrics across multiple runs
    
    Args:
        api_url: API endpoint URL
        api_key: API key
        run_ids: List of run IDs to compare
    
    Returns:
        ComparisonResult with common metrics and data
        
    Example:
        comparison = optixlog.compare_runs(api_url, api_key, ["run1", "run2"])
        print(f"Common metrics: {comparison.common_metrics}")
    """
    if len(run_ids) < 2:
        _print("✗ Need at least 2 runs to compare", "red")
        return None
    
    api_url = api_url.rstrip("/")
    headers = {"Authorization": f"Bearer {api_key}"}
    
    try:
        # Get info and metrics for each run
        runs = []
        all_metrics = {}
        
        # Use the compare-runs endpoint if available
        try:
            response = requests.get(
                f"{api_url}/api/sdk/compare-runs",
                headers=headers,
                params={"run_ids": run_ids}
            )
            
            if response.ok:
                comparison_data = response.json()
                if comparison_data:
                    # Convert run dictionaries to RunInfo objects
                    run_infos = []
                    for run_data in comparison_data.get('runs', []):
                        run_info = RunInfo(
                            run_id=run_data.get('id', ''),
                            name=run_data.get('name'),
                            project_id=run_data.get('project_id', ''),
                            project_name=run_data.get('project_name', 'Unknown'),
                            config=run_data.get('config', {}),
                            created_at=run_data.get('created_at', ''),
                            status=run_data.get('status', 'completed')
                        )
                        run_infos.append(run_info)
                    
                    _print(f"✓ Compared {len(run_infos)} runs with {len(comparison_data.get('common_metrics', []))} common metrics", "green")
                    return ComparisonResult(
                        runs=run_infos,
                        common_metrics=comparison_data.get('common_metrics', []),
                        metrics_data=comparison_data.get('metrics_data', {})
                    )
        except Exception as e:
            # Fallback to individual calls
            pass
        
        # Fallback: get each run individually
        for run_id in run_ids:
            run_info = get_run(api_url, api_key, run_id)
            if not run_info:
                _print(f"⚠ Skipping run '{run_id}'", "yellow")
                continue
            
            runs.append(run_info)
            metrics = get_metrics(api_url, api_key, run_id)
            all_metrics[run_id] = metrics
        
        if len(runs) < 2:
            _print("✗ Not enough valid runs to compare", "red")
            return None
        
        # Find common metrics
        metric_sets = [set(metrics.keys()) for metrics in all_metrics.values()]
        common_metrics = list(set.intersection(*metric_sets))
        
        # Organize data
        metrics_data: Dict[str, Dict[str, List[float]]] = {}
        for metric_name in common_metrics:
            metrics_data[metric_name] = {}
            for run_id in all_metrics:
                if metric_name in all_metrics[run_id]:
                    # Extract just the values (ignore steps for now)
                    values = [v for _, v in all_metrics[run_id][metric_name]]
                    metrics_data[metric_name][run_id] = values
        
        result = ComparisonResult(
            runs=runs,
            common_metrics=common_metrics,
            metrics_data=metrics_data
        )
        
        _print(f"✓ Compared {len(runs)} runs with {len(common_metrics)} common metrics", "green")
        return result
        
    except Exception as e:
        _print(f"✗ Error: {e}", "red")
        return None


def list_projects(api_url: str, api_key: str) -> List[ProjectInfo]:
    """
    List all projects
    
    Args:
        api_url: API endpoint URL
        api_key: API key
    
    Returns:
        List of ProjectInfo objects
        
    Example:
        projects = optixlog.list_projects(api_url, api_key)
        for project in projects:
            print(f"{project.name}")
    """
    api_url = api_url.rstrip("/")
    headers = {"Authorization": f"Bearer {api_key}"}
    
    try:
        # Use the SDK endpoint that returns projects
        response = requests.get(f"{api_url}/api/sdk/initialize-run-check", headers=headers)
        
        if not response.ok:
            _print(f"✗ Failed to fetch projects ({response.status_code})", "red")
            return []
        
        data = response.json()
        # Response is {"projects": [...]}
        projects_data = data.get("projects", data) if isinstance(data, dict) else data
        projects = []
        
        for proj in projects_data:
            project_info = ProjectInfo(
                project_id=proj["id"],
                name=proj["name"],
                created_at=proj.get("created_at", ""),
                run_count=proj.get("run_count")
            )
            projects.append(project_info)
        
        _print(f"✓ Found {len(projects)} projects", "green")
        return projects
        
    except Exception as e:
        _print(f"✗ Error: {e}", "red")
        return []

