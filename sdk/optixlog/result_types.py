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
Type definitions for OptixLog SDK

This module contains all return types and type hints for the SDK.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class MetricResult:
    """Result from logging metrics"""
    step: int
    metrics: Dict[str, Any]
    success: bool
    timestamp: Optional[datetime] = None
    error: Optional[str] = None
    
    def __bool__(self) -> bool:
        """Allow boolean checks: if result: ..."""
        return self.success
    
    def __repr__(self) -> str:
        if self.success:
            return f"MetricResult(step={self.step}, metrics={len(self.metrics)} values, ✓)"
        return f"MetricResult(step={self.step}, ✗ {self.error})"


@dataclass
class MediaResult:
    """Result from logging images or files"""
    key: str
    success: bool
    media_id: Optional[str] = None
    url: Optional[str] = None
    file_size: Optional[int] = None
    content_type: Optional[str] = None
    error: Optional[str] = None
    
    def __bool__(self) -> bool:
        """Allow boolean checks: if result: ..."""
        return self.success
    
    def __repr__(self) -> str:
        if self.success:
            size_str = f", {self.file_size} bytes" if self.file_size else ""
            return f"MediaResult(key='{self.key}', url='{self.url}'{size_str}, ✓)"
        return f"MediaResult(key='{self.key}', ✗ {self.error})"


@dataclass
class BatchResult:
    """Result from batch operations"""
    total: int
    successful: int
    failed: int
    results: List[Any] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Return success rate as percentage"""
        return (self.successful / self.total * 100) if self.total > 0 else 0.0
    
    def __bool__(self) -> bool:
        """Batch is successful if all operations succeeded"""
        return self.failed == 0
    
    def __repr__(self) -> str:
        return f"BatchResult({self.successful}/{self.total} successful, {self.success_rate:.1f}%)"


@dataclass
class RunInfo:
    """Information about a run"""
    run_id: str
    name: Optional[str]
    project_id: str
    project_name: str
    config: Dict[str, Any]
    created_at: str
    status: str = "running"
    
    def __repr__(self) -> str:
        return f"RunInfo(id='{self.run_id}', name='{self.name}', status='{self.status}')"


@dataclass
class ArtifactInfo:
    """Information about an uploaded artifact"""
    media_id: str
    key: str
    kind: str
    url: str
    content_type: str
    file_size: Optional[int] = None
    created_at: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return f"ArtifactInfo(key='{self.key}', kind='{self.kind}', url='{self.url}')"


@dataclass
class ProjectInfo:
    """Information about a project"""
    project_id: str
    name: str
    created_at: str
    run_count: Optional[int] = None
    
    def __repr__(self) -> str:
        runs = f", {self.run_count} runs" if self.run_count is not None else ""
        return f"ProjectInfo(id='{self.project_id}', name='{self.name}'{runs})"


@dataclass
class ComparisonResult:
    """Result from comparing multiple runs"""
    runs: List[RunInfo]
    common_metrics: List[str]
    metrics_data: Dict[str, Dict[str, List[float]]]  # metric -> run_id -> values
    
    def __repr__(self) -> str:
        return f"ComparisonResult({len(self.runs)} runs, {len(self.common_metrics)} common metrics)"

