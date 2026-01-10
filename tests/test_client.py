"""Unit tests for OptixLog SDK client."""

import os
from unittest import mock
import pytest

from optixlog.client import Optixlog, Project, Run, OxInvalidTaskError


class TestOptixlogClient:
    """Test Optixlog client initialization and methods."""

    def test_client_initialization(self):
        """Test client initialization with API key."""
        client = Optixlog(api_key="test_key")
        assert client.api_key == "test_key"
        assert client.api_url == "https://optixlog.com"

    def test_client_custom_url(self):
        """Test client initialization with custom URL."""
        client = Optixlog(api_key="test_key", api_url="https://custom.com")
        assert client.api_url == "https://custom.com"

    def test_client_missing_api_key(self):
        """Test that missing API key raises error."""
        with pytest.raises(ValueError, match="Missing API key"):
            Optixlog(api_key="")

    def test_client_strips_api_key(self):
        """Test that API key is stripped."""
        client = Optixlog(api_key="  test_key  ")
        assert client.api_key == "test_key"

    @mock.patch("optixlog.client.requests.get")
    def test_fetch_projects_success(self, mock_get):
        """Test fetching projects successfully."""
        mock_response = mock.MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "projects": [
                {"id": "proj1", "name": "Project 1"},
                {"id": "proj2", "name": "Project 2"}
            ]
        }
        mock_get.return_value = mock_response

        client = Optixlog(api_key="test_key")
        projects = client._fetch_projects()

        assert len(projects) == 2
        assert projects[0]["name"] == "Project 1"

    @mock.patch("optixlog.client.requests.get")
    def test_fetch_projects_unauthorized(self, mock_get):
        """Test that unauthorized returns error."""
        mock_response = mock.MagicMock()
        mock_response.status_code = 401
        mock_get.return_value = mock_response

        client = Optixlog(api_key="bad_key")

        with pytest.raises(ValueError, match="Invalid API Key"):
            client._fetch_projects()

    @mock.patch("optixlog.client.requests.get")
    def test_fetch_projects_connection_error(self, mock_get):
        """Test connection error handling."""
        mock_get.side_effect = Exception("Connection failed")

        client = Optixlog(api_key="test_key")

        with pytest.raises(ValueError, match="Cannot connect"):
            client._fetch_projects()

    @mock.patch("optixlog.client.Optixlog._fetch_projects")
    def test_get_project_by_name(self, mock_fetch):
        """Test getting project by name."""
        mock_fetch.return_value = [
            {"id": "proj1", "name": "My Project"},
            {"id": "proj2", "name": "Other Project"}
        ]

        client = Optixlog(api_key="test_key")
        project = client.project(name="My Project")

        assert project.id == "proj1"
        assert project.name == "My Project"

    @mock.patch("optixlog.client.Optixlog._fetch_projects")
    def test_get_project_by_id(self, mock_fetch):
        """Test getting project by ID."""
        mock_fetch.return_value = [
            {"id": "proj1", "name": "My Project"},
        ]

        client = Optixlog(api_key="test_key")
        project = client.project(name="proj1")

        assert project.id == "proj1"

    @mock.patch("optixlog.client.Optixlog._fetch_projects")
    def test_get_nonexistent_project(self, mock_fetch):
        """Test that nonexistent project raises error."""
        mock_fetch.return_value = [
            {"id": "proj1", "name": "My Project"},
        ]

        client = Optixlog(api_key="test_key")

        with pytest.raises(ValueError, match="not found"):
            client.project(name="Nonexistent")

    @mock.patch("optixlog.client.Optixlog._fetch_projects")
    @mock.patch("optixlog.client.requests.post")
    def test_create_project(self, mock_post, mock_fetch):
        """Test creating a new project."""
        mock_fetch.return_value = []

        mock_response = mock.MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {"id": "new_proj", "created": True}
        mock_post.return_value = mock_response

        client = Optixlog(api_key="test_key")
        project = client.project(name="New Project", create_if_not_exists=True)

        assert project.id == "new_proj"
        mock_post.assert_called_once()


class TestProject:
    """Test Project class."""

    def test_project_initialization(self):
        """Test project initialization."""
        client = Optixlog(api_key="test_key")
        project = Project(client, "proj1", "Test Project")

        assert project.id == "proj1"
        assert project.name == "Test Project"

    @mock.patch("optixlog.client.requests.post")
    def test_create_run(self, mock_post):
        """Test creating a run."""
        mock_response = mock.MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {"id": "run123"}
        mock_post.return_value = mock_response

        client = Optixlog(api_key="test_key")
        project = Project(client, "proj1", "Test Project")
        run = project.run(name="Test Run", config={"lr": 0.001})

        assert run.run_id == "run123"
        assert run.name == "Test Run"


class TestRun:
    """Test Run class."""

    @mock.patch("optixlog.client.requests.post")
    def test_run_initialization(self, mock_post):
        """Test run initialization."""
        mock_response = mock.MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {"id": "run123"}
        mock_post.return_value = mock_response

        client = Optixlog(api_key="test_key")
        project = Project(client, "proj1", "Test Project")
        run = Run(client, project, name="Test Run", config={"lr": 0.001})

        assert run.run_id == "run123"
        assert run.name == "Test Run"
        assert run._config == {"lr": 0.001}

    @mock.patch("optixlog.client.requests.post")
    def test_log_metrics(self, mock_post):
        """Test logging metrics."""
        # Mock run creation
        mock_create_response = mock.MagicMock()
        mock_create_response.ok = True
        mock_create_response.json.return_value = {"id": "run123"}

        # Mock log metric
        mock_log_response = mock.MagicMock()
        mock_log_response.ok = True

        mock_post.side_effect = [mock_create_response, mock_log_response]

        client = Optixlog(api_key="test_key")
        project = Project(client, "proj1", "Test Project")
        run = Run(client, project)

        result = run.log(step=0, loss=0.5, accuracy=0.9)

        assert result is not None
        assert result.success is True
        assert result.step == 0
        assert result.metrics["loss"] == 0.5

    @mock.patch("optixlog.client.requests.post")
    def test_log_metrics_with_nan(self, mock_post):
        """Test that logging NaN fails."""
        mock_response = mock.MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {"id": "run123"}
        mock_post.return_value = mock_response

        client = Optixlog(api_key="test_key")
        project = Project(client, "proj1", "Test Project")
        run = Run(client, project)

        result = run.log(step=0, loss=float('nan'))

        assert result.success is False
        assert "nan" in result.error.lower()

    @mock.patch("optixlog.client.requests.post")
    def test_set_config(self, mock_post):
        """Test setting config."""
        mock_create_response = mock.MagicMock()
        mock_create_response.ok = True
        mock_create_response.json.return_value = {"id": "run123"}

        mock_config_response = mock.MagicMock()
        mock_config_response.ok = True

        mock_post.side_effect = [mock_create_response, mock_config_response]

        client = Optixlog(api_key="test_key")
        project = Project(client, "proj1", "Test Project")
        run = Run(client, project)

        result = run.set_config({"lr": 0.001, "epochs": 100})

        assert result == run  # Should be chainable
        assert run._config["lr"] == 0.001
        assert run._config["epochs"] == 100

    @mock.patch("optixlog.client.requests.post")
    def test_log_image(self, mock_post):
        """Test logging image."""
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("PIL not available")

        mock_create_response = mock.MagicMock()
        mock_create_response.ok = True
        mock_create_response.json.return_value = {"id": "run123"}

        mock_log_response = mock.MagicMock()
        mock_log_response.ok = True
        mock_log_response.json.return_value = {"id": "media123"}

        mock_post.side_effect = [mock_create_response, mock_log_response]

        client = Optixlog(api_key="test_key")
        project = Project(client, "proj1", "Test Project")
        run = Run(client, project)

        img = Image.new('RGB', (100, 100))
        result = run.log_image("test_image", img)

        assert result is not None
        assert result.success is True
        assert result.key == "test_image"
        assert result.media_id == "media123"

    @mock.patch("optixlog.client.requests.post")
    def test_log_matplotlib(self, mock_post):
        """Test logging matplotlib figure."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("Matplotlib not available")

        mock_create_response = mock.MagicMock()
        mock_create_response.ok = True
        mock_create_response.json.return_value = {"id": "run123"}

        mock_log_response = mock.MagicMock()
        mock_log_response.ok = True
        mock_log_response.json.return_value = {"id": "media123"}

        mock_post.side_effect = [mock_create_response, mock_log_response]

        client = Optixlog(api_key="test_key")
        project = Project(client, "proj1", "Test Project")
        run = Run(client, project)

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])

        result = run.log_matplotlib("test_plot", fig)

        assert result is not None
        assert result.success is True
        plt.close(fig)

    @mock.patch("optixlog.client.requests.post")
    def test_context_manager(self, mock_post):
        """Test run as context manager."""
        mock_response = mock.MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {"id": "run123"}
        mock_post.return_value = mock_response

        client = Optixlog(api_key="test_key")
        project = Project(client, "proj1", "Test Project")

        with project.run() as run:
            assert run.run_id == "run123"

        # Should exit cleanly

    @mock.patch("optixlog.client.requests.post")
    def test_mpi_detection(self, mock_post):
        """Test MPI environment detection."""
        mock_response = mock.MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {"id": "run123"}
        mock_post.return_value = mock_response

        # Test with MPI environment variable
        with mock.patch.dict(os.environ, {"OMPI_COMM_WORLD_RANK": "0"}):
            client = Optixlog(api_key="test_key")
            assert client.is_master is True
            assert client.rank == 0

        with mock.patch.dict(os.environ, {"OMPI_COMM_WORLD_RANK": "1"}):
            client = Optixlog(api_key="test_key")
            assert client.is_master is False
            assert client.rank == 1


class TestMPISupport:
    """Test MPI support."""

    @mock.patch("optixlog.client.requests.post")
    def test_non_master_no_logging(self, mock_post):
        """Test that non-master processes don't log."""
        mock_response = mock.MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {"id": "run123"}
        mock_post.return_value = mock_response

        # Simulate rank 1 (non-master)
        with mock.patch.dict(os.environ, {"OMPI_COMM_WORLD_RANK": "1"}):
            client = Optixlog(api_key="test_key")
            project = Project(client, "proj1", "Test Project")
            run = Run(client, project)

            # Should return None for non-master
            result = run.log(step=0, loss=0.5)
            assert result is None


class TestErrorHandling:
    """Test error handling."""

    @mock.patch("optixlog.client.requests.post")
    def test_network_error_during_log(self, mock_post):
        """Test that network errors during logging are handled."""
        mock_create_response = mock.MagicMock()
        mock_create_response.ok = True
        mock_create_response.json.return_value = {"id": "run123"}

        mock_post.side_effect = [
            mock_create_response,
            Exception("Network error")
        ]

        client = Optixlog(api_key="test_key")
        project = Project(client, "proj1", "Test Project")
        run = Run(client, project)

        result = run.log(step=0, loss=0.5)

        # Should return error result, not crash
        assert result.success is False
        assert result.error is not None

    @mock.patch("optixlog.client.requests.post")
    def test_server_error(self, mock_post):
        """Test that server errors are handled."""
        mock_create_response = mock.MagicMock()
        mock_create_response.ok = True
        mock_create_response.json.return_value = {"id": "run123"}

        mock_error_response = mock.MagicMock()
        mock_error_response.ok = False
        mock_error_response.status_code = 500

        mock_post.side_effect = [mock_create_response, mock_error_response]

        client = Optixlog(api_key="test_key")
        project = Project(client, "proj1", "Test Project")
        run = Run(client, project)

        result = run.log(step=0, loss=0.5)

        assert result.success is False
        assert "500" in result.error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
