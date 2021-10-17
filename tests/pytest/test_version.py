"""Test package version."""

from edbscan import __version__


def test_version() -> None:
    """Test that the version string can be loaded."""
    assert isinstance(__version__, str)
