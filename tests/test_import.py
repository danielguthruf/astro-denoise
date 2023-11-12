"""Test astro-denoise."""

import astro_denoise


def test_import() -> None:
    """Test that the package can be imported."""
    assert isinstance(astro_denoise.__name__, str)
