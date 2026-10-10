"""Test minimal : le package est installé et importable."""

import pinn_chiarella


def test_import():
    assert pinn_chiarella.__version__
