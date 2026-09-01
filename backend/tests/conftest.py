"""Force pytest to run with the project root as cwd.

Relative test paths such as ml/models/... and data/... are defined from ROOT.
This fixture does not change production runtime; wrappers also resolve via ROOT.
"""
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def pytest_configure(config):
    os.chdir(ROOT)
