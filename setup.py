"""Compatibility shim for older tooling.

Modern installs use pyproject.toml.  This file intentionally contains only a
single setup() call.
"""

from setuptools import setup

setup()
