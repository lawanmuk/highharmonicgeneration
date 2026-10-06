from importlib.metadata import version

import hhg


def test_version_matches_package_metadata():
    # pyproject.toml and hhg.__version__ must be bumped together for a release.
    assert hhg.__version__ == version("hhg")
