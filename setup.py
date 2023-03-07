# -*- coding: utf-8 -*-

from pathlib import Path
from setuptools import Distribution, find_packages, setup


PACKAGE = 'vitpose'


def _get_version():
    """"Helper to get the package version."""

    version_path = Path() / PACKAGE / 'version.py'
    if not version_path.exists:
        return None
    with open(version_path, 'r') as version_file:
        ns = {}
        exec(version_file.read(), ns)
    return ns['__version__']

exclude_packages = [
    # 'smpl*',
    'tests'
]

setup(
    name="vitpose",
    version=_get_version(),
    packages=find_packages(exclude=exclude_packages),
    package_data={
        # PACKAGE: ['support_data/*', 'support_data/conf/*', 'support_data/conf/parallel_conf/*']
    },
    author="",
    author_email="",
    maintainer="",
    maintainer_email="",
    url="",
    description="Pytorch ViTPose",
    license="See LICENSE",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    # install_requires=["numpy"],

    # include_dirs=["."],
    dependency_links=[],
    classifiers=[
        "Intended Audience :: Research",
        "Natural Language :: English",
        "Operating System :: POSIX",
        "Operating System :: POSIX :: BSD",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7", ],
)
