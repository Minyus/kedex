from setuptools import setup
from codecs import open
from os import path
import re

package_name = "kedex"

setup(
    name=package_name,
    packages=[package_name],
    version="0.0.0",
    license="BSD 2-Clause",
    author="Yusuke Minami",
    author_email="me@minyus.github.com",
    url="https://github.com/Minyus/kedex",
    description="Python utility.",
    install_requires=[],
    keywords="minyus",
    zip_safe=False,
    test_suite="tests",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Natural Language :: English",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Operating System :: OS Independent",
    ],
)
