from setuptools import find_packages, setup

package_name = "kedex"

setup(
    name=package_name,
    packages=find_packages(where="src", exclude=["tests"]),
    package_dir={"": "src"},
    version="0.0.5",
    license="Apache Software License (Apache 2.0)",
    author="Yusuke Minami",
    author_email="me@minyus.github.com",
    url="https://github.com/Minyus/kedex",
    description="Kedro extension for quick, reusable, and readable coding",
    install_requires=["kedro>=0.15.2", "flatten-dict>=0.1.0"],
    keywords="pipelines, machine learning, data pipelines, data science, data engineering",
    zip_safe=False,
    test_suite="tests",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Operating System :: OS Independent",
    ],
)
