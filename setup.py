import os
import io

from setuptools import setup, find_packages


def read(fname):
    with io.open(os.path.join(os.path.dirname(__file__), fname), encoding="utf-8") as f:
        return f.read()

setup(
    name="ml",
    version='1.0',
    packages=find_packages("."),
    package_dir={"": "."},
    python_requires=">=3.6",
    install_requires=[
        "torch>=1.9",
        "mlflow",
    ],
    include_package_data=True,
    license="MIT",
)
