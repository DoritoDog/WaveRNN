from pathlib import Path
from setuptools import setup

project_root = Path(__file__).parent

install_requires = (project_root / 'requirements.txt').read_text().splitlines()

setup(
    name="wavernn",
    version="0.0.1",
    packages=["wavernn"],
    python_requires=">=3.6",
    install_requires=install_requires,
)