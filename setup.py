from setuptools import find_packages
from setuptools import setup

packages=[
"numpy",
"PyYAML",
"torch",
"matplotlib",
]

setup(
      name="vaepi_sampler",
      version="0.0",
      dscription="project description",
      packages=find_packages(),
      install_requires=packages      
)
