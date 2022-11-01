"""
Installing gam_purification.
"""
from setuptools import setup, find_packages

setup(
    name='gam_purification',
    author="Ben Lengerich",
    description="Purify additive models.",
    url="https://github.com/blengerich/gam_purification",
    packages=find_packages(),
    version='0.0.1',
    install_requires=[
      'numpy',
      'scikit-learn',
      'matplotlib',
      'pandas',
      'numpy>=1.19.2',
      'ipywidgets',
      'interpret>=0.2.0'
    ],
)
