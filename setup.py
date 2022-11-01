from setuptools import setup, find_packages

setup(name='gam_purification',
      packages=find_packages(),
      version='0.0.0',
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
