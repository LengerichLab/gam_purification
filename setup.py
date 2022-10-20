import setuptools

setuptools.setup(name='gam_purification',
      packages=['gam_purification'],
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
