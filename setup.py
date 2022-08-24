import setuptools

setuptools.setup(name='gam_purification',
      packages=['gam_purification'],
      version='0.0.0',
      install_requires=[
          'numpy',
          'scikit-learn',
          'python-igraph',
          'matplotlib',
          'pandas',
          'tensorflow>=2.4.0',
          'tensorflow-addons',
          'numpy>=1.19.2',
          'ipywidgets',
          'interpret'
      ],
)
