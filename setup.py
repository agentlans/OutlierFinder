# This file is if we want to be able to pip install it & upload the package to pypi

from setuptools import setup

setup(name='OutlierFinder',
      version='0.0.1',
      description='Finds 1D outliers that deviate from a normal distribution',
      author='Alan Tseng',
      url='https://github.com/agentlans/OutlierFinder',
      packages=['OutlierFinder'],
      license='GPL3',
      )
