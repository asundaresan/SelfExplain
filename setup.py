#!/usr/bin/env python

from setuptools import setup, find_packages


with open('requirements.txt') as f:
    requirements = f.read()

setup(name='self_explain',
      version='0.0.2',
      description='Self Explain module',
      url='http://github.com/asundaresan/SelfExplain.git',
      author='Dheeraj Rajagopal, Vidisha Balachandran, Artidoro Pagnoni, Aravind Sundaresan',
      author_email='dheeraj@cs.cmu.edu',
      maintainer='Aravind Sundaresan',
      maintainer_email='aravind.sundaresan@sri.com',
      packages=find_packages(),
      scripts=[
          ],
      install_requires=requirements, 
      zip_safe=False
      )

