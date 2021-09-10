#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='self_explain',
      version='0.0.1dev',
      description='Self Explain module',
      url='http://github.com/asundaresan/SelfExplain.git',
      author='Dheeraj Rajagopal, Vidisha Balachandran, Artidoro Pagnoni, Aravind Sundaresan',
      author_email='dheeraj@cs.cmu.edu',
      maintainer='Aravind Sundaresan',
      maintainer_email='aravind.sundaresan@sri.com',
      packages=find_packages(),
      scripts=[
          ],
      install_requires=[
          "numpy>=1.7",
          "pandas",
          "nltk",
          "benepar",
          "overrides",
          "scipy",
          "pytorch_lightning",
          "torch",
          "requests",
          "tqdm",
          "transformers",
          "wordcloud",
          ],
      zip_safe=False
      )

