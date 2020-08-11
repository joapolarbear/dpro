from setuptools import setup, find_packages

setup(name='byteprofile analysis',
      version='0.1',
      description='A profiler, replayer and optimizer for distributed ML',
      url='https://github.com/joapolarbear/byteprofile-analysis.git',
      # author='xxxx',
      # author_email='xxx',
      # license='xxx',
      packages=find_packages(),
      install_requires=[
          'intervaltree', 
          'networkx', 
          'ujson', 
          'xlsxwriter', 
          'scapy',
          'xgboost',
          'sklearn'
      ],
      # zip_safe=False
      )