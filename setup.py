from setuptools import setup, find_packages

setup(name='byteprofile analysis',
      version='0.1',
      description='A profiler, replayer and optimizer for distributed ML',
      url='https://github.com/joapolarbear/byteprofile-analysis.git',
      # author='Flying Circus',
      # author_email='flyingcircus@example.com',
      # license='MIT',
      packages=find_packages(),
      install_requires=[
          'intervaltree', 
          'networkx', 
          'ujson', 
          'xlsxwriter', 
          'scapy'
      ],
      # zip_safe=False
      )