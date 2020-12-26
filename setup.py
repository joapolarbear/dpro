from setuptools import setup, find_packages

def fix_setuptools():
    """Work around bugs in setuptools.                                                                                                                                                        

    Some versions of setuptools are broken and raise SandboxViolation for normal                                                                                                              
    operations in a virtualenv. We therefore disable the sandbox to avoid these                                                                                                               
    issues.                                                                                                                                                                                   
    """
    try:
        from setuptools.sandbox import DirectorySandbox
        def violation(operation, *args, **_):
            print("SandboxViolation: %s" % (args,))

        DirectorySandbox._violation = violation
    except ImportError:
        pass

# Fix bugs in setuptools.                                                                                                                                                                     
fix_setuptools()

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
          'sklearn',
          'seaborn',
          'matplotlib',
          'tensorflow',
          'tqdm',
          "cvxpy",
          "cvxopt",
        
          "python-igraph",
          'pymc3',
      ],
      # zip_safe=False
      )
