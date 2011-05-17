from setuptools import setup
from numpy.distutils.misc_util import Configuration
import os
config = Configuration('pdefields',parent_package=None,top_path=None)

config.add_extension(name='manifolds.stripackd',sources=['pdefields/manifolds/stripackd.f90'])

config.packages = ["pdefields"]
if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**(config.todict()))