import os
import sys
from setuptools import setup, find_packages

if sys.version_info.major != 3:
    print("This Python is only compatible with Python 3, but you are running "
          "Python {}. The installation will likely fail.".format(sys.version_info.major))

setup(
    name='hand_imitation',
    version='0.1.0',
    packages=find_packages(),
    description='environments simulated in MuJoCo',
    long_description="TBD",
    url='https://github.com/yzqin/hand_imitation.git',
    author="Xiaolong Wang's Lab",
    install_requires=[
        'transforms3d', 'gym>=0.13', 'mujoco-py<2.1,>=2.0', 'numpy', 'imageio', 'pygifsicle', 'Pillow', 'open3d',
        'scipy', 'trimesh'
    ],
)
