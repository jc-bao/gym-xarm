from setuptools import setup, find_packages
from pathlib import Path

setup(
    name='gym_xarm',
    author="Chaoyi Pan",
    author_email="pcy19@mails.tsinghua.edu.cn",
    version='0.0.1',
    description="An OpenAI Gym Env for Xarm",
    long_description=Path("README.md").read_text(),
    packages=find_packages(include="gym_panda_reach*"),
    package_data={
        # If any package contains *.txt or *.rst files, include them:
        "": ["*.urdf"],
    },

    install_requires=['gym', 'pybullet', 'numpy'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        ],
    python_requires='>=3.6'
)
