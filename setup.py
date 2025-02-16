from setuptools import setup, find_packages

setup(
    name="romatch",
    packages=find_packages(include=("romatch*",)),
    version="0.0.2",
    author="Johan Edstedt",
    install_requires=open("requirements.txt", "r").read().split("\n"),
)
