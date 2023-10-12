from setuptools import setup

setup(
    name="roma",
    packages=find_packages(include=("roma*",)),
    version="0.0.1",
    author="Johan Edstedt",
    install_requires=open("requirements.txt", "r").read().split("\n"),
)
