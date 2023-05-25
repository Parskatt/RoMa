from setuptools import setup

setup(
    name="roma",
    packages=["roma"],
    version="0.0.1",
    author="Johan Edstedt",
    install_requires=open("requirements.txt", "r").read().split("\n"),
)
