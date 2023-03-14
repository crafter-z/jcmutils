from setuptools import setup,find_packages

VERSION = '1.0.2'
DESCRIPTION = "A general utils for jcmsuite"

setup(
    name="jcmutils",
    version=VERSION,
    author="crafter-z",
    author_email="crafterz@163.com",
    description=DESCRIPTION,
    packages=find_packages(),
    install_requires=["numpy","matplotlib"],
    keywords=["jcmsuite","utils"],
    classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    ]
)