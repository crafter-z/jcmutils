from setuptools import setup,find_packages

VERSION = '2.0.1'
DESCRIPTION = "A general utils for jcmsuite"

setup(
    name="jcmutils",
    version=VERSION,
    author="crafter-z",
    author_email="crafterz@163.com",
    description=DESCRIPTION,
    packages=find_packages(),
    install_requires=["numpy","matplotlib","opencv-python","pyyaml"],
    keywords=["jcmsuite","utils"],
    classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    ]
)
