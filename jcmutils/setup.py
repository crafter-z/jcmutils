from setuptools import setup,find_packages

VERSION = '1.0.1'
DESCRIPTION = "A general utils for jcmsuite"

setup(
    name="jcmutils",
    version=VERSION,
    author="crafter-z",
    author_email="crafterz@163.com",
    description=DESCRIPTION,
    packages=find_packages(),
    install_requires=["numpy","matplotlib"],
    keywords=["jcmsuite","utils"]

)
