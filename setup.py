#!/usr/bin/python

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('requirements.txt') as f:
    requirements = list(f.readlines())

setup(
    author="Gideon Dresdner",
    author_email='gideon@inf.ethz.ch',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English'
    ],
    description="Code base for the paper \"Boosting Black Box Variational Inference,\" cf. arXiv:1806.02185",
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    include_package_data=True,
    name='boosting_bbvi',
    packages=find_packages(include=['boosting_bbvi']),
    url='https://github.com/ratschlab/boosting_bbvi',
    version='0.0.0'
)
