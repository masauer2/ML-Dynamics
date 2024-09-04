from setuptools import setup, find_packages

setup(
    name='MLDynamics',
    version='0.1.0',
    packages=find_packages(include=['ML-Dynamics', 'ML-Dynamics.*']),
    install_requires=['numpy>=1.14.5']
)
