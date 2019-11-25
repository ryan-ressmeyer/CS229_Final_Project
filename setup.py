from setuptools import setup
from setuptools import find_packages
setup(
    name='CS229Project',
    version = '0.1',
    install_requires = [
        'h5py',
        'keras',
        'nltk',
        'pandas',
        'tensorflow',
        'tensorflow-gpu>=1.1.0',
    ],
    packages=find_packages()
)