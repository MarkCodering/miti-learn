from setuptools import setup, find_packages

setup(
    name='miti_learn',
    version='1.0.0',
    author='Mark Hao-Yuan Chen',
    author_email='mark.chen.sstm@gmail.com',
    description='A package for quantum error mitigation through machine learning',
    packages=find_packages(),
    install_requires=[
        'timm',
        'numpy',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)