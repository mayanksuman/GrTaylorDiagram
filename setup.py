from setuptools import setup
from GrTaylorDiagram import __version__ as version
from GrTaylorDiagram import __author__ as author

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
   name='GrTaylorDiagram',
   version=version,
   description='Grouped Taylor Diagram',
   license="Apache-2.0",
   long_description=long_description,
   author = author,
   author_email='mayanksuman@live.com',
   packages=['GrTaylorDiagram'],
   install_requires=['matplotlib', 'numpy'],
   classifiers=["License :: OSI Approved :: Apache Software License",
                "Development Status :: 5 - Production/Stable",
                "Operating System :: OS Independent",
                "Programming Language :: Python",
                "Programming Language :: Python :: 3",
                "Programming Language :: Python :: 3.7",
                "Programming Language :: Python :: 3.8",
                "Programming Language :: Python :: 3.9",
                ],
)
