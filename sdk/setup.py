from setuptools import setup, find_packages
from pathlib import Path

# Read the README for long description
readme_path = Path(__file__).parent.parent / "README.md"
long_description = ""
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name='optixlog',
    version='0.2.0',
    author='FluxBoard Team',
    author_email='support@fluxboard.com',
    description='A Python SDK for logging and tracking photonic simulation experiments',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/fluxboard/Optixlog',
    project_urls={
        'Homepage': 'https://optixlog.com',
        'Documentation': 'https://optixlog.com/docs',
        'Source': 'https://github.com/fluxboard/Optixlog',
        'Bug Reports': 'https://github.com/fluxboard/Optixlog/issues',
        'Discussions': 'https://github.com/fluxboard/Optixlog/discussions',
    },
    packages=find_packages(),
    classifiers=[
        # Development status
        'Development Status :: 4 - Beta',

        # Intended audience
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',

        # Topic
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Software Development :: Libraries :: Python Modules',

        # License
        'License :: OSI Approved :: GNU Lesser General Public License v2 (LGPLv2)',

        # Python versions
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',

        # Operating systems
        'Operating System :: OS Independent',
        'Operating System :: POSIX',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',

        # Environment
        'Environment :: Console',

        # Framework
        'Framework :: Matplotlib',
    ],
    keywords=[
        'photonics',
        'simulation',
        'fdtd',
        'experiment-tracking',
        'logging',
        'meep',
        'tidy3d',
        'mpi',
        'scientific-computing',
        'visualization',
        'optics',
        'electromagnetics',
    ],
    python_requires='>=3.8',
    install_requires=[
        'requests>=2.25.0',
        'numpy>=1.19.0',
        'matplotlib>=3.3.0',
        'pillow>=8.0.0',
        'rich>=10.0.0',
    ],
    extras_require={
        'mpi': ['mpi4py>=3.0.0'],
        'meep': ['meep>=1.18.0'],
        'tidy3d': ['tidy3d>=2.0.0'],
        'dev': [
            'pytest>=6.0.0',
            'pytest-cov>=2.10.0',
            'black>=21.0',
            'flake8>=3.8.0',
            'mypy>=0.900',
        ],
        'all': [
            'mpi4py>=3.0.0',
            'meep>=1.18.0',
            'tidy3d>=2.0.0',
        ],
    },
    license='LGPL-2.1',
    license_files=['LICENSE'],
    zip_safe=False,
)
