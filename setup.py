from setuptools import setup

setup(
    name='axon2d',
    version='0.1',
    description='Axon growth in 2 dimensions',
    url='https://github.com/nichchris/axon2d',
    author='Nicholas Christiansen',
    author_email='nich.h.chris@gmail.com',
    license='MIT',
    packages=['axon2d'],
    install_requires=['setuptools',
                      'numpy',
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.9',
    ],
)
