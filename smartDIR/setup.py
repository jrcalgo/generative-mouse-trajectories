from setuptools import setup

setup(
    name='smartDIR',
    version='1.0.0',
    packages=['smartdir'],
    package_dir={'': 'smartDIR'},
    license='MIT License',
    author='jrcalgo',
    author_email='jacksonr121@outlook.com',
    long_description='SmartDIR is a Python module for collecting, processing, and training a GAN on mouse and keyboard data.'
                     'It is designed to be used in conjunction with the smartDIR GUI collection environment.'
                     'Replicates and imitates human mouse and keyboard activity.',
    install_requires=[
        'pysdl2',
        'keras',
        'tensorflow',
        'tensorboard',
        'numpy'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ]
)
