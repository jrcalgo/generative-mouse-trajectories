from setuptools import setup

setup(
    name='smartDIR',
    version='1.0.0',
    packages=['generative-mouse-trajectories'],
    package_dir={'': 'generative-mouse-trajectories'},
    license='MIT License',
    author='jrcalgo',
    author_email='jacksonr121@outlook.com',
    long_description='Generative-Mouse-Trajectories is for processing and training a GAN on mouse data.'
                     'It is designed to be used in conjunction with the Mouse-Collection-Environment Rust GUI.'
                     'Learns to replicates and imitates human mouse behavior. Python demo GUI included.',
    install_requires=[
        'keras',
        'pytorch',
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
