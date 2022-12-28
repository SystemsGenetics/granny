import setuptools

requirements = """pandas
ipython
numpy
opencv-python
matplotlib>=3.5
h5py<3.0.0
scikit-image
tensorflow<=1.15
pillow
protobuf<=3.20
keras==2.3
""".split()


setuptools.setup(
    name='granny',
    packages=setuptools.find_packages(),
    url="https://github.com/SystemsGenetics/granny",
    version='1.0.4',
    description='GRANNY is an implementation of Mask-RCNN and image processing techniques,\
        developed by the Ficklin Research Program, to rate disorder severity in "Granny Smith" apple.',
    author='Nhan H. Nguyen',
    license='GNU General Public License v3.0',
    python_requires='<=3.7.15',
    install_requires=[
        requirements,
    ],
    entry_points={'console_scripts': [
        'granny = GRANNY.command:main',
    ]},
)
