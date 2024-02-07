import setuptools

requirements = """
ultralytics
numpy
opencv-python
""".split()


setuptools.setup(
    name="granny",
    packages=setuptools.find_packages(),
    url="https://github.com/SystemsGenetics/granny",
    version="1.0",
    description="GRANNY is an implementation of Mask-RCNN and image processing\
    techniques, developed by the Ficklin Research Program, to rate disorder\
    severity in pome fruits.",
    author="Nhan H. Nguyen",
    license="GNU General Public License v3.0",
    python_requires=">=3.8",
    install_requires=[
        requirements,
    ],
    entry_points={
        "console_scripts": [
            "granny = GRANNY.GrannyBase:cli",
            "granny-gui = GRANNY.GrannyBase:gui",
        ]
    },
)
