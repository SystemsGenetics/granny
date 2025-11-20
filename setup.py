import setuptools

requirements = """
ultralytics
numpy
opencv-python
pytest
""".split()


setuptools.setup(
    name="granny",
    packages=setuptools.find_packages(),
    url="https://github.com/SystemsGenetics/granny",
    version="1.0a1",
    description="GRANNY is a software package used to rate disorder severity in pome fruits.",
    author="Nhan H. Nguyen, Heidi Hargarten, Loren Honaas, Stephen P. Ficklin",
    license="GNU General Public License v3.0",
    python_requires=">=3.9",
    install_requires=[
        requirements,
    ],
    entry_points={
        "console_scripts": [
            "granny = Granny.GrannyBase:run",
        ]
    },
)
