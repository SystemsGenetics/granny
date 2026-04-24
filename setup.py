import setuptools

install_requires = [
    "ultralytics>=8.0,<9.0",
    "numpy>=1.24",
    "opencv-python>=4.8",
    "pandas>=2.0",
    "pyzbar>=0.1.9",
]

extras_require = {
    "dev": [
        "pytest>=7.0",
        "pytest-cov>=4.0",
    ]
}

setuptools.setup(
    name="granny",
    packages=setuptools.find_packages(),
    url="https://github.com/SystemsGenetics/granny",
    version="1.0a1",
    description="GRANNY is a software package used to rate disorder severity in pome fruits.",
    author="Nhan H. Nguyen, Heidi Hargarten, Loren Honaas, Stephen P. Ficklin",
    license="GNU General Public License v3.0",
    python_requires=">=3.9",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "granny = Granny.GrannyBase:run",
        ]
    },
)
