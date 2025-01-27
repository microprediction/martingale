import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text()

setup(
    name="martingale",
    version="0.0.6",
    description="Martingale generation, puzzles and filtering",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/microprediction/martingale",
    author="microprediction",
    author_email="peter.cotton@microprediction.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    packages=["martingale",
              "martingale.processes",
              "martingale.benchmarks",
              "martingale.challenges",
              "martingale.stats"
              ],
    test_suite='pytest',
    tests_require=['pytest','riskparityportfolio'],
    include_package_data=True,
    install_requires=['numpy'],
    entry_points={
        "console_scripts": [
            "martingale=martingale.__main__:main",
        ]
    },
)