import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    requires = fh.read()

setuptools.setup(
    description="SKeletOn ObjecT Segmentation (SKOOTS)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=requires,
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "skoots-train = skoots.train.__main__:main",
            "skoots = skoots.__main__:main",
            "skoots-validate = skoots.validate.__main__:main",
        ]
    },
)
