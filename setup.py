import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="knockpy",
    version="0.0.1",
    author="Asher Spector",
    author_email="amspector100@gmail.com",
    description="Knockoffs for variable selection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/amspector100/knockpy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)