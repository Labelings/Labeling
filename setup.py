import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="labeling",
    version="0.1.10",
    author="Tom Burke",
    author_email="burke@mpi-cbg.de",
    description="A package to create labeling/segmentation information based on pixel values.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Labelings/pyLabeling",
    project_urls={
        "Bug Tracker": "https://github.com/Labelings/pyLabeling/issues"
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent"
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "tifffile==2021.4.8",
        "pillow==9.0.1",
        "numpy==1.21"
    ],
)