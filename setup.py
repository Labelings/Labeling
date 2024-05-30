import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="labeling",
    version="0.1.14",
    author="Tom Burke",
    author_email="",
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
    python_requires=">=3.7",
    install_requires=[
        "tifffile",
        "pillow",
        "numpy",
        "scipy"
    ],
    extras_require={
        "dev": [
            "pytest",
            "flake8"
        ]
    }
)
