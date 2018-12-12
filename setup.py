import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pygbmexpl",
    version="0.0.1",
    author="Richard Angell",
    author_email="richardangell37@gmail.com",
    description="Package to explain gbm predictions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/richardangell/pygbmexpl",
    packages=setuptools.find_packages(),
    #install_requires=[
    #    'xgboost'
    #],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
)