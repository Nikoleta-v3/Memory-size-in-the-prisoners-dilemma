from setuptools import find_packages, setup

# Read in the version number
exec(open("src/opt_mo/version.py", "r").read())

requirements = ["pandas"]

setup(
    name="opt_mo",
    version=__version__,
    install_requires=requirements,
    author="Nikoleta Glynatsi",
    author_email=("glynatsine@cardiff.ac.uk"),
    packages=find_packages("src"),
    package_dir={"": "src"},
    description="A package used in the study of memory one strategies.",
)
