import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Komori",
    version="0.20",
    author="Domagoj K. Hackenberger",
    author_email="domagoj@bioquant.hr",
    description="komori",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta"],
)
