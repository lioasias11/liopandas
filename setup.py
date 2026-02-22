from setuptools import setup, find_packages

setup(
    name="liopandas",
    version="0.1.0",
    description="A minimal pandas-like library built from scratch using NumPy.",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=["numpy"],
)
