from setuptools import setup, find_packages
setup(
    name="lobe",
    version="0.0.1",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "pillow",
        "requests",
        "tensorflow>=1.13.1"
    ]
)
