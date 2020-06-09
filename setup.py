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
        "tensorflow>1.14.0,<2;platform_machine!='armv7l'",
        "tensorflow<1.14.0 ; platform_machine=='armv7l'"
    ],
    dependency_links=["https://www.piwheels.org/simple/tensorflow"]
)
