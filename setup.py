from setuptools import setup, find_packages
setup(
    name="lobe",
    version="0.2.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "pillow",
        "requests",
        "tensorflow>=1.15.2,<2;platform_machine!='armv7l'",
        "tflite_runtime ; platform_machine=='armv7l'"
    ],
    dependency_links=[
        "https://www.piwheels.org/simple/tensorflow",
        "https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_armv7l.whl"
    ]
)
