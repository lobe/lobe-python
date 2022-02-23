from setuptools import setup, find_packages
import pathlib

parent = pathlib.Path(__file__).parent
# get the readme for use in our long description
readme = (parent / "README.md").read_text()

requirements = [
    "pillow~=9.0.1",
    "requests",
    "matplotlib~=3.5.1",
]
tf_req = "tensorflow~=2.8.0 ; platform_machine != 'armv7l'"
onnx_req = "onnxruntime~=1.10.0 ; platform_machine != 'armv7l' and python_version <= '3.9'"  # onnxruntime not to 3.10 yet
tflite_req = "tflite-runtime~=2.7.0 ; platform_system == 'Linux' and python_version <= '3.9'"  # tflite not to 3.10 yet

setup(
    name="lobe",
    version="0.6.2",
    description="Lobe Python SDK",
    long_description=readme,
    long_description_content_type="text/markdown",
    keywords='lobe ai machine learning',
    url="https://github.com/lobe/lobe-python",
    license="MIT",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=requirements,
    extras_require={
        'all': [tf_req, onnx_req, tflite_req],
        'tf': [tf_req],
        'onnx': [onnx_req],
        'tflite': [tflite_req],
    },
    python_requires='>=3.7',
    classifiers=sorted([
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ]),
)
