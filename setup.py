from setuptools import setup, find_packages
import sys
import pathlib
import platform

parent = pathlib.Path(__file__).parent
# get the readme for use in our long description
readme = (parent / "README.md").read_text()

python_version = platform.python_version().rsplit('.', maxsplit=1)[0]

mac_v, _, _ = platform.mac_ver()
if mac_v != '':
    mac_v_split = mac_v.split('.')
    mac_major_version = mac_v_split[0]
    mac_minor_version = mac_v_split[1]
    mac_version = '.'.join([mac_major_version, mac_minor_version])
else:
    mac_major_version = None
    mac_version = None

requirements = [
    "pillow~=8.4.0",
    "requests",
    "matplotlib~=3.4.3",
]
tf_req = "tensorflow~=2.5.0;platform_machine!='armv7l'"
onnx_req = "onnxruntime~=1.8.1;platform_machine!='armv7l'"
tflite_req = None

# get the right TF Lite runtime packages based on OS and python version: https://www.tensorflow.org/lite/guide/python#install_just_the_tensorflow_lite_interpreter
tflite_python = None
tflite_platform = None
tflite_machine = None

# get the right python string for the version
if python_version == '3.6':
    tflite_python = 'cp36-cp36m'
elif python_version == '3.7':
    tflite_python = 'cp37-cp37m'
elif python_version == '3.8':
    tflite_python = 'cp38-cp38'
elif python_version == '3.9':
    tflite_python = 'cp39-cp39'

# get the right platform and machine strings for the tflite_runtime wheel URL
sys_platform = sys.platform.lower()
machine = platform.machine().lower()
if sys_platform == 'linux':
    tflite_platform = sys_platform
    tflite_machine = machine
elif sys_platform == 'win32':
    tflite_platform = 'win'
    tflite_machine = machine
elif sys_platform == 'darwin' and machine == 'x86_64':
    if mac_version == '10.15':
        tflite_platform = 'macosx_10_15'
    elif mac_major_version == '11':
        tflite_platform = 'macosx_11_0'
    tflite_machine = machine

# add it to the requirements, or print the location to find the version to install
if tflite_python and tflite_platform and tflite_machine:
    tflite_req = f"tflite_runtime @ https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-{tflite_python}-{tflite_platform}_{tflite_machine}.whl"
else:
    print(
        f"Couldn't find tflite_runtime for your platform {sys.platform}, machine {platform.machine()}, python version {python_version}, and mac version {mac_version}. If you are trying to use TensorFlow Lite, please see the install guide for the right version: https://www.tensorflow.org/lite/guide/python#install_just_the_tensorflow_lite_interpreter"
    )

setup(
    name="lobe",
    version="0.6.1",
    description="Lobe Python SDK",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/lobe/lobe-python",
    license="MIT",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=requirements,
    extras_require={
        'all': [tf_req, onnx_req],
        'tf': [tf_req],
        'onnx': [onnx_req],
    }
)
