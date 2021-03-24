from setuptools import setup, find_packages
import sys
import platform


python_version = platform.python_version().rsplit('.', maxsplit=1)[0]

mac_v, _, _ = platform.mac_ver()
if mac_v != '':
    mac_version = '.'.join(mac_v.split('.')[:2])
else:
    mac_version = None

requirements = [
    "pillow>=8.1.1",
    "requests",
    "numpy~=1.19.3",
    "tensorflow~=2.4;platform_machine!='armv7l'",
    "onnxruntime~=1.7.0;platform_machine!='armv7l'"
]

# get the right TF Lite runtime packages based on OS and python version: https://www.tensorflow.org/lite/guide/python#install_just_the_tensorflow_lite_interpreter
tflite_python = None
tflite_machine = None

# get the right python string for the version
if python_version == '3.5':
    tflite_python = 'cp35-cp35m'
elif python_version == '3.6':
    tflite_python = 'cp36-cp36m'
elif python_version == '3.7':
    tflite_python = 'cp37-cp37m'
elif python_version == '3.8':
    tflite_python = 'cp38-cp38'

# get the right machine string
if sys.platform == 'win32':
    tflite_machine = 'win_amd64'
elif sys.platform == 'darwin' and mac_version == '10.15':
    tflite_machine = 'macosx_10_15_x86_64'
elif sys.platform == 'linux':
    if platform.machine() == 'x86_64':
        tflite_machine = 'linux_x86_64'
    elif platform.machine() == 'armv7l':
        tflite_machine = 'linux_armv7l'

# add it to the requirements, or print the location to find the version to install
if tflite_python and tflite_machine:
    requirements.append(f"tflite_runtime @ https://github.com/google-coral/pycoral/releases/download/release-frogfish/tflite_runtime-2.5.0-{tflite_python}-{tflite_machine}.whl")
else:
    print(
        f"Couldn't find tflite_runtime for your platform {sys.platform}, machine {platform.machine()}, and python version {python_version}, please see the install guide for the right version: https://www.tensorflow.org/lite/guide/python#install_just_the_tensorflow_lite_interpreter"
    )

setup(
    name="lobe",
    version="0.4.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=requirements,
)
