# Lobe on Raspberry Pi

This guide provides a walkthrough for getting a Lobe TensorFlow model onto the Raspberry Pi 4.*  

It is assumed you are starting with a Pi in a remote headless configuration already enabled with [SSH](https://github.com/microsoft/rpi-resources/tree/master/headless-setup), Remote Desktop or VNC, and WiFi access.

**Note that this procedure will work for a Pi 3, however it is unlikely the model will run successfully.*

## Things you'll need
### Hardware
* Raspberry Pi 4 
* SD Card with Raspbian (desktop version recommended)
* USB-C Power Supply (5.1V, 3A)
* Pi Camera 
* Case (optional but recommended)
* Pi Camera Mount (optional but recommended)

### Software
* WinSCP (or other remote file transfer program)
* Remote Desktop or VNC

## Setup
1. Carefully plug in your Pi Camera module. [Instructions can be found here.](https://projects.raspberrypi.org/en/projects/getting-started-with-picamera/2) 
2. Power up your Pi and log in to the desktop view.
3. On the Pi, open a terminal window and create a directory where you want to store your Lobe model and code. For example, in the /home/pi folder:
    ```
        mkdir Lobe
    ```
4. Navigate into your Lobe directory, and download the setuptools and the lobe-python package:
    ```
        pip3 install setuptools
        pip3 install git+https://github.com/lobe/lobe-python
    ```

## Getting the Lobe TF Model on your Pi
*Note: you can also use a USB drive to transfer the files*
1. On your PC or Mac, open WinSCP (or your preferred remote file transfer method) and connect to the Pi. 
1. On the Pi side navigate to the *Lobe* directory and create a new folder called "*model*". 
2. Select and the Lobe TensorFlow model files on your PC or Mac into the *model* folder on the Pi.
    
    *Note: You should have two files -- "saved_model.pb" and "signature.json" -- as well as a folder named "variables" with two files inside.*


## Running the Lobe TF Model
1. On the Pi, open the *rpi_quickstart.py* file with your favorite Python editor.
1. In line 4, update the model path to point to the folder with your Lobe TF model files. *Hint: if you created the directory with the same name as above you don't need to change anything.*
1. Run your python program with Python 3:
    '''
    python3 rpi_quickstart.py
    ''' 

### Troubleshooting
1. Check that the Pi power light is bright red. A dim red light indicates insufficient power.
2. Be sure you're installing and running your TF code using Python 3.
3. If the TensorFlow module is not recognized, try re-installing:
    ''' 
        pip3 install tensorflow == 1.13.1
    '''
4. If you're using multiple components, be sure they are all connected to the same common ground.