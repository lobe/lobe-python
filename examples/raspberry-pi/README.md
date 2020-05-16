# Lobe on Raspberry Pi

This guide provides a walkthrough for getting a Lobe TensorFlow model onto the Raspberry Pi 4.*  

It is assumed you are starting with a Pi in a remote headless configuration already enabled with [SSH](https://www.raspberrypi.org/documentation/remote-access/ssh/), Remote Desktop or VNC, and WiFi access.

**Note that this procedure will work for a Pi 3, however it is unlikely the model will run successfully.*

## Things you'll need
### Hardware
* Raspberry Pi 4 
* SD Card with Raspbian (desktop version recommended)
* USB-C Power Supply (5.1V,3A)
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
4. Navigate into your Lobe directory, and download the setuptools, TensorFlow library version 1.13.1 and the lobe-python package:
    ```
        pip3 install setuptools
        pip3 install tensorflow==1.13.1
        pip3 install git+https://github.com/lobe/lobe-python
    ```

## Getting the Lobe TF Model on your Pi
1. On your PC or Mac, open WinSCP (or your preferred remote file transfer method) and connect to the Pi. Once you are connected, on the Pi side, navigate to the *Lobe* directory and create a new folder called "*model*".
2. Select and the Lobe TensorFlow model files on your PC or Mac into the *model* folder on the Pi.
    
    *Note: You should have two files -- "saved_model.pb" and "signature.json" -- as well as a folder named "variables" with two files inside.*

## Running the Lobe TF Model
1. On the Pi, open the *basic_usage.py* file with your favorite Python editor.
2. In line 4, update the model path to point to the folder with your Lobe TF model files.
3. Decide which image import option you're using: 
    Option 1: Image from file
    Option 2: Image from URL
    Option 3: Image from Pillow image
4. Update the image path for your selected option and comment out the other two.
5. Run your python program with Python 3:
    '''
    python3 basic_usage.py
    ''' 

### Troubleshooting
1. Check that the Pi power light is bright red. A dim red light indicates insufficient power.
2. Be sure you're installing and running your TF code using Python 3.
3. If the TensorFlow module is not recognized, try re-installing:
    ''' 
        pip3 install tensorflow == 1.13.1
    '''
4. If you're using multiple components, be sure they are all connected to the same common ground.

## Example Code and Going Further
Sample code is included to show you how to get started using the Pi Camera and use the GPIO pins as inputs and outputs.

### Pushbutton Image Capture
This sample code is a simple program for using a pushbutton to take an image snapshot. The program saves the resulting image in the Documents folder on the Pi. 

#### Running the program
Connect a pushbutton to GPIO pin 2 (or change the code to use another GPIO pin). Then run the program using Python 3 and press the pushbutton to take a snapshot:
    ''' 
    python3 Pushbutton_Image_Capture.py
    '''

Read through the comments to learn how the code works. You do not need to change anything for it to run.

### LED Control
This sample code shows you how to use the Lobe model prediction results to turn on different colored LEDs.

For this program to run successfully, you will need to update the following things in the code:
1. Lobe model folder path (line 16)
2. Image path (line 19)
3. Lobe model labels (lines 26, 29, 32). 
    Note: If you have more or less than three labels, remove or add LEDs accordingly.


#### Running the program
After you've made the updates to the progam outlined above, connect the positive side of a red LED to GPIO pin 17, a yellow LED to GPIO pin 27, and a green LED to GPIO pin 22. Connect 100Ohm or similar resistors to the negative LED pins, and connect those to a Pi GND pin.

Run the program with Python 3. Note that the TensorFlow model can take some time.
    ''' 
    python3 LED_Control.py
    '''

You can change, remove, or add LED GPIO pins using lines 10 - 12.

If the program does not behave as expected, you can use the test function in line 46, just be sure to pass in a label from your Lobe model. Note that the labels are case sensitive.