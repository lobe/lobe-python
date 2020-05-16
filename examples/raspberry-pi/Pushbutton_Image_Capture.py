# Import Pi GPIO library button class
from gpiozero import Button
# Import Pi Camera library
from picamera import PiCamera
from time import sleep

# Create a button object for GPIO pin 2
# --> Change GPIO pin as needed
button = Button(2)

# Create a camera object
camera = PiCamera()

while True:
    if button.is_pressed:
        print("Pressed")
        # Start the camera preview, make slightly transparent to see any python output
        #   Note: preview only shows if you have a monitor connected directly to the Pi
        camera.start_preview(alpha=200)
        # Pi Foundation recommends waiting 2s for light adjustment
        sleep(5) 
        # Optional image rotation for camera
        # --> Change or comment out as needed
        camera.rotation = 180
        #Input image file path here
        # --> Change image path as needed
        camera.capture('/home/pi/Documents/image.jpg') 
        #Stop camera
        camera.stop_preview()
    else:
        print("waiting")
    sleep(1)