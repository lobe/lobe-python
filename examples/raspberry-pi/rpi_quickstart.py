# Import Pi Camera library
from picamera import PiCamera
from time import sleep

#Import Lobe python library
from lobe import ImageModel

# Create a camera object
camera = PiCamera()

# Load Lobe TF model
# --> Change model path
model = ImageModel.load('/home/pi/model')



if __name__ == '__main__':
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

    # Run photo through Lobe TF model and get prediction results
    result = model.predict_from_file('/home/pi/Documents/image.jpg')

    print(result)

    sleep(1)