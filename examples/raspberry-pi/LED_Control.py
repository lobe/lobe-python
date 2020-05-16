#import Pi GPIO library button class
from gpiozero import LED
from time import sleep

#Import Lobe python library
from lobe import ImageModel

# Define LEDs and GPIO pin numbers
# --> Change GPIO pins as needed
red_led = LED(17)
yellow_led = LED(27)
green_led = LED(22)

# Load Lobe TF model
# --> Change model path
model = ImageModel.load('/path/to/model')
# Run photo through Lobe TF model and get prediction results
# --> Change image path
result = model.predict_from_file('/path/to/image/image.jpg')

# Function that takes in a single string, label, and turns on corresponding LED
# --> Change label1, label2, and label3 to reflect your Lobe model labels.
#     Note: Labels are case sensitive!
def ledSelect(label):
    print(label)
    if label == "label1": 
        yellow_led.on()
        sleep(5)
    if label == "label2":
        green_led.on()
        sleep(5)
    if label == "label3":
        red_led.on()
        sleep(5)
    else:
        yellow_led.off()
        green_led.off()
        red_led.off()

while True:
    # result.prediction outputs a single string corresponding to the 
    # top prediction from Lobe model classes
    ledSelect(result.prediction)

    # Test function (uncomment for debugging)
    #ledSelect('label1')

    sleep(1)
