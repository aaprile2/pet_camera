''' MAIN PROGRAM FILE '''
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import process_frames
from tensorflow import keras
import tensorflow as tf

''' Load ANN for dog detection '''
print('Loading model....')
model = keras.models.load_model('dog-seg-model/dog_seg_model')

''' Intialize camera and grab a reference to the raw camera capture '''
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 1
raw_capture = PiRGBArray(camera, size=(640, 480))

# Allow camera to warmup
time.sleep(0.1)

''' Processing function '''
def process(img):
    # Histogram equalization
    equ = process_frames.histogram_equalization(img)
    
    # Detect dogs in image
    mask = process_frames.get_dog_mask(equ, model)
    
    # Blur background
    processed = process_frames.blur_background(equ, mask)

    return processed

''' Main function '''
def main():
    # Allow camera to warmup
    time.sleep(0.1)

    # Capture frames from the camera
    print('Broadcasting....')
    for frame in camera.capture_continuous(raw_capture, format='bgr', use_video_port=True):
        # Grab the raw NumPy array and initialize timestamp & occupied/unoccupied text
        img = frame.array
        
        # Process image by enhancing with histogram equalization and blurring 
        # the background (i.e. blurring any pixels not containing a dog)
        img = process(img)
        
        # Show the frame
        cv2.imshow('Dog Cam', img)
        key = cv2.waitKey(1) & 0xFF
        
        # Clear the stream in preparation for the next frame
        raw_capture.truncate(0)
        
        # If the 'q' key was pressed, break from the loop
        try:
            if cv2.getWindowProperty('Dog Cam', 0) == -1:
                break
        except:
            break
            
    print('Exited. Goodbye!')
    exit()
    
''' Execute main function '''
if __name__ == "__main__":
    main()