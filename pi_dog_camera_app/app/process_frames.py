''' FRAME PROCESSING FUNCTIONS '''
import cv2
import numpy as np

# Perform histogram equalization on image (to improve contrast)
def histogram_equalization(img):
    # Convert to YUV color space for processing
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # Perform histogram equalization
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, : , 0])

    # Convert back to BGR color space
    bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    return bgr
    

# Detect dog in image; produce mask
def get_dog_mask(img, model):
    # Reshape/make into a 'batch' to fit classifier
    x = np.zeros((1, ) + (160, 160) + (3, ), dtype = 'float32')
    img_resized = cv2.resize(img, (160, 160))
    x[0] = img_resized

    # Make prediction
    pred = model.predict(x)

    # Define mask & threshold (so binary)
    mask = np.argmax(pred[0], axis=-1)

    # Markers 0 and 2 specify dog; 1 is background
    # - map 0 and 2 together
    mask[mask == 2] = 0

    # Mask output is in range [0:1]; scale to [0:255]
    mask = mask.astype('float64')
    mask = mask * 255    # Scale by 255
    mask = mask.astype('uint8')

    # Mask is now a binary image; resize to original frame size
    mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))

    return mask_resized
    
    
# Blur image background (i.e. pixels not containing a dog)
def blur_background(img, mask):
    # Create a blurred copy of image - use large kernel and sigma so not smoothing effect
    blurred = cv2.GaussianBlur(img, (25, 25), 200)

    # Stack mask into three-channels
    mask = np.stack((mask,)*3, axis=-1)

    # Replace background pixels with blurred pixels
    camera_out = np.where(mask == 255, blurred, img)

    return camera_out