----------------------------
-        DOG CAMERA        -
----------------------------
Allison Aprile 
2021

aaprile@stevens.edu / allisonaprile22@gmail.com
----------------------------

Hardware: 
	* Raspberry Pi 4 Model B
	* Arducam Raspberry Pi Official Camera Module V2 wih 8 MegaPixel IMX219 Autofocus Replacement
	* Raspberry Pi Camera Mount (optional)

Software Requirements:
	* Python 3.x
	* TensorFlow 2.2.0  (included for build in pi_requirements directory)
	* OpenCV 2 (On Wheels)
	* NumPy
	* PiCamera

----------------------------

To run:
	* Install requirements.txt (in pi_requirements directory) on Pi
	* Install the camera (prior PiCamera testing suggested)
	* Navigate to app directory
	* Run app.py

-----------------------------

Notes:
	* To exit, click the 'x' on the dialouge window. Ctrl+C is not suggested
	* To maximize performance, run on GPU and set camera.framerate to a small number (default: 1)
	* To increase screen resolution, change camera.resolution and raw_capture size to (1080, 720) (as recommended by camera documentation)

