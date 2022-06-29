# Privacy-Conscious Pet Monitoring System

## Background
**Technical Overview:** Integrating real-time Image Processing/Computer Vision semantic segmentation techniques for use with a Raspberry-Pi powered camera. Skills include _thresholding, clustering, edge detection, filtering, and deep networks such as U-Net Convolutional Neural Nets with and without attention mechanisms_. 

Pet monitoring systems, such as Furbo and Companion are becoming a necessity for pet owners. Some can even provide owners with real-time updates about their pets, or even allows owners to interact with them. However, these can be quite invasive of privacy!

Taking inspiration from Zoom’s ‘Blurred Background’ feature, I am aiming to build a prototype of a Privacy-Conscious Pet Monitoring System. The goals are to:

**I) Classify pet, pet boundary, and non-pet pixels using semantic segmentation, and apply a blur effect to the latter (CURRENT STEP - focusing on optimizing prediction time)**
II) Send bark, growl, and "quiet time" alerts using sound classification
III) Send activity alerts, such as playing or sleeping

## Iteration 1 (August 2021)
In the first iteration, I focused on hardware setup and initial experimentation. This was initiated as a project for an CPE-645 (Image Processing and Computer Vision) at Stevens Institute of Technology, as part of my M.S. Machine Learning. 

For the hardware, I selected a Raspberry Pi 4 Model B and Arducam Raspberry Pi Official Camera Module V2. I also created a rig out of wood and velcro strips. 

Based on the coursework, I experimented with several unsupervised techiques for semantic segmentation:

* **Thresholding:** Applied on grayscaled, Gaussian smoothed image; tried global and variable (adaptive); after looking at histograms, clear that too much noise/variability to be effective
* **K-Means Clustering/Watershed:** Gaussian smoothed image first and tried up to 6 centroids, but again too noisy/variable to be effective
* **Edge Detection:** Single intensity value cannot provide adequate information, especially in noisier images. So attempted finding the change in the largest direction for each pixel using Sobel and Canny approaches, which was an improvement but still too nosiy (even after Hough and Morphological transformations to dissolve and link prominent edges). 

Before even applying these methods, I was aware that it would be unsuccessful given the nature of my data. Therefore, I quickly attempted an ANN to using the Oxford Pets dataset. I only used the dog images, as I only had dog test subjects. After reading some papers, I decided on a **U-Net Convolutional Neural Network**:

* Builds on CNN, focusing on learning anomalies
* Doubles the number of feature channels at each downsampling step, which allows network to propogate context to higher resolution layers
* Fast on GPU (ideal for real-time processing)
* Activation is pixel-wise softmax over final feature map, using cross entropy loss
* Final training validation loss was 0.2

After finding success with classifying the dog pixels, I integrated the privacy aspect. Each camera frame is first copied, which is then processed with a Gaussian filter with a high sigma to blur. Then, the original frame is ran through the classifier to identify dog and non-dog pixels. A new array of the same video frame dimensions is constructed, taking the dog pixels from the original frame and the non-dog pixels from blurred frame. This frame is then pushed to the video feed.

This iteration was ultimately successful given the time and the outlined goals. For future iterations, I want to optimize the prediction and processing tiem as there was sufficient lag; I would like to make use of the built-in GPU on the Raspberry Pi. I also am considering applying Morphological closing or another time of edge linking/smoothing on the boundary pixels, as dog outlines are very chunky.


## Iteration 2 (May 2022)
In this iteration of the project, I focused on better optimizing the Deep Learning model back-ending the blur effect. Using the Oxford III-T Pet dataset, I trained two variations of the U-Net Convolutional Neural Network architecture for the task of Semantic Segmentation: one with Attention
Mechanisms and one without. Given an image, the networks output a pixel-wise classification (i.e. a mask) into the categories of pet, boundary, and not pet.

This iteration was initiated as a project for an CPE-608 (Optimization) course at Stevens Institute of Technology, as part of my M.S. Machine Learning. It focuses on understanding and evaluating the performance of three popular optimizers for Deep Learning: RMSProp, Adam, and Adagrad. I found it as a good opportunity to continue development on this project, especially for improving the backend semantic segmentation model. Given the recent popularity of Attention Mechanisms and Vision Transformers in the medical image segmentation tasks, I really wanted to apply them to this task. The data is also now inclusive of both dogs and cats, increasing the generalizability of the model. 

The **U-Net CNN** was used in Iteration 1, but very quickly in order to get a result. Therefore, it is revisited to properly compare with the **U-Net CN with Attention**. Descriptions of these architectures can be found in the final report provided in the Iteration 2 branch. It is important to note that **Soft Attention** is used due to its differentiability, and therefore compatability with backpropogation. The Attention is groundbreaking as it helps to highlight relevant activations, and therefore weight relevant features, during traning.

This iteration was so far successful with the use of the Attention U-Net CNN. The results of the Attention U-Net CNN against its counterpart are much smoother, although next I will try to add more postprocessing smoothing/closing to improve the boundaries. 

<img width="848" alt="Screen Shot 2022-06-29 at 11 19 07 AM" src="https://user-images.githubusercontent.com/49654275/176473648-3ed38260-29b6-49c7-8e6c-c8107cec7f6f.png">

I also noticed on some other experiments that it had issues with overlapping animals/humans. I hope to expand the data domain by either adding semantic classification (and further blurring) for humans (or at least their faces), and of course integrate this into the camera setup for further evaluation.
