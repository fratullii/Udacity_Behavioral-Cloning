# Behavioral Cloning

![initial](driving_udacity.png)

This repository contains my submission for the Udacity - Self-Driving Car Nanodegree project. The file of interest - i.e. what it is different from the original starting repo -  are:

- the ``` model.py ``` script, containing the neural network definition and training
- ``` video.mp4```, showing the final result on the first track

Here a description 00 of the solution proposed towards the solution.

### Architecture

The architecture used is the plain vanilla NVIDIA architecture, proposed in the paper [End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/deep-learning-self-driving-cars/) , consisting of 5 convolutional layers and 4 fully connected layers. No batch normalization layers have been included, since the network performed well without it.

There are two preprocessing layer that have been included in the network:

- A normalization layer: normalize in the range 0-255, with mean and standard deviation 1.
- A cropping layer: only the area of interest is retained, where the road and its boundaries can be visualized.

Here the architecture containing all the information about the different layers ([source](https://devblogs.nvidia.com/wp-content/uploads/2016/08/cnn-architecture-624x890.png)).

![Nvidia](cnn-architecture.png)

### Training datasets

The datasets used is the example dataset provided by Udacity. It contains images from the first track. By proper data augmentation it is possible to obtain other data useful for training the model.

All three images are used (center, left and right camera are used). The angle related to the images from the lateral cameras is the angle related to the central image, shifted towards the opposite side of a certain corrective factor. This shift is meant to suggest the car to return at the center of the road when the position is lateral, since the only image considered when the neural network drives is the central. In this way the model learns how to recover the car if it ends up on the road sides.

The correction factor affect how the car drives. If too low, the car does not learn how to deal with sharp curves and risks to go out of the track, if too high it reacts aggressively by steering too much when not at the center of the road, causing it to wiggle and risking not to stay on track when approaching a sharp curve. After some attempts were made, the value of **0.2**  turned out to be a good compromise.

The track for this project is made mostly of left curves and only one sharp right curve. To compensate this bias, all the images in a batch are flipped with respect to the central vertical axis (alongside the steering angle undergoing a change of sign), and added to the original images. The model can thus better generalize in that right curve avoiding overfitting.

The loss function used during training, is the **Mean Square Error**, the optimizer **Adam**. The model is trained over 8 epochs with batches of 32 frames each. Considering the three cameras and the flip operation, batch size ends up to be 32 * 3 * 2 = **192**.

The script used a python generator for training. It allows to retain in memory (and process) only one batch at a time, saving memory space.

### Result

The file ```video.mp4``` shows the front camera of the car, driving in the road and never leaving the track.
