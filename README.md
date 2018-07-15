# **Behavioral Cloning** 

---

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/rand_image.png "Normal Image"
[image7]: ./examples/rand_image_flipped.png "Flipped Image"
 
---
### Files in the repo

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results
* vidoe.mp4 showing the car driving in autonomous mode for about 1.5 laps on track #1

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes and depths 5 (model.py lines 93-99) 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 90). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 98). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 45-73).
In particular, images from multiple cameras have been used (center, left, and right). In addition, flipped images are added. 

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track (please refer to video.mp4).

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 117).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach


My first step was to use a convolution neural network model similar to the AlexNet,  I thought this model might be appropriate because it performs well on image classification and it matters here becuase it is essential to recognize lane lines in different conditions (direction, curvature) to make the model work.  

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set (a ratio of 0.2). I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model by adding a dropout layer. Additionaly, adding more trainning data (i.e. adding a recovery run, using multiple cameras, and adding flipped images) helped to reduce overfitting. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes:
*Node that all the params were adopted from the lecture. 

Layer (type)                 Output Shape              Param #    
=================================================================
cropping2d_1 (Cropping2D)    (None, 65, 320, 3)        0         
_________________________________________________________________
lambda_1 (nomarlization)     (None, 65, 320, 3)        0         
_________________________________________________________________
conv2d_1 (6, (5,5))          (None, 61, 316, 6)        456       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 30, 158, 6)        0         
_________________________________________________________________
conv2d_2 (6, (5,5))           (None, 26, 154, 6)        906       
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 13, 77, 6)         0         
_________________________________________________________________
dropout_1 (0.5)              (None, 13, 77, 6)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 6006)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 120)               720840    
_________________________________________________________________
dense_2 (Dense)              (None, 84)                10164     
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 85        
=================================================================
Total params: 732,451
Trainable params: 732,451
Non-trainable params: 0

#### 3. Creation of the Training Set & Training Process
In addition to the sample data provided by 

To capture good driving behavior, I first recorded two laps on track one using center lane driving and one lap in reverse direction. 

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from off-track conditions. 

To augment the data sat, I also flipped images and angles thinking that this would help with the left/right turn bias. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

After the collection process, I had around 40k number of data points.
I then preprocessed this data by 
1. Crop the images (top 70 and bottom 25 pixels are cropped out)
2. Normalize the data (x / 255.0 -0.5)

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I trained the model for 10 epochs and saved the best model. I used an adam optimizer so that manually training the learning rate wasn't necessary.
