# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./report_images/centrelane.jpg "Centre Lane"
[image3]: ./report_images/recovery1_1.jpg "Recovery Image"
[image4]: ./report_images/recovery1_2.jpg "Recovery Image"
[image5]: ./report_images/recovery1_3.jpg "Recovery Image"
[image6]: ./report_images/recovery1_4.jpg "Recovery Image"
[image7]: ./report_images/flipimage.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

The variable set_speed is set to 30 for a max. speed of 30 km/h

In the function telemetry(), rgb image_array is converted to bgr as the images used for training are in bgr colorspace. The code for conversion is referenced to the Udacity forum for this project.


        image_array = np.asarray(image)
        # following lines of code is newly added for color space conversion.
        rgb = image_array             
        bgr = rgb[...,::-1]
        steering_angle = float(model.predict(bgr[None, :, :, :], batch_size=1))

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model (model.py code cell [6] lines 18-24) is referenced that one used by Nvidia for end-to-end learning for self-driving. (https://arxiv.org/pdf/1604.07316.pdf). The network consists of 9 layers, including a normalization layer, 5 convolutional layers
and 3 fully connected layers

The image data is normalized in the model using a Keras lambda layer (code line 9) and then cropped to extrract the region of interest (code line 10). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py code cell [6] lines 13, 15, 17). 

The model was trained and validated on different data sets (stored in same file directory) to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 31).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, reverse direction driving, image flipping and a "wild" driving, which makes strong steering at top speed (31km/h) around curves and recovering from the left and right sides of the road when the car is tending to driving off to the side

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the Nvidia for end-to-end learning for self-driving. I thought this model might be appropriate because behaviorial cloning in this project is basically an end-to-end learning. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model by adding dropout layer (0.5) after each of the first three convolution layers.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, apart from the normal driving, I added more driving scenarios to the dataset by e.g. reverse direction driving, image flipping and a "wild" driving, which makes strong steering at top speed (31km/h) around curves and recovering from the left and right sides.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. It seems if the dataset contains a more portion of normal driving data plus a certain portion of exceptional cases e.g. strong steering around curves, recovering can help to come up a more generalized model.

#### 2. Final Model Architecture

The final model architecture (model.py code cell [6] lines 6-18) consisted of a convolution neural network with the following layers and layer sizes ...

model.add(Convolution2D(24, 5, 5, subsample = (2,2), activation="relu"))
model.add(Dropout(0.5))
model.add(Convolution2D(36, 5, 5, subsample = (2,2), activation="relu"))
model.add(Dropout(0.5))
model.add(Convolution2D(48, 5, 5, subsample = (2,2), activation="relu"))
model.add(Dropout(0.5))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded four laps (two laps clockwise, two laps counter-clockwise) on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... The following images show a recovery when the car is about to running off the track, it is steered back to the middle.

![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image7]

Finally I recorded one lap where I used nearly full speed (31 km/h) to run even around curves. 

After the collection process, I had 38043 number of data points. I then preprocessed this data by normalization

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The number of epochs I used was 10 and I used an adam optimizer so that manually training the learning rate wasn't necessary.
