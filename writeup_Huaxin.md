#**Traffic Sign Recognition** 

##Writeup
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[https://github.com/JackyGu1042/German-traffic-sign-classifier.git/]: # (Image References)

[image4]: ./Stop.jpg "Traffic Sign 1"
[image5]: ./30.jpg "Traffic Sign 2"
[image6]: ./60.jpg "Traffic Sign 3"
[image7]: ./Kindergarten.jpg "Traffic Sign 4"
[image8]: ./Turn_right_ahead.jpg "Traffic Sign 5"
[image9]: ./50.jpg "Traffic Sign 6"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

###Data Set Summary & Exploration

####1. Basic summary of the data set
The code for this step is contained in the 2nd code cell of the IPython notebook.  

I used the pickle library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.
The code for this step is contained in the 3rd code cell of the IPython notebook.  

* I print out the basic information of dataset, like data size, shape and classes.
* Print out the train, valid and test dataset's bar chart for visualization in the 4th code cell.

###Design and Test a Model Architecture

####1. Preprocessed the image data
The code for this step is contained in the 5th code cell of the IPython notebook.

I use skimage library's color.rgb2gray function to realize this feature. And I normalized the image by just divide 255, because smaller value could reduce the operation error of computer.

I have tried two different preprocession:
* LeNet structure: remain the input as (32, 32, 3) RGB image
    1. rate = 0.0020, epoch = 35, the final valid accuracy is 84.8%
    2. rate = 0.0015, epoch = 35, the final valid accuracy is 90.8% 
    3. rate = 0.0010, epoch = 35, the final valid accuracy is 88.4% 
    4. rate = 0.0005, epoch = 35, the final valid accuracy is.89.8%
* LeNet structure: color to grayscale preprocession
    1. rate = 0.0020, epoch = 35, the final valid accuracy is 92.6%
    2. rate = 0.0015, epoch = 35, the final valid accuracy is 93.5%
    3. rate = 0.0010, epoch = 35, the final valid accuracy is 90.3% 
    4. rate = 0.0005, epoch = 35, the final valid accuracy is 90.1%
         
After some test, I found with same model structure, learning rate and epoch, the valid accuracy is lower without color to grayscale preprocession. And the training time also decreases, because decrease the dimension from 32x32x3 to 32x32x1.         
         
So finally, I keep the color to grayscale as the preprocession, and one more reason is that after read the image of dataset I found the biggest difference between each traffic signs is not color but shape. So remove the color information could increase the efficiency of classification.

In the 6th cell, it also prints out an example of a traffic sign image after grayscaling.

####2. Set up training, validation and testing data  
* Follow the original dataset, use train.p as training set and valid.p as validation set, and test.p as test set.

* Randomly shuffle the trianing dataset in the 6th code cell.

####3. Final model architecture
The code for my final model is located in the 7th cell of the ipython notebook. 

I use LeNet structure for model architecture, add one more convolution and one more full connected layer. And change the activation to RELU6 function.
My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscacle image   					| 
| Convolution 5x5x6    	| 1x1 stride, Valid padding, outputs 28x28x6 	|
| RELU6					|												|
| Max pooling	      	| 1x1 stride,  outputs 28x28x6  				|
| Convolution 5x5x10   	| 1x1 stride, Valid padding, outputs 24x24x10 	|
| RELU6					|												|
| Max pooling	      	| 2x2 stride,  outputs 12x12x10    		 		|
| Convolution 3x3x16   	| 1x1 stride, Valid padding, outputs 10x10x16 	|
| RELU6					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16    		 		|
| Flaten        	    | outputs 400  									|
| Fully connected		| outputz 120  									|
| RELU6					|												|
| Fully connected		| outputz 84 									|
| RELU6					|												|
| Fully connected		| outputz n_classes								|


####4. Trained model
The code for training the model is located in the 7th cell of the ipython notebook. 

To train the model:
* The learning rate is 0.0011
* The Epochs is 50
* The Batch size is 128

####5. The approach taken for finding a solution. 

The code for calculating the accuracy of the model is located in the 9th cell of the Ipython notebook.

My final model results were:
* validation set accuracy of 95.6% 
* test set accuracy of 94.1%

If an iterative approach was chosen:
####1. What was the first architecture that was tried and why was it chosen?
I use LeNet structure as the initial architecture. Because this architecture is the most advanced or complex architecture which I can find in the lesson.

####2. What were some problems with the initial architecture?
I have try to use different learning rate(from 0.0002 to 0.0012) and epoch (from 10 to 40), I found the accuracy of valid set is not easy to reach 93% for the initial architecture.

####3. How was the architecture adjusted and why was it adjusted? 

Test different structure with different layer:
* LeNet structure: 2 convolution layer, 2 fully connected layer:
     1. rate = 0.0015, epoch = 35, the final valid accuracy is 93.5%
     2. rate = 0.0010, epoch = 35, the final valid accuracy is 90.3% 
     3. rate = 0.0005, epoch = 35, the final valid accuracy is 90.1%
* LeNet structure: 2 convolution layer, 1 fully connected layer:
     1. rate = 0.0015, epoch = 35, the final valid accuracy is 92.2%
     2. rate = 0.0010, epoch = 35, the final valid accuracy is 92.3%
     3. rate = 0.0005, epoch = 35, the final valid accuracy is 90.1%
* LeNet structure: 2 convolution layer, 3 fully connected layer:
     1. rate = 0.0015, epoch = 35, the final valid accuracy is 92.7%
     2. rate = 0.0010, epoch = 35, the final valid accuracy is 93.8%
     3. rate = 0.0005, epoch = 35, the final valid accuracy is 90.6%

In this structure, the average of all epoch is obvious higher than first two structure.

* LeNet structure: 1 convolution layer, 2 fully connected layer:
     1. rate = 0.0015, epoch = 35, the final valid accuracy is 88.9%
     2. rate = 0.0010, epoch = 35, the final valid accuracy is 89.5%
     3. rate = 0.0005, epoch = 35, the final valid accuracy is 88.9%
* LeNet structure: 3 convolution layer, 2 fully connected layer:
     1. rate = 0.0015, epoch = 35, the final valid accuracy is 95.0%
     2. rate = 0.0010, epoch = 35, the final valid accuracy is 93.7%
     3. rate = 0.0005, epoch = 35, the final valid accuracy is 91.5%
* LeNet structure: 3 convolution layer, 3 fully connected layer:
     1. rate = 0.0015, epoch = 35, the final valid accuracy is 94.1%
     2. rate = 0.0010, epoch = 35, the final valid accuracy is 95.6%
     3. rate = 0.0005, epoch = 35, the final valid accuracy is 91.6%

According to valid accuracy and test accuracy, the best architecture is 3 convolution layer, 3 fully connected layer with 0.010 learning rate and 35 epoch. However, I found the accuracy of new image classification decrease with this structure.

Then choose the best one of the test above, and change the pooling function and activation:
* Original LeNet structure: change max pooling to average pooling:
     1. rate = 0.0015, epoch = 35, the final valid accuracy is 93.6%
     2. rate = 0.0010, epoch = 35, the final valid accuracy is 93.3%
     3. rate = 0.0005, epoch = 35, the final valid accuracy is 90.0%

According to above result, it seems that the performance become worse when change max pooling to average pooling.

* Original LeNet structure: change activation to relu6 function:
     1. rate = 0.0015, epoch = 35, the final valid accuracy is 95.2%
     2. rate = 0.0010, epoch = 35, the final valid accuracy is 95.1%
     3. rate = 0.0005, epoch = 35, the final valid accuracy is 92.2%

This result is the best one. However, the new image classification decrease again.

After several test above, I found the best architecture is 3 convolution layer, 3 fully connected layer with max pooling and relu6 activation.

####4. Which parameters were tuned? How were they adjusted and why?
* Number of convolution layer, when convolution increase, the accuracy become better.
* Number of fully connected layer, when fully connected layer increase, the accuracy become better.
* Value of learning rate, when this rate increase, the accuracy could improve fast by each epoch. But if learning is too large, the final result is also not good(I think this case is overfitting).  
* Number of epoch, when learning rate is small, this number need to be big.  

####5. What are some of the important design choices and why were they chosen? 

#####1. What architecture was chosen?
    
I chose 3 convolution layer, 3 fully connected layer with max pooling and relu6 activation.

#####2. Why did you believe it would be relevant to the traffic sign application?

I think that traffic sign is not like handwrite number classification, in order to identify traffic sign, the model need three level to process:

* Identify different basic angle or line.
* Collect the basic element into various shape groups or number groups. 
* According to the different shape or number's combination, to judge the class of sign.  

So two convolution or two full connect layer is not enough for traffic sign classification.

####6. How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

After change the architecture of model, the accuracy has obvious improvement in both of training, validation and test stage. However, as mention before, the accuracy of new image which I use from Internet is decrease. 

I think the possible reason is that the new image's sign relative size is different from training data set. Specifically, in the training dataset, the sign's size is around 12x12, but in the new image, the sign's size is around 20x20. After upgrade the model architecture, when classify the new image, the model become a little overfitting with the training data.  

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

1.![alt text][image4] 2.![alt text][image5] 3.![alt text][image6] 
4.![alt text][image7] 5.![alt text][image8]

* The 1st to 4th the size is bigger than original training dataset
* The 5th is the different color and size with original training dataset
* The 6th is both different color and size as training dataset, moreover, the shape of the sign is still different because of graph shooting angle. 

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        | Prediction    	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| 30 km/h     			| 30 km/h 										|
| 60 km/h				| 60 km/h										|
| Children crossing		| Children crossing				 				|
| Turn right ahead		| Turn right ahead     							|
| 50 km/h	    		| No entry          							|


The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83.333%. This is lower than the test set accuracy.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. 

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

* For the 1st image, the model is relatively sure that this is a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.99e-01     			| Stop      									| 
| 2.68e-07 				| No entry 										|
| 2.52e-08				| Keep right									|
| 1.42e-08	   			| No vehicles					 				|
| 1.15e-08			    | Speed limit (30km/h) 							| 

* For the 2nd image, the model is relatively sure that this is a 30 km/h sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.99e-01     			| Speed limit (30km/h)							| 
| 7.30e-05 				| Speed limit (20km/h)							|
| 8.03e-06				| Speed limit (50km/h)							|
| 4.87e-08	   			| Keep right 					 				|
| 1.15e-08			    | Speed limit (70km/h) 							|


* For the 3rd image, the model is relatively sure that this is a 60 km/h sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.94e-01     			| Speed limit (60km/h)							| 
| 5.49e-03 				| Speed limit (50km/h)							|
| 5.61e-05				| Wild animals crossing							|
| 2.63e-05	   			| Speed limit (30km/h)			 				|
| 1.83e-05			    | Keep right        							| 

* For the 4th image, the model is relatively sure that this is a Children crossing sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.99e-01     			| Children crossing								| 
| 2.42e-06 				| Dangerous curve to the right					|
| 1.72e-08				| Go straight or right							|
| 1.13e-08	   			| Keep right					 				|
| 2.74e-09			    | No passing         							| 

* For the 5th image, the model is relatively sure that this is a turn right ahead sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00e+00     			| Turn right ahead								| 
| 4.42e-08 				| Keep left										|
| 1.09e-08				| Stop											|
| 3.65e-09	   			| Ahead only					 				|
| 8.25e-10			    | Go straight or left  							| 

* For the 6th image, the model is relatively sure that this is a No entry sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.62e-01     			| No entry   									| 
| 3.77e-02 				| Turn right ahead								|
| 8.02e-05				| Go straight or left							|
| 4.26e-05	   			| No passing					 				|
| 1.76e-05			    | Turn left ahead      							| 

