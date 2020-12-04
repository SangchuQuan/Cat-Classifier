# Cat-Classifier

# Packages Required and Version Suggestion
To run this project, the users should import 3 packages, which are 'Numpy', 'h5py' and 'pyplot' from 'matplotlib'. These packages are composury. Python 3.8.5 is suggested.

# Introduction
This code is aiming at classifying input image, which is contended in a h5.file. The classification result will be 'This image is cat' or 'This image is not a cat'. The structure is a 2-layer neural network. For the hidden layer, there are four neuron. The backpropagation was achieved by applying gradient desent to optimize the weight and bias, which can control the cost of the training. As for training part, the functions to squeeze activation(both hidden layer and last layer) have three options, sigmoid, relu and tanh. User can pick which on to use from the written Class. 
To be more clear, this code would use the training set from h5.file to optimize the weight and bias by gradient desent. At last, it will classify the testing image whether or not it is a cat image.

# The Output Image
If user run this code, the first output image should be one image chosen from training set by user. This can somehow visualize what is in the h5.file. The second image should be a classification result with an image from testing set. 

# The Output Graph
The output graph has one reference cost-iteration curve and one user assigned cost-iteration cuver. Both of them have same number of iterations, but the reference one has 0.0075 learning rate, which is default value. By comparing two curves, user can have better idea what number of iterations and learning rate could be.

# The Output statement
After loading the h5.file, the user would be able to view the dimension of the training set of image, the training set of label, the testing set of image.
To conclude, the three output are some simple unit testing, which visulizing some part of the code to show users what are the pc doing now.

# Instruction for users
There are four input values need users to assign. They are 'number of iterations', 'learning rate', 'first image to view' and 'second image to view'. Here gives some suggestion values for these four inputs. Also, some the reasonable boundary values are also be set so that avoiding input as like learning rate = 0.5, or number of iteration = 200. These unreasonable input will give user an error.

number of iterations: 2500
learning rate: 0.009
first image to view: 24 (This is a street view.)
second image to view: 25 (As the image is 64 * 64 pixel, the cat is not looks well. Better to be prepared to view it.)

User might notice in the main block, the seed used for randomly initializing weight has been assign a fixed value. Please keep this fixed value or it might face the overflow error! There are some ohter error handling which will be activated when user changed some vital part of the code.

# Discussion
This code has an accuracy of 96% for training set and 68%(by choosing proper iteration times, might reach 69%) for testing set. What interesting here is, when I set up a single layer network,  the classification accuracy of testing set is 70%. It should be more accurate as the layers are growing. But it has been not.
The possible reason I think is the number of neurons in the hidden layer is not enough. For this code, it is 4. And, maybe deeper learning is also helpful.

I indeed refered to a blog to find a function for calculating the Cost. Becasue the Cost function introduced by the Youtube video C(w,b)≡1/2n∑∥y(x)−a∥^2, did not work well on the code.

# Reference
Youtube. 2017. Neural Networks. [online] Available at: <https://www.youtube.com/watch?v=IHZwWFHWa-w&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=2> [Accessed 16 November 2020].
Blog.csdn.net. 2020. Neural Network And Deep Learning. [online] Available at: <https://blog.csdn.net/u013733326/article/details/79639509> [Accessed 22 November 2020].
Nielsen, M., 2020. Neural Networks And Deep Learning. [online] Neuralnetworksanddeeplearning.com. Available at: <http://neuralnetworksanddeeplearning.com/chap1.html> [Accessed 20 November 2020].



