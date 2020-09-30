# Using a Neural Network to do simple addition

Neural Networks are collections of digital neurons that individually perfrom fairly simple tasks: take in numbers, multiply them, output a new number. The whole is greater than the sum of its parts in a Neural Network, however, as groups of these simple neurons can create logic gates, the building blocks of complex computation. This allows Neural Networks to regonize features in images, do speech to text transcription, out-compete world champion Go and Starcraft players, drive passengers thousands of miles, and more.

So why would someone use a Neural Network to add two numbers together? I did this in order to better understand NNs and how to make them. I'm a stubborn programmer who would rather spend hundreds of hours reinventing the wheel so as to understand how it's made on a fundamental level. Most tutorials offered by companies like Google use packaged datasets and don't explain what's going on "under the hood" of their network architecture.

The task of adding two numbers together using a NN will employ Supervised Learning, a type of Machine Learning which simply means we have an input (in this case the two numbers we want to add) and an output (the sum of the two numbers) and a bunch of existing data that we can use to train the model. The existing data will come into play soon, but it's just sets of numbers and their corresponding sums. It is also a "regression" problem as the output of our NN will be a real number (as opposed to a classification, like "even" or "odd") and to be more specific, it will be a floating point number which simply means it will have decimal precision as opposed to being a whole number.

The process will essentially be:
1. Create the training data. This will basically look like this: (1,2,3), (5,5,10), (-2,3,1) and so on. Notice that the first two numbers in each set are added to create the third number. We have to supply the network with a lot of training data in order for it to learn. We will do this programmatically, so we don't have to waste time creating a spreadsheet with all this data, though we could do it that way.
2. Build the Neural Network model. This is the fun part, and because neural nets are so new, there isn't really a best-practice way of creating their architecture. The more neurons you put into the network, the more complex it can become, but more complex models are not always the most accurate, and they can be slow to train.
3. Use the network. Put in two numbers and see if it adds them up correctly.

I'm brand new to Python so I will take the time to explain a bit of the syntax and logic behind my code. I won't go over an introduction to Google Colab, but for that you can refer to other videos or articles. In short, it's a way of writing and executing Python code on Google's cloud-based computers, which are equipped with powerful hardware intended for use in machine learning.

## Create the Training Data
Let's dive into some code. The first two lines enlist the help of two existing code libraries, which give us some helpful functions we can use.

```python
import numpy as np 
import random
```

numpy is a library for Python that gives us some handy math functions. For the most part we will just be using the array data type it offers, as we will be creating large groups of data and need it to be organized in a way that Keras (the machine learning library we will be using) can understand. The "as np" bit means we can refer to numpy in shorthand as "np". The "random" library allows us to easily create random numbers.

```python
# disable scientific notation (1e-4) in favor of decimals (.0001)
np.set_printoptions(suppress=True)
```
This handy function from numpy makes it so numbers aren't presented in scientific notation. I find it easier to quickly read .0001 as opposed to 1e-4, and find this to be true for most non-scientists. The "hashtag" on the first line of that segment allows one to insert comments into code.

```python
# need to initialize np arrays
train_data = np.array([[1.0,1.0]])
train_targets = np.array([2.0])
```

In order to create training data, we have to create a container within which to store it. In this case we will have an array (a collection) for the numbers we are adding, and separately an array for the sum of those numbers. Because we are adding two numbers together, the train_data array we create is 2-dimensional. You can think of it as being a spreadsheet with two columns. The sum array is 1-dimensional. So in this case, we have 1+1=2 stored in these arrays. More accurately, we have 1.0+1.0=2.0, as I want the network to be able to add floating point numbers. We will still be able to use it to add whole numbers, if we simply round the output of the network to the nearest whole number (integer).

```python
# experimenting: might be better to have fully random inputs
for i in range(10000):
  q = random.randint(-50000,50000)
  # axis=0 just means add the new data as rows, to maintain the 2-d structure
  train_data = np.append(train_data,[[i,q]],axis=0)
  train_targets = np.append(train_targets,[i+q])
  ```
  
  The above code may look somewhat complicated, but it's fairly simple when broken down. The for i in range line simply means to do the next segment of code 10,000 times. We need lots of training data, so the more sets of numbers we generate the better. 
  
  We have the variable "i" which goes from zero to 10,000. Each iteration of the loop increases "i" by one, and that number will be used as the first number in each group of two. "q" will be randomly generated between -50,000 and 50,000 so that we have lots of possible numbers to work with. So for the first loop, "i" will be zero, and q will be a random number, let's say 2,500. In that case we have [0, 2500] as the two numbers in that row of data. 
  
  The line "train_data = np.append(train_data,[[i,q]],axis=0)" simply combines these numbers into that format. "train_data" is the array that stores these numbers, so we want to append each new entry to it. Remember, think of this as a spreadsheet with two columns and 10,000 rows. The notation ```[[i,q]]``` and ```axis=0``` tells numpy to package the numbers as a single row. It might look something like this:
  
  ```
  0, 2500
  1, -500
  2, 48
  3, 9521
  ...
  10000, -521
  ```
  
  ```train_targets = np.append(train_targets,[i+q])``` appends the sum of these numbers to an array. So it would look something like this:
  
  ```
  2500
  -499
  50
  9524
  ...
  9479
  ```
  
  Now we do the same to generate some testing data. It's important to separate training data from testing data because we don't want to just ask the network about sums of numbers we've already given it. If you were making a NN to do classification on images to determine if they were of cats or dogs, you wouldn't want to give it a photo of a dog you used to train it, since it would already know the answer!
  
  ```python
# now generate some testing data
test_data = np.array([[2.0,3.0]])
test_targets = np.array([5.0])
for i in range(1000):
  q = random.randint(-50000,50000)
  test_data = np.append(test_data,[[i,q]],axis=0)
  test_targets = np.append(test_targets,[i+q])
  ```
  
  This is virtually identical to the code we used to generate training data. 
  
  ## Build the Neural Network architecture
  
  As with the above code, we first need to import some libraries
  
  ```python
import tensorflow as tf
from tensorflow import keras
```

The "tensorflow" and "keras" libraries are used in the creation of the actual neural network. Again, the "as tf" syntax just allows us to use shorthand to refer to tensorflow. Keras is part of tensorflow that gives us easy to use functions with which to create model architecture. In a nutshell, Keras is the "front end" software that we will use to construct the network, and tensorflow is the "back end" software that takes care of the heavy lifting mathematics involved in training the network. At one point these were entirely separate projects, but recently Keras has become an official part of Tensorflow.

```python
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(2,)),
    keras.layers.Dense(10, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.relu),
    keras.layers.Dense(1)
])
```

The above code creates the network architecture. That's it! What's happening here is that we've given the network 2 input parameters (the numbers we will add together), which are connected to another layer of 10 neurons, which are in turn connected to another layer of 10 neurons, which is connected to a final output layer of 1 single neuron (our sum number).

The network can be visaulized like this:
![[Screenshot_2020-09-14 NN SVG.png]]

(drawn with http://alexlenail.me/NN-SVG/index.html)

Going line by line, I'll explain a bit more about what's going on in the network code.

With ```model = keras.Sequential([``` we are creating a "sequential" neural network using Keras. A sequential model is one that simply moves from left to right, from input to output neurons, as with the diagram above. This is opposed to a functional model which can be much more complex and have links between non-sequential layers.

With ```keras.layers.Flatten(input_shape=(2,)),``` we specify the first layer of our network, the input layer. To "Flatten" in this case just means to make each neuron in the layer be its own row. This makes more sense when thinking about using images as input, which are grids of pixels; in order to put these through a network they must be "flattened" into a single column of data. The ```input_shape``` simply means there are two neurons as input.

With ```keras.layers.Dense(10, activation=tf.nn.relu),``` we specify our first "hidden layer" in the network. A hidden layer is one that is not exposed for data input or output from the user. It takes data from the previous layer and then feeds it forward after processing. Our two hidden layers are exactly the same. They are "Dense" layers which mean they are fully connected to the layer before. The activation function is where things get a bit tricky and require some math to explain. In order to keep this somewhat simple, I will say that when data is passed from one neuron to the other, it goes through an activation function to determine what that neuron should pass forward to the next neuron. A neuron might have an activation function that is essentially "only pass on the number you receive if it is greater than zero", for example. "ReLU" is a Rectified Linear Unit, and is the activation function of choice for most networks.

Finally, ```keras.layers.Dense(1)``` is our output layer of one single neuron, the sum. Though we do not specify an activation function, the default is "linear" which just passes along the value without a condition.

```python
model.compile(optimizer='adam', 
              loss='mse',
              metrics=['mae'])
```

Here we compile the model into a useable form. This is an example of Keras and Tensorflow doing a ton of work behind the scenes. That single line ```optimizer='adam'``` activates the "Adam" optimizer for our network. I won't get into the various optimizers, but it's important to know that the optimizer is what makes the network learn. At first, our network is just a bunch of connected neurons with random values. They can't produce meaningful output until they have been tuned to do so. The "loss" of a network is essentially how far-off a prediction is from reality. In other words, if we supply our network with 1 and 2 as the input, it should give us a 3. If it gives us 3.5, then the loss is .5 and the neurons should be tuned accordingly. 

Neural Networks use something called "back propogation" to tune the neurons (or more precisely, their associated weights and biases). So if the network knows it was off by .5, it works backward to change the neurons such that it would have given a more accurate answer for that input pair of 1 + 2. The Adam optimizer is simply a means of doing that efficiently, and as such it is the optimizer of choice for many networks.

The ```loss='mse'``` bit is a way of calculating the loss. In this case we are using mse or Mean Squared Error. This also means I lied when I said the loss of 1 + 2 = 3.5 is .5, since it will actually be the square of the distance between the prediction and actual values. This is useful in a few ways that I will gloss over in favor of getting into the weeds of mathematics, but one example of why this is a good idea is because you want large errors to be significantly (quadratically) more penalized than small ones. 

If the network thinks 1 + 2 = 3.5, the squared error is .25 (.5 squared) whereas if the network thinks 1 + 2 = 10, the difference is 7, or 49 as the error squared. It's the "mean" squared error because we average all the errors for an epoch of training, more on that in a sec.

```metrics=['mae']``` is a bit like the loss parameter, except metrics aren't used to optimize the network but rather for you, the researcher, to see how your model is performing. It might make more sense for the training process to more heavily penalize outliers, but in terms of understanding how well the network is performing, it makes more sense in this case to look at the mean average error, or "mae". If we had three predictions, such as:

```
1 + 2 = 3.5
5 + 5 = 10
10 + 20 = 40
```

The predictions were off by .5, 0, and 10, respectively. So the mean average error is 10.5/3 or 3.5.

Finally, ```model.fit(train_data, train_targets, epochs=10, batch_size=1)``` does the actually training of our network. This step will take a while, since it needs to run several simulations, use the loss from each attempt to tune the neurons, and repeat. An epoch is Machine Learning is when a model is trained on all available training data, once. For us, this means it goes through the initial 10,000 training samples and makes adjustments along the way to the network. We have ```epochs=10``` so it runs through the data 10 times. The ```train_data, train_targets``` code just means use the numbers we generated for the training. Remember, the train_data array has the number pairs to be added, and the train_targets array has their respective sums, in order. ```batch_size``` is more useful when considering images as a data source, since loading 10,000 images into memory might be too much. In that case, it would make sense to load the images in batches of 1,000 images at a time, for example.

```python
test_loss, test_acc = model.evaluate(test_data, test_targets)
print("\n", "Test accuracy:", test_acc, "\n")
```

The above code finds the accuracy of the model after training, using the metric we specified (mean average error). After running that code I got:

```Test accuracy: 0.15269197523593903```

This means that the average prediction was .15 off of the actual value. 1 + 2 might have come out as 3.15, for example.

## Use the network to make predictions

Now that we have built and trained the network, we can use it to make predictions.

```python
# use the model to make predictions
a = np.array([[1,2],[4999,1]])
print(model.predict(a))
```

We can feed an arbitrary number of values into the model. In this case we are feeding in two pairs: ```[1, 2]``` and ```[4999, 1]```. We should expect to get the numbers 3 and 5000 in return.

When I run this code I get:

```
[[   3.3818846]
 [5000.2295   ]]
 ```
 
 Not exactly correct! There are a few things to consider: 
 1. Neural Networks never give exact answers. They are predictions with levels of confidence. We knew that our model had an accuracy of plus or minus 0.15, which accounts for some of this discrepancy. 
 2. Anyone familiar with training neural networks knows I did something very wrong. Networks want all training and test data to be "normalized" in order to function properly. A normalized number is simply a number between -1 and 1. I skipped this step because it's easier to understand how things work conceptually without this step, and because the network works despite the numbers not being normalized. Since we generated numbers between 0 and 50,000 in this example, we would simply map the data from 0->50,000 to -1->1 and feed that into the network. If you do this, don't forget to re-scale the numbers after the network returns a result!
 3. We can train it more. 10 epochs is pretty good, but running it for 20 will give you better results. More training data also helps. Both of these solutions will add time to the learning process, so it's a trade-off. Also, training has a diminishing margin of return, meaning at a certain point putting it through more epochs won't make much of a difference at all. 
