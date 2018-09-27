'''

A 2-Hidden Layers Fully Connected Neural Network (a.k.a Multilayer Perceptron)
implementation with TensorFlow. This example is using the MNIST database
of handwritten digits (http://yann.lecun.com/exdb/mnist/).
This example is using TensorFlow layers, see 'neural_network_raw' example for
a raw implementation with variables.

'''

from __future__ import print_function

#Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data
