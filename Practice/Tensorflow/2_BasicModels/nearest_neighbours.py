'''
A nearest Neighbour learning algorithm example using tensorflow library using Mnist Dataset

'''

from __future__ import print_function

import numpy as np 
import tensorflow as tf  

#import Mnist data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/",one_hot = True)

# here we are limiting mnist data
Xtr ,Ytr = mnist.train.next_batch(5000)
Xte, Yte = mnist.train.next_batch(200)

#tf graph input
xtr = tf.placeholder("float",[None,784])
xte = tf.placeholder("float",[784])

# Nearest neighbour calculating using L1 distance
distance = tf.reduce_sum(tf.abs(tf.add(xtr,tf.negative(xte))), reduction_indices =1)
#prediction: get min distance index(Nearest Neighbour)
pred = tf.arg_min(distance,0)

accuracy =0

#Initialize the variables(i.e assign their default value)
init = tf.global_variables_initializer()

#Start Training
with tf.Session() as sess:
    sess.run(init)

    #Loop over the test data
    for i in range(len(Xte)):
        #get the nearest neighbour
        nn_index = sess.run(pred, feed_dict:{xtr:Xtr, xte:Xte[i,:]})
        # Get nearest neighbor class label and compare it to its true label
        print("test",i,"prediction",np.argmax(Ytr[nn_index]),"True class", i, np.argmax(Yte[i]))

        # Claculate accuracy
        if np.argmax(Ytr[nn_index] == np.argmax(Yte[i])):
            accuarcy += 1./len(Xte)
    print("Accuracy",accuarcy)
    print("done dana done!!!!!!!!!")        


