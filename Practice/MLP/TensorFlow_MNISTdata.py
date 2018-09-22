''' MultiLayer Perceptron

A Multilayer Perceptron (Neural Network) implementation example using Tensorflow Library.This example is using the MNIST database of handwritten digits

Links: [MNIST Dataset](https://yann.lecun.com/exdb/mnist/).

Author : Suman Sourav

'''
#------------------------------------------
#                 FUN Begin Here
#------------------------------------------

from __future__ import print_function

# Import MNIST Data

from Tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot= True)

import tensorflow as tf

#Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

#Network parameters

n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
n_input = 784 # MNIST data input
n_classes = 10 # MNIST total classes(0-9 digits)

#tf Graph input

X = tf.placeholder("float",[None, n_input])
Y = tf.placeholder("float",[None, n_classes])

#Store layers weight and biases

weights = {
    'h1' : tf.Variable(tf.random_normal([n_input,n_hidden_1])),
    'h2' : tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3' : tf.Variable(tf.random_normal([n_hidden_2,n_classes]))
}

biases ={
    'b1' : tf.Variable(tf.random_normal([n_hidden_1])),
    'b2' : tf.Variable(tf.random_normal([n_hidden_2])),
    'b3' : tf.Variable(tf.random_normal([n_classes]))
}

#Create Model

def multilayer_perceptron(x):
    # Hidden Fully Connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_2 = tf.add(tf.mattmul(layer_1, weights['h2'], biases['b2']))
    out_layer = tf.add(tf.matmul(layer_2, weights['h3'], biases['b3']))
    
    return out_layer


# Construct Model
 logits = multilayer_perceptron(X)
 
# Define loss and Optimizer
 loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = Y))
optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
train_op = optimizer.minimize(loss_op)

# Initializing Varibales

init = tf.global_varibales_initializer()

with tf.Session() as sess:
    sess.run(init)

    # Training Cycle

    for epoch in range(training_epochs):
        avg_cost = 0 
        total_batch = int(mnist.train.num_examples/batch_size)
        #Loop over all the batches
        for i in range(total_batch):
            batch_x,batch_y = mnist.train.next_batch(batch_size)
            #Run optimization op(backprop) and cost op (to get loss value)
            _, c = sess.run([train_op, loss_op], feed_dict ={X:batch_x,Y:batch_y})
        
            # Compute average loss

            avg_cost += c/total_batch

        # Display loss per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost = {:.9f}".format(avg_cost))
    print("Optimzation Finished YEHHHHHH")

    # TEST MODEL

    pred = tf.nn.softmax(logits)  # apply sotfmax to logits
    correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(Y,1))

    #Calculate Accuracy

    accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))

     print("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))

