'''
 a logistic Regression learning algorithm example using tensorflow libarary(MNIST data)
'''
form __future__ import print_function
import tensorflow as tf 

#Import Mnist data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/",one_hot=True)

#parameters
learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1

# Tf graph Input
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

# set model weights
w = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#Construct model
pred = tf.nn.softmax(tf.matmul(x,w) + b)

# mininmize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred)),reduction_indices =1)

# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

#Initilize the varibales
init = tf.global_variables_initializer()

#Start Training
with tf.Session as sess:
    sess.run(init)

    # Training Cycle
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            #Run optimization
            _,c = sess.run([optimizer,cost],feed_dict={x:batch_xs,y:batch_ys})
            avg_cost +=c/total_batch
        #display logs per epoch step
        if(epoch+1)%display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))   
    print("optimizer finished")

    # Test Model
    correct_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
    #calculate accuarcy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))         

