'''
Basic Operations example using TensorFlow library.

'''
from __future__ import print_function
 
import tensorflow as tf 

# Basic constant operations
# the value returned by the constructor represents the output
# of the constant op.

a = tf.constant(2)
b = tf.constant(3)

# Launch the default graph
with tf.Session() as sess:
    print("a=2, b=3")
    print("addition with constant: %i" %sess.run(a+b))
    print("Multiplication with constant: %i", %sess.run(a*b))

# Basic operations with variable as graph input
# the value returned by the constructor represents the output 
# of the variable op.(define as input when running session)
#  tf Graph input
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

# Define some operations
add = tf.add(a,b)
multiply = tf.multiply(a,b)

# Launch the default graph
with tf.Session as sess:
    print("Addition Of two numbers %i" %sess.run(add,feed_dict{a:2,b:3}))
    print("Multiplication of two numbers %i"%sess.run(multiply,feed_dict{a=4,b=8}))


# Matrix multiplication from tensorflow

# Create a constant op that produce a 1*2 matrix. the op is added as the default graph
matrix1 = tf.constant([[3.,3.]])

# Create another contant that produces 2*1 Matrix
matrix2 = tf.constant([[2.],[2.]])

#create a matmul operation that multiply two matrix 
product = tf.matmul(matrix1,matrix2)

# to run the matmul op we call the session "run()" method, passing 'product'
# which represnts the output of the matmul op. this indicates to the call that
# we want ti get the output of the matmul op back
# 
# All inputs needed by the op are run automatically by the session.they typically 
# are run in parallel.
# 
# the call 'run(product)' thus leds to the execution of three ops in the graph: the two constants 
# and matmul  and output of op is returned in 'result' as a numpy 'ndarray' object.

with tf.Session() as sess:
    result = sess.run(product)
    print(result)