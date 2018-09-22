'''
Basic Introduction to tensorflow's Eager API

What is Eager API?
" Eager execution is an imperative, define-by-run interface where operations are
executed immediately as they are called from Python. This makes it easier to
get started with TensorFlow, and can make research and development more
intuitive. A vast majority of the TensorFlow API remains the same whether eager
execution is enabled or not. As a result, the exact same code that constructs
TensorFlow graphs (e.g. using the layers API) can be executed imperatively
by using eager execution. Conversely, most models written with Eager enabled
can be converted to a graph that can be further optimized and/or extracted
for deployment in production without changing code. " - Rajat Monga

'''
from __future__ import absolute_import,division, print_function

import numpy as np 
import tensorflow as tf 
import tensorflow.contrib.eager as tfe 

#set eager API
print("Setting Eager mode...")
tfe.enable_eager_execution() 

#Define constants tensors
print("Define constant tensors")
a = tf.constant(2)
print("a=%i"%a)
b = tf.constant(3)
print("b =%i" %b)

# Run the session without the need of tf.session
print("Running Operations, without tf.session")

c = a+b
print("a+b =%i"%c)

#Full Compatability with numpy
print("Mixing Operations with tensors and Numpy Arrays")

#Define constant tensors
a = tf.constant([[2.,1.],
                [1.,0.]],dtype=tf.float32)
print("Tensor:\n a=%s"%a)
b = np.array([[3.,0.],
                [5.,1.]],dtype=np.float32)
print("Numpy Array:\n b=%s"%b)                
c = a+b 
d = tf.matmul(a,b)
print("c = %s"%c)
print("d = %s"%c)

