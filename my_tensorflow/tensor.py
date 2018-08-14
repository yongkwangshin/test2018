# B1 import
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# B2 initial setting
x=tf.placeholder(tf.float32,[None,3])

# B3
# W=tf.Variable([[0.1,0.1],[0.1,0.1],[0.1,0.1]])
# b=tf.Variable([0.2,0.2])

# B4
# random_normal default:mean=0.0 and stddev=1.0
W=tf.Variable(tf.random_normal([3,2]))
b=tf.Variable(tf.random_normal([2,1]))

# B5 set output
output = tf.matmul(x,W)+b

# B6 instead with usd Session
sess=tf.Session()
sess.run(tf.global_variables_initializer())
input=[[1,2,3],[4,5,6]]

print("input :",input)
print("W :",sess.run(W))
print("b :",sess.run(b))
print("output:",sess.run(output,feed_dict={x:input}))

