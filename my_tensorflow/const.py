import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

print(2+3)
import tensorflow as tf
print('using tf')
a=tf.constant(2)
b=tf.constant(3)
x=tf.add(a,b)
with tf.Session() as sess:
    print(sess.run(x))
# print(x)

a=tf.constant([2,2])
b=tf.constant([[0,1],[2,3]])
x=tf.add(a,b)
y=tf.multiply(a,b)
with tf.Session() as sess:
    x,y=sess.run([x,y])
    print('x:',x)
    print('y:',y)


a=tf.constant([[2,2]])
b=tf.constant([[0,1],[2,3]])
z=tf.matmul(a,b)
with tf.Session() as sess:
    z=sess.run(z)
    print('z:',z)


