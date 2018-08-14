import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# set initial val and model
W=tf.Variable(tf.random_uniform([1],-1.0,1.0))
b=tf.Variable(tf.random_uniform([1],-1.0,1.0))

# W=tf.Variable(tf.zeros([1]))
# b=tf.Variable(tf.zeros([1]))


# W=tf.Variable(0.5)
# b=tf.Variable(-0.5)

x=tf.placeholder(tf.float32) # input
y=tf.placeholder(tf.float32) # true val

model =W*x+b # guess val



# chose how to do
cost= tf.reduce_mean(tf.square(model-y))

opt=tf.train.GradientDescentOptimizer(learning_rate=0.0067)
# if sort MAX is 67E-4, limit err=0.272 in 1000, guess 14.54 W=0.499 b=-0.445
# unsort MAX is 73E-4, limit err=0.451 in 2000, guess -5.016 W=-0.493 b=9.78
train=opt.minimize(cost)


# session def
train_tot=10000
sess=tf.Session()
sess.run(tf.global_variables_initializer())

x_tr=[10,15,16,1,4,6,18,12,14,7]
y_tr=[5,2,1,9,7,8,1,5,3,6]
x_tr.sort()
y_tr.sort()

for i in range(train_tot):
    error,_=sess.run([cost,train],feed_dict={x:x_tr,y:y_tr})
    if (i%100) ==0:
        print(i,'error = %.3f' % error,'W=%.3f' % sess.run(W),'b=%.3f' % sess.run(b))
print(i,'error = %.3f' % error,'W=%.3f' % sess.run(W),'b=%.3f' % sess.run(b))
test = 30
guess =sess.run(model,feed_dict={x:test})
print('\ntest=',test,'guess=%.3f'%guess)