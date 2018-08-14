import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# set initial val and model
W=tf.Variable(tf.random_uniform([1],-1.0,1.0))
b=tf.Variable(tf.random_uniform([1],-1.0,1.0))

# W=tf.Variable(tf.zeros([1]))
# b=tf.Variable(tf.zeros([1]))

x=tf.placeholder(tf.float32) # input
y=tf.placeholder(tf.float32) # true val

model =W*x+b # guess val



# chose how to do
cost= tf.reduce_mean(tf.square(model-y))

opt=tf.train.GradientDescentOptimizer(learning_rate=0.05)

train=opt.minimize(cost)


# session def
train_tot=3
sess=tf.Session()
sess.run(tf.global_variables_initializer())

x_tr=[1,2,3]
y_tr=[1,2,3]

for i in range(train_tot):
    error,_=sess.run([cost,train],feed_dict={x:x_tr,y:y_tr})
    print(i,'error = %.3f' % error,'W=%.3f' % sess.run(W),'b=%.3f' % sess.run(b))


test = 5
guess =sess.run(model,feed_dict={x:test})
print('\ntest=',test,'guess=%.3f'%guess)