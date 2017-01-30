import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#import mnist
mnist = input_data.read_data_sets("../MNIST/", one_hot=True)

#placeholder of shape [any length, 784] (mnist images have 784 pixels)
x = tf.placeholder(tf.float32, [None, 784])

#variables initialized at zero
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#model x*W+b instead W*x+b because it is a trick to deal with x being a 2D tensor
y = tf.nn.softmax(tf.matmul(x, W) + b)

#loss (the first cross_entropy is unstable)
y_ = tf.placeholder(tf.float32, [None, 10])
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

#training, minimize loss with gradient descent (tensorflow automatically use backpropagation)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#initialize variables and launch model in a session
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

#launch the train, stochastic training (batch of 100 points)
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

