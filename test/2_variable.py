# https://www.youtube.com/watch?v=yX8KuPZCAMo&t=2496s
import tensorflow as tf

W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)

# Inputs and Outputs
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

linear_model = W * x + b


# Loss
sqaured_delta = tf.square(linear_model - y)
loss = tf.reduce_sum(sqaured_delta)

# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.02)
train = optimizer.minimize(loss)

# init all variables
init = tf.global_variables_initializer()

# dataset
dataset = {x:[1,2,3,4], y:[0,-1,-2,-3]}

with tf.Session() as sess:
  sess.run(init)
  for i in range(1000):
    sess.run(train, dataset)
    if i%20==0:
      print( 'loss',sess.run(loss, dataset))
  print(sess.run([W, b]))
  # result = sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]})
  # print(result)