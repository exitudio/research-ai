import tensorflow as tf

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
c = tf.constant(6.0) # automatically assign type

adder_node = (a+b) * c

# 1. style : start session
sess = tf.Session()
output = sess.run(adder_node, {a: [1, 3], b: [2, 4]})
print('style1:', output)
writer = tf.summary.FileWriter('./graph', sess.graph)
sess.close()

# 2. style : start session
with tf.Session() as sess:
  output = sess.run(adder_node, {a: [1, 2], b:[2, 3]})
  print('style2:', output)


# show computational graph
# $tensorboard --logdir="./"