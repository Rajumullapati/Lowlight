import tensorflow as tf

saver = tf.train.Saver()

with tf.Session() as session:
  saver.restore(session,"tensor_overnight.meta")