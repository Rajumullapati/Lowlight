import tensorflow as tf
import numpy as np
#import sys
#import glob
#import os
#import random

def SSIM(input_image, output_image):

  x = tf.placeholder(tf.int32, shape=[28,28,1])
  y = tf.placeholder(tf.int32, shape=[28,28,1])

  #input_image = tf.cast(,tf.int32)
  #output_image = tf.cast(tf.reshape(output_image, shape=[28,28,1]),tf.int32)

  input_image = tf.image.convert_image_dtype(tf.reshape(input_image, shape=[28,28,1]), dtype=tf.int32)
  output_image = tf.image.convert_image_dtype(tf.reshape(output_image, shape=[28,28,1]), dtype=tf.int32)

  print("Input image size = ",tf.size(input_image))
  print("Output image shape = ",tf.size(output_image))

  x = tf.identity(input_image)
  y = tf.identity(output_image)

  print("Input image size = ",tf.size(x))
  print("Output image shape = ",tf.size(y))

  mean_x, variance_x = tf.nn.moments(x, [0])
  mean_y, variance_y = tf.nn.moments(y, [0])

  c1 = (0.01*255)*(0.01*255)
  c2 = (0.03*255)*(0.03*255)

  x_y_covariance, x_y_optimiser = tf.contrib.metrics.streaming_covariance(x, y)
  x_x_covariance, x_x_optimiser = tf.contrib.metrics.streaming_covariance(x, x)
  y_y_covariance, y_y_optimiser = tf.contrib.metrics.streaming_covariance(y, y)

  with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    m_x, _ = sess.run([mean_x, variance_x])
    m_y, _ = sess.run([mean_y, variance_y])

    print("I GOT IN HERE")

    sess.run([x_y_optimiser])
    covariance_x_y = sess.run([x_y_covariance])
    sess.run([x_x_optimiser])
    variance_x = sess.run([x_x_covariance])
    sess.run([y_y_optimiser])
    variance_y = sess.run([y_y_covariance])

    numerator = np.multiply(sum(np.multiply(2,m_x,m_y),c1),(sum(np.multiply(2,covariance_x_y),c2)))
    denominator = np.multiply(sum(sum(np.multiply(m_x,m_x),np.multiply(m_y,m_y)),c1),sum(sum(variance_x,variance_y),c2))
    SSIM = numerator/denominator

    loss = 1 - SSIM

    return loss