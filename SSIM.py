import tensorflow as tf
import numpy as np
import sys
import glob
import os
import random

class ImageCoder(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Create a single Session to run all image coding calls.
    self._sess = tf.Session()

    # Initializes function that converts PNG to JPEG data.
    self._png_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_png(self._png_data, channels=3)
    self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def png_to_jpeg(self, image_data):
    return self._sess.run(self._png_to_jpeg,
                          feed_dict={self._png_data: image_data})

  def decode_jpeg(self, image_data):
    image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

def _find_image_files(data_dir):
  """Build a list of all images files and labels in the data set.
  Args:
    data_dir: string, path to the root directory of images.
    Assume images are all .png
  Returns:
    filenames: list of strings; each string is a path to an image file.
  """

  #construct the list of filenames
  filenames = glob.glob(os.path.join(data_dir, "*.png"))

  # Shuffle the ordering of all image files in order to guarantee
  # random ordering of the images with respect to label in the
  # saved TFRecord files. Make the randomization repeatable.
  shuffled_index = list(range(len(filenames)))
  random.seed(12345)
  random.shuffle(shuffled_index)

  filenames = [filenames[i] for i in shuffled_index]
  return filenames

def _process_image(filename):
  """Process a single image file.
  Args:
    filename: string, path to an image file e.g., '/path/to/example.JPG'.
  Returns:
    input_image_return: image to be filtered
    output_image_return: desired image output
  """
  # Read the image file.
  with tf.gfile.FastGFile(filename, 'rb') as f:
    image_data = f.read()

  coder = ImageCoder()
  # Convert any PNG files to JPEG files
  image_data = coder.png_to_jpeg(image_data)

  # Decode the RGB JPEG.
  image = coder.decode_jpeg(image_data)
  #take from 3 channels to 1 (greyscale) (because processing time)
  image = tf.image.rgb_to_grayscale(image)

  # Convert to Tensor
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  # Set shape into a form used in next step
  image.set_shape([None, None, 1])

  # Split complete image into 2 images. For an explanation see source 3)
  width = tf.shape(image)[1]  # [height, width, channels]
  input_image_return = ((image[:, :width // 2, :])*2-1)
  output_image_return = ((image[:, width // 2:, :])*2-1)

  return input_image_return, output_image_return

#To do: make this as passing an argument, but that is 0% a priority
filenames = _find_image_files("C:\\Users\\HWRacing\\AppData\\Local\\Programs\\Python\\Python36\\Scripts\\tensor_test\\training_data_1")

#For each image, get the image and process
#Save list of input images and output images
x = tf.placeholder(tf.int32, shape=[28, 28, 1])
y = tf.placeholder(tf.int32, shape=[28, 28, 1])

for i in filenames:
    input, output = _process_image(i)

x = tf.identity(input)
y = tf.identity(output)

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

  sess.run([x_y_optimiser])
  covariance_x_y = sess.run([x_y_covariance])
  sess.run([x_x_optimiser])
  variance_x = sess.run([x_x_covariance])
  sess.run([y_y_optimiser])
  variance_y = sess.run([y_y_covariance])

  #SSIM = (2*m_x*m_y+c1)*(2*covariance_x_y+c2)/((tf.square(m_x)+tf.square(m_y)+c1)*(v_x + v_y + c2))
  #SSIM = 2*m_x*m_y*c1
  #SSIM = (2*m_x*m_y+c1)#*(2*covariance_x_y+c2)
  #SSIM = (2*covariance_x_y+c2)
  numerator = np.multiply(sum(np.multiply(2,m_x,m_y),c1),(sum(np.multiply(2,covariance_x_y),c2)))
  denominator = np.multiply(sum(sum(np.multiply(m_x,m_x),np.multiply(m_y,m_y)),c1),sum(sum(variance_x,variance_y),c2))
  SSIM = numerator/denominator

  loss = 1 - SSIM

  return loss
  print("numerator = "+str(numerator))
  print("denominator = "+str(denominator))
  print("SSIM = "+str(SSIM))
  print("loss = "+str(loss))


