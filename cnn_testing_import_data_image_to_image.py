import tensorflow as tf
import sys
import numpy as np
import glob
import os
import random

# Sources List:
# Read input from file
# 1) https://agray3.github.io/2016/11/29/Demystifying-Data-Input-to-TensorFlow-for-Deep-Learning.html
# 2) https://github.com/tensorflow/models/blob/master/research/inception/inception/data/build_image_data.py
# 3) https://github.com/affinelayer/pix2pix-tensorflow
# Neural Network Structure
# 4) https://software.intel.com/en-us/articles/an-example-of-a-convolutional-neural-network-for-image-super-resolution-tutorial
# CNN Implementation
# 5) http://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-tensorflow/
# 6) http://adventuresinmachinelearning.com/python-tensorflow-tutorial/

# Input images should be 2 (n*n) sized images next to each other. Total image size should be 2n*n
# [image_to_be_filtered][desired_output_image]

# Number of classes is 2 (squares and triangles)
nClass = 2

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
filenames = _find_image_files("C:\\Users\\Yola\\TensorFlowTest\\image_directory")
image_input=[]
image_output=[]
#For each image, get the image and process
#Save list of input images and output images
for i in filenames:
    print("i=",i)
    input, output = _process_image(i)
    image_input.append(input)
    image_output.append(output)

print("Input image size = ",tf.size(image_input))
print("Output image shape = ",tf.size(image_output))

# associate the "label" and "image" objects with the corresponding features read from
# a single example in the training data file
#label_input, image_input = getImage("data/train-00000-of-00001")
#label_output, image_output = getImage("data/train-00000-of-00001")

#Python optimisation variables
learning_rate = 0.0001
epochs = 10
batch_size = 100

#declare the training placeholders
x = tf.placeholder(tf.float32, [None, 28, 28, 1])
#now declare the output placeholder - 10 digits
y = tf.placeholder(tf.float32, [None, 28, 28, 1])

def create_new_conv_layer(input_data, num_input_channels, num_filters, filter_shape, pool_shape, name):
  #setup the filter input shape for tf.nn.conv_2d
  conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels, num_filters]
  #initialise the weights and bias for the filter
  weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03), name=name+'_W')
  bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')

  #setup the convolutional layer operation
  out_layer = tf.nn.conv2d(input_data, weights, [1,1,1,1], padding='SAME')

  #add the bias
  out_layer = tf.nn.relu(out_layer)

  #now perform max_pooling
  ksize = [1, pool_shape[0], pool_shape[1], 1]
  strides = [1,2,2,1]
  out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides, padding='SAME')

  return out_layer

#create some convolutional layers
layer1 = create_new_conv_layer(x, 1, 32, [5,5], [2,2], name='layer1')
layer2 = create_new_conv_layer(layer1, 32, 64, [5,5], [2,2], name='layer2')

flattened = tf.reshape(layer2, [-1, 7*7*64])

wd1 = tf.Variable(tf.truncated_normal([7*7*64, 1000], stddev=0.03), name='wd1')
bd1 = tf.Variable(tf.truncated_normal([1000], stddev=0.01), name='bd1')
dense_layer1 = tf.matmul(flattened, wd1) + bd1
dense_layer1 = tf.nn.relu(dense_layer1)

wd2 = tf.Variable(tf.truncated_normal([1000, nClass], stddev=0.03), name='wd2')
bd2 = tf.Variable(tf.truncated_normal([nClass], stddev=0.01), name='bd2')
dense_layer2 = tf.matmul(dense_layer1, wd2) + bd2
y_ = tf.nn.softmax(dense_layer2)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dense_layer2, labels=y))
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

#add an optimiser
optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

#define an accurate assessment operation
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('C:\\Users\\Yola\\TensorFlowTest\\import_data_test')

sess = tf.InteractiveSession()

#set up the initialisation operator
sess.run(tf.global_variables_initializer())

#I'm not entirely sure if I need this any more but we'll see
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess,coord=coord)

#Algorithm for computing mean squared error
#Comes from https://github.com/tensorflow/tensorflow/issues/1666
def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def psnr(accuracy):
    rmse = tf.sqrt(accuracy)
    final_accuracy = 20 * log10(255.0 / rmse)
    return final_accuracy

#initialise the variables
#total_batch = int(len(mnist.train.labels) / batch_size)
total_batch = 100
print("Batch size = ",total_batch)
for epoch in range(epochs):
    avg_cost = 0
    for i in range(total_batch):
      print("Batch = ",i)
      #vimageBatch, vlabelBatch = mnist.train.next_batch(batch_size=batch_size)
      batch_xs, batch_ys = sess.run([image_input, image_output])
      #print("Optimiser shape = ",tf.size(optimiser))
      _, c = sess.run([optimiser, cross_entropy], feed_dict={x:batch_xs, y:batch_ys})
      avg_cost += c / total_batch
    #test_acc = sess.run(accuracy, [imageBatch, labelBatch])
    test_acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys})
    print("Epoch:", (epoch+1), "cost =", "{:.3f}".format(avg_cost), "test_accuracy: {:.3f}".format(test_acc))
    summary_epoch = sess.run(merged, feed_dict={x: batch_xs, y: batch_ys})
    writer.add_summary(summary_epoch, epoch)

print("\nTraining complete!")
writer.add_graph(sess.graph)
print(sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys}))