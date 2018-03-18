import tensorflow as tf
import numpy as np
import sys
import glob
import os
import random

from SSIM_redo import SSIM_calculate

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

#Python optimisation variables
learning_rate = 0.0001
epochs = 100
batch_size = 5

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
filenames = _find_image_files("C:\\Users\\Yola\\git\\AdvancedReading\\training_data")
image_input=[]
image_output=[]

#For each image, get the image and process
#Save list of input images and output images
counter = 0
for i in filenames:
    counter = counter + 1
    print("i=",counter)
    input, output = _process_image(i)
    image_input.append(input)
    image_output.append(output)

print("Input image size = ",tf.size(image_input))
print("Output image shape = ",tf.size(image_output))

print("Input length = ",len(image_input))
print("Output length = ",len(image_output))

image_input_batch=[]
image_output_batch=[]

image_input_test=[]
image_output_test=[]

origin_dir = "C:\\Users\\Yola\\TensorTest\\input_data\\"
for i in range(0,5):
    image_input_test.append(image_input[i])
    image_output_test.append(image_output[i])
    png = tf.image.encode_png(tf.cast((tf.reshape(image_output[i], [28, 28, 1])+1.)*127.5,tf.uint8))
    sess = tf.Session()
    _output_png = sess.run(png)
    input_filename = origin_dir+"input_"+str(i)+".png"
    open(input_filename, 'wb').write(_output_png)

def generate_batch():
    image_indices=random.sample(range(101),batch_size)
    for i in image_indices:
        print("Index = ",i)
        #print("Value at index = ",image_indices[i])
        image_input_batch.append(image_input[i])
        image_output_batch.append(image_output[i])

    print("Size of input batch = ",len(image_input_batch))
    print("Size of output batch = ",len(image_output_batch))

def create_new_conv_layer(input_data, num_input_channels, num_filters, filter_shape, name):
    #setup the filter input shape for tf.nn.conv_2d
    conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels, num_filters]

    #initialise weights and bias for the filter
    weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03), name=name+'_W')
    bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')

    #setup the convolutional layer operation
    out_layer = tf.nn.conv2d(input_data, weights, [1,1,1,1], padding='SAME')

    #add the bias
    out_layer += bias

    #apply a ReLU non-linear activation
    out_layer = tf.nn.relu(out_layer)

    #ksize = [1, pool_shape[0], pool_shape[1], 1]
    #strides = [1,2,2,1]
    #out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides, padding='SAME')

    return out_layer

#declare training data placeholders
#input x - for 28 x 28 pixels = 784 - this is the flattened image data that is drawn from mnist.train.nextbatch()

x = tf.placeholder(tf.float32, [None, 28, 28, 1])
#reshape the input so that it is a 4D tensor
#x_shaped = tf.reshape(x, [-1, 28, 28, 1])
#now declare the output data placeholder - 10 digits
    #this needs to be 784, what does None do?
y = tf.placeholder(tf.float32, [None, 28, 28, 1])

#create some convolutional layers
#layer1 = create_new_conv_layer(x_shaped, 1, 32, [5,5], [2,2], name='layer1')
#layer2 = create_new_conv_layer(layer1, 32, 64, [5,5], [2,2], name='layer2')

patch_extraction = create_new_conv_layer(x, 1, 64, [9, 9], name='patch_extraction')
non_linear_mapping = create_new_conv_layer(patch_extraction, 64, 32, [1, 1], name='non_linear_mapping')
reconstruction = create_new_conv_layer(non_linear_mapping, 32, 1, [5, 5], name='reconstruction')

print("Size of reconstruction = ",reconstruction)
y_ = tf.reshape(reconstruction, [-1, 28, 28, 1])
#y_ = tf.nn.relu(reconstruction)

cross_entropy = tf.reduce_sum(tf.square(y_ - y))
#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dense_layer2, labels=y))

#define an accurate assessment operation
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
print("Correct prediction = ",correct_prediction)
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

#y_argmax = tf.argmax(y,1)
#y__argmax = tf.argmax(y_,1)

accuracy_old = tf.reduce_mean(tf.square(tf.cast(correct_prediction, tf.float32)))
accuracy = psnr(accuracy_old)
#loss_calculator = SSIM_CLASS()
#add an optimiser

loss = SSIM_calculate(y, y_)
optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)


#y = tf.argmax(y, 1)
#y_ = tf.argmax(y_, 1)
#y = tf.reshape(y,[28, 28, 1])
#y_ = tf.reshape(y_, [28, 28, 1])
#accuracy = SSIM(tf.argmax(y,1), tf.argmax(y_,1))


#print("Shape of y = ",tf.shape(y))
#print("Shape of y_ = ",tf.shape(y_))
#accuracy = SSIM(y[0], y_[0])

#set up the initialisation operator
init_op = tf.global_variables_initializer()

#saver for model
saver = tf.train.Saver()

#set up recording variables
#add a summary to store the accuracy
tf.summary.scalar('Cross Entropy', cross_entropy)
tf.summary.scalar('SSIM', accuracy)
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('C:\\Users\\Yola\\TensorTest\\07-03')

#sess=tf.Session()
#sess.run(init_op)

sess = tf.InteractiveSession()

#set up the initialisation operator
sess.run(tf.global_variables_initializer())

#Round 2 for output images... We got this
#png = tf.image.encode_png(tf.cast((tf.reshape(y_, [28, 28, 1])+1.)*127.5,tf.uint8))
png = tf.image.encode_png(tf.cast((tf.reshape(y_[0], [28, 28, 1])+1.)*127.5,tf.uint8))
#data = tf.eye(256, batch_shape=[1])
#bgr = tf.stack([data, data, data], axis = 3)
#png = tf.image.encode_png(tf.cast((tf.reshape(bgr, [256, 256, 3])+1.)*127.5,tf.uint8))


#I'm not entirely sure if I need this any more but we'll see
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess,coord=coord)

#generate_batch()
total_batch = batch_size
for epoch in range(epochs):
    avg_cost = 0
    for i in range(total_batch):
        batch_xs, batch_ys = sess.run([image_input, image_output])
        _, c, acc = sess.run([optimiser, cross_entropy, accuracy], feed_dict={x: batch_xs, y: batch_ys})
        avg_cost += acc/total_batch
        print("Average cost = ",c)
    test_acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys})
    print("Epoch:", (epoch+1), "cost =", "{:.3f}".format(avg_cost), " test accuracy: {:.3f}".format(test_acc))
    #summary = sess.run(merged, feed_dict={x: mnist.validation.images, y: mnist.validation.images})
    summary = sess.run(merged, feed_dict={x:batch_xs, y: batch_ys})
    writer.add_summary(summary, epoch)
    #png_data_ = sess.run(png)
output_dir = "C:\\Users\\Yola\\TensorTest\\output_data\\"
for i in range(0,5):
    batch_xs_test, batch_ys_test = sess.run([image_input_test, image_output_test])
    _, acc, cross, _png_data = sess.run([optimiser, accuracy, cross_entropy, png], feed_dict={x: batch_xs_test, y: batch_ys_test})
    output_filename = output_dir + "output_" + str(i) + ".png"
    open(output_filename, 'wb').write(_png_data)



print("Training complete!")
#saver.save(sess, 'C:\\Users\\HWRacing\\AppData\\Local\\Programs\\Python\\Python36\\Scripts\\tensor_overnight')
#print("Finished saving")
writer.add_graph(sess.graph)
print(sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys}))
#prediction=tf.argmax(y_,1)
#print("Output = ")
#print(prediction.eval(feed_dict={x: mnist.validation.images}, session=sess))
