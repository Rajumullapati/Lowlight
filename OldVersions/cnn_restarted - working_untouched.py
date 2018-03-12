import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def run_cnn():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    #Python optimisation variables
    learning_rate = 0.0001
    epochs = 10
    batch_size = 50

    #declare training data placeholders
    #input x - for 28 x 28 pixels = 784 - this is the flattened image data that is drawn from mnist.train.nextbatch()

    x = tf.placeholder(tf.float32, [None, 784])
    #reshape the input so that it is a 4D tensor
    x_shaped = tf.reshape(x, [-1, 28, 28, 1])
    #now declare the output data placeholder - 10 digits
        #this needs to be 784, what does None do?
    y = tf.placeholder(tf.float32, [None, 784])

    #create some convolutional layers
    #layer1 = create_new_conv_layer(x_shaped, 1, 32, [5,5], [2,2], name='layer1')
    #layer2 = create_new_conv_layer(layer1, 32, 64, [5,5], [2,2], name='layer2')

    patch_extraction = create_new_conv_layer(x_shaped, 1, 64, [9, 9], name='patch_extraction')
    non_linear_mapping = create_new_conv_layer(patch_extraction, 64, 32, [1, 1], name='non_linear_mapping')
    reconstruction = create_new_conv_layer(non_linear_mapping, 32, 1, [5, 5], name='reconstruction')

    print("Size of reconstruction = ",reconstruction)
    y_ = tf.reshape(reconstruction, [-1, 784])
    #y_ = tf.nn.relu(reconstruction)

    cross_entropy = tf.reduce_sum(tf.square(y_ - y))
    #cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dense_layer2, labels=y))

    #add an optimiser
    optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

    #define an accurate assessment operation
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #set up the initialisation operator
    init_op = tf.global_variables_initializer()

    #saver for model
    saver = tf.train.Saver()

    #set up recording variables
    #add a summary to store the accuracy
    tf.summary.scalar('accuracy', accuracy)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('C:\\Users\\HWRacing\\TensorTest\\tensor_monday')
    with tf.Session() as sess:
        #initialise variables
        sess.run(init_op)
        total_batch = int(len(mnist.validation.images) / batch_size)
        print("Total batch = ",total_batch)
        for epoch in range(epochs):
            avg_cost = 0
            for i in range(total_batch):
                #print("Batch = ",i)
                batch_x, batch_y = mnist.validation.next_batch(batch_size=batch_size)
                _, c = sess.run([optimiser, cross_entropy], feed_dict={x: batch_x, y: batch_x})
                avg_cost += c/total_batch
                #print("Average cost = ",avg_cost)
            test_acc = sess.run(accuracy, feed_dict={x: mnist.validation.images, y: mnist.validation.images})
            print("Epoch:", (epoch+1), "cost =", "{:.3f}".format(avg_cost), " test accuracy: {:.3f}".format(test_acc))
            summary = sess.run(merged, feed_dict={x: mnist.validation.images, y: mnist.validation.images})
            writer.add_summary(summary, epoch)

        print("Training complete!")
        saver.save(sess, 'C:\\Users\\HWRacing\\TensorTest\\tensor_monday\\my_test_model')
        print("Finished saving")
        writer.add_graph(sess.graph)
        print(sess.run(accuracy, feed_dict={x: mnist.validation.images, y: mnist.validation.images}))

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

if __name__ == "__main__":
    run_cnn()

