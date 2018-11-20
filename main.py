#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import numpy as np
from timeit import default_timer as timer

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

num_classes = 3
epochs = 250
batch_size = 16
dropout_keep = 0.5
l2_scale = 1e-2
lr = 1e-4
target_loss = 0.01
loss_to_save = 0.015

def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    vgg_input = graph.get_tensor_by_name('image_input:0')
    vgg_keep_prob = graph.get_tensor_by_name('keep_prob:0')
    vgg_layer3_out = graph.get_tensor_by_name('layer3_out:0')
    vgg_layer4_out = graph.get_tensor_by_name('layer4_out:0')
    vgg_layer7_out = graph.get_tensor_by_name('layer7_out:0')
    return vgg_input, vgg_keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes, l2_regularizer_scale=l2_scale):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    layer8_1by1_out = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, strides=(1,1), padding='same',
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_regularizer_scale)) # 1x1
    layer9_out = tf.layers.conv2d_transpose(layer8_1by1_out, vgg_layer4_out.get_shape()[-1], 4, strides=(2,2), padding='same',
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_regularizer_scale)) #2x2
    print(vgg_layer4_out.get_shape(), " add to ", layer9_out.get_shape())
    layer10_in = tf.add(vgg_layer4_out, layer9_out) # skip kayer 4
    layer10_out = tf.layers.conv2d_transpose(layer10_in, vgg_layer3_out.get_shape()[-1], 4, strides=(2,2), padding='same',
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_regularizer_scale)) # 4x4
    print(vgg_layer3_out.get_shape(), " add to ", layer10_out.get_shape())
    layer11_in =  tf.add(vgg_layer3_out, layer10_out)
    layer11_out = tf.layers.conv2d_transpose(layer11_in, num_classes, 16, strides=(8,8), padding='same',
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_regularizer_scale), name='layer11_out') # 32 x 32

    return layer11_out

def optimize(nn_last_layer, correct_label, learning_rate):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    num_classes = nn_last_layer.get_shape()[-1]
    print("Num classes: ", num_classes, ", ", nn_last_layer.get_shape())
    logits = tf.reshape(nn_last_layer, (-1, num_classes), name='segmentation_logits')
    correct_label = tf.reshape(correct_label, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy_loss)
    return logits, train_op, cross_entropy_loss

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    :param l2_regularizer_scale: TF Placeholder for l2 regularizer scale
    """
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        total_loss = 0
        total_images = 0
        start = timer()
        for images, gt in get_batches_fn(batch_size):
            labels = np.reshape(gt, (-1, gt.shape[-1])).astype(np.float32)
            labels = (labels.T / labels.sum(1)).T # normalize each row, here we alow a pixel to be in more than one class, but cannot be in no class
            _, loss = sess.run([train_op, cross_entropy_loss], 
                                feed_dict={  
                                    input_image: images,
                                    correct_label: labels,
                                    learning_rate: lr,
                                    keep_prob: dropout_keep})
            total_loss += loss 
            total_images += 1
        end = timer()
        mean_loss = total_loss / total_images
        print("Epoch: ", epoch, ", time: ", end - start, ", total loss: ", total_loss, ", images: " , total_images, ", average loss: ", mean_loss)
        if mean_loss < loss_to_save:
            helper.save_model(sess, "models", "model{lr}".replace('.', '-'), epoch)
        if mean_loss < target_loss:
            return loss
    return loss

def run():
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape, num_classes)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Build NN using load_vgg, layers, and optimize function
        vgg_input, vgg_keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        #l2_regularizer_scale = tf.placeholder(tf.float32, name='l2_regularizer_scale')
        layer11_out = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes, l2_scale)

        labels = tf.placeholder(tf.bool, name='correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        logits, train_op, cross_entropy_loss = optimize(layer11_out, labels, learning_rate)

        # Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, vgg_input, labels, vgg_keep_prob, 
                                learning_rate)
    
        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, vgg_keep_prob, vgg_input, num_classes)
        # OPTIONAL: Apply the trained model to a video

#tests.test_load_vgg(load_vgg, tf)
#tests.test_layers(layers, num_classes)
#tests.test_optimize(optimize)
#tests.test_train_nn(train_nn, num_classes)

if __name__ == '__main__':
    run()
