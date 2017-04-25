from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import variational_autoencoder as vae

slim = tf.contrib.slim


##################
# Training Flags #
##################
tf.app.flags.DEFINE_string('train_dir',
                           '',
                           'Directory where checkpoints and event logs are written to.')
tf.app.flags.DEFINE_integer('max_steps',
                            100000,
                            'The maximum number of training steps.')
tf.app.flags.DEFINE_integer('save_steps',
                            10000,
                            'The step per saving model.')

#################
# Dataset Flags #
#################
tf.app.flags.DEFINE_integer('batch_size',
                            64,
                            'The number of samples in each batch.')

########################
# Learning rate policy #
########################
tf.app.flags.DEFINE_float('initial_learning_rate',
                          0.001,
                          'Initial learning rate.')

#######################
# VAE network setting #
#######################
tf.app.flags.DEFINE_integer('z_dim',
                            100,
                            'The dimension of latent variable z.')
tf.app.flags.DEFINE_integer('h_dim',
                            128,
                            'The dimension of hidden layer.')

FLAGS = tf.app.flags.FLAGS


def main(_):

  if tf.gfile.Exists(FLAGS.train_dir):
    raise ValueError('This folder already exists.')
  tf.gfile.MakeDirs(FLAGS.train_dir)

  with tf.Graph().as_default():

    # Build the model.
    model = vae.VAE(mode="train")
    model.build()

    # Create global step
    global_step = slim.create_global_step()

    # No decay learning rate
    learning_rate = tf.constant(FLAGS.initial_learning_rate)
    tf.summary.scalar('learning_rate', learning_rate)

    # Create an optimizer that performs gradient descent for Discriminator.
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # Minimize optimizer
    opt_op = opt.minimize(model.loss,
                          global_step=global_step)


    # Set up the Saver for saving and restoring model checkpoints.
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1000)


    # Start running operations on the Graph.
    with tf.Session() as sess:
      # Build an initialization operation to run below.
      init = tf.global_variables_initializer()
      sess.run(init)

      # Start the queue runners.
      tf.train.start_queue_runners(sess=sess)

      # Create a summary writer, add the 'graph' to the event file.
      summary_writer = tf.summary.FileWriter(
                          FLAGS.train_dir,
                          sess.graph)

      # Retain the summaries and build the summary operation
      summary_op = tf.summary.merge_all()


      # Read MNIST data
      mnist = input_data.read_data_sets("data/MNIST_data/", one_hot=True)

      for step in range(FLAGS.max_steps+1):
        start_time = time.time()

        batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
        feed_dict = {model.inputs: batch_xs}

        _, loss = sess.run([opt_op,
                            model.loss],
                            feed_dict=feed_dict)

        epochs = step * FLAGS.batch_size / mnist.train.num_examples
        duration = time.time() - start_time

        if step % 10 == 0:
          examples_per_sec = FLAGS.batch_size / float(duration)
          print("Epochs: %.2f step: %d  loss: %f (%.1f examples/sec; %.3f sec/batch)"
                    % (epochs, step, loss, examples_per_sec, duration))

        if step % 200 == 0:
          summary_str = sess.run(summary_op, feed_dict=feed_dict)
          summary_writer.add_summary(summary_str, step)

        # Save the model checkpoint periodically.
        if step % FLAGS.save_steps == 0:
          checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
          saver.save(sess, checkpoint_path, global_step=step)

    print('complete training...')



if __name__ == '__main__':
  tf.app.run()
