from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

import tensorflow as tf

slim = tf.contrib.slim
layers = tf.contrib.layers
arg_scope = tf.contrib.framework.arg_scope

FLAGS = tf.app.flags.FLAGS



class VAE(object):
  """Variational AutoEncoder
  """

  def __init__(self, mode):
    """Basic setup.
    """
    assert mode in ["train", "generate"]
    self.mode = mode

    self.batch_size = FLAGS.batch_size

    self.z_dim = FLAGS.z_dim    # number of dimension of the latent variable `z`
    self.h_dim = FLAGS.h_dim    # number of dimension of hidden layer

    print('The mode is %s.' % self.mode)
    print('complete initializing model.')


  def read_mnist_from_placeholder(self):
    # Setup the placeholder of data
    if self.mode == "train":
      with tf.variable_scope('read_mnist'):
        self.inputs = tf.placeholder(dtype=tf.float32,
                                     shape=[self.batch_size, 784],
                                     name='inputs')

    if self.mode == "generate":
      with tf.variable_scope('random_z'):
        self.random_z = tf.random_normal(shape=[self.batch_size, self.z_dim],
                                         name='latent_z')


  def Encoder(self, inputs):
    """Q(z|X) encoder

    Args:
      inputs: data X (ex. images)

    Returns:
      z_mu: Multi variate normal distribution parameters
      z_log_sigma: Multi variate normal distribution parameters
    """
    with tf.variable_scope('Encoder'):
      layer1 = layers.fully_connected(inputs=inputs,
                                      num_outputs=self.h_dim,
                                      scope='fc1')
      layer2 = layers.fully_connected(inputs=layer1,
                                      num_outputs=self.h_dim,
                                      scope='fc2')

      z_mu = layers.fully_connected(inputs=layer2,
                                    num_outputs=self.z_dim,
                                    activation_fn=None,
                                    scope='mu')
      z_log_sigma = layers.fully_connected(inputs=layer2,
                                           num_outputs=self.z_dim,
                                           activation_fn=None,
                                           scope='log_sigma')
      return z_mu, z_log_sigma


  def sampling_z(self, mu, log_sigma):
    """sampling z using reparmeterization trick

    Args:
      z_mu: Multi variate normal distribution parameters
      z_log_sigma: Multi variate normal distribution parameters

    Return:
      sampling z: sample z using reparmeterization trick
    """
    with tf.variable_scope('sampling_z'):
      epsilon = tf.random_normal(shape=tf.shape(mu))
      return mu + tf.exp(log_sigma / 2.) * epsilon


  def Decoder(self, inputs):
    """P(X|z) decoder

    Args:
      inputs: z (latent variable)

    Returns:
      logits: logits from P(X|z) for reconstruction loss
      probability: sigmoid activation of logits for generating images
    """
    with tf.variable_scope('Decoder') as scope:
      layer1 = layers.fully_connected(inputs=inputs,
                                      num_outputs=self.h_dim,
                                      scope='fc1')
      layer2 = layers.fully_connected(inputs=layer1,
                                      num_outputs=self.h_dim,
                                      scope='fc2')

      logits = layers.fully_connected(inputs=layer2,
                                      num_outputs=784,
                                      activation_fn=None,
                                      scope='logits')
      probability = tf.nn.sigmoid(logits)

      return logits, probability


  def vae_loss(self, inputs, logits, z_mu, z_log_sigma):
    """Calculate the VAE loss
    reconstruction loss + KL divergence loss
    for each dataset in minibatch.

    Args:
      inputs: data X
      logits: logits from P(X|z) for reconstruction loss
      z_mu: Multi variate normal distribution parameters
      z_log_sigma: Multi variate normal distribution parameters

    Returns:
      loss: reconstruction loss + KL divergence
    """
    with tf.variable_scope('VAE_loss'):
      # reconstruction loss
      loss_reconst = tf.reduce_sum(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                            logits=logits,
                            labels=inputs), 1)

      # KL Divergence D_KL( Q(z|X) || P(z|X) )
      loss_KL = 0.5 * tf.reduce_sum(
                          tf.exp(z_log_sigma) + z_mu**2 - 1. - z_log_sigma, 1)

      # VAE loss
      loss = tf.reduce_mean(loss_reconst + loss_KL)

      return loss



  def build(self):
    # read images from mnist
    self.read_mnist_from_placeholder()

    if self.mode == "generate":
      # generating images from Decoder via latent variable z
      _, self.X_samples = self.Decoder(self.random_z)

    if self.mode == "train":
      # Create the Encoder
      self.z_mu, self.z_log_sigma = self.Encoder(self.inputs)

      # Sampling z
      self.z_sample = self.sampling_z(self.z_mu, self.z_log_sigma)

      # Create the Decoder
      self.logits, _ = self.Decoder(self.z_sample)

      # Calculate the loss
      self.loss = self.vae_loss(self.inputs, self.logits, self.z_mu, self.z_log_sigma)

      # Print all training variables
      t_vars = tf.trainable_variables()
      for var in t_vars:
        print(var.name)


    print('complete model build.')

