from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import time
import copy
import cv2

import tensorflow as tf

import variational_autoencoder as vae

slim = tf.contrib.slim


####################
# Generating Flags #
####################
tf.app.flags.DEFINE_string('checkpoint_path',
                           '',
                           'Directory where to read model checkpoints.')
tf.app.flags.DEFINE_integer('checkpoint_step',
                            -1,
                            'The step you want to read model checkpoints.'
                            '-1 means the latest model checkpoints.')
tf.app.flags.DEFINE_integer('batch_size',
                            32,
                            'The number of samples in each batch.')
tf.app.flags.DEFINE_integer('seed',
                            0,
                            'The seed number.')
tf.app.flags.DEFINE_boolean('make_gif',
                            False,
                            'Whether make gif or not.')
tf.app.flags.DEFINE_integer('save_steps',
                            5000,
                            'The step per saving model.')

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



def make_squared_image(generated_images):
  N = len(generated_images)
  black_image = np.zeros(generated_images[0].shape, dtype=np.int32)
  w = int(np.minimum(10, np.sqrt(N)))
  h = int(np.ceil(N / w))

  one_row_image = generated_images[0]
  for j in range(1, w):
    one_row_image = np.concatenate((one_row_image, generated_images[j]), axis=1)
  
  image = one_row_image
  for i in range(1, h):
    one_row_image = generated_images[i*w]
    for j in range(1, w):
      try:
        one_row_image = np.concatenate((one_row_image, generated_images[i*w + j]), axis=1)
      except:
        one_row_image = np.concatenate((one_row_image, black_image), axis=1)
    image = np.concatenate((image, one_row_image), axis=0)

  return image



def ImageWrite(image, step):
  r,g,b = cv2.split(image)
  image = cv2.merge([b,g,r])

  filename = 'generated_images_%06.d.jpg' % step
  cv2.imwrite(filename, image)



def GIFWrite(generated_gifs, duration=4):
  for i, image in enumerate(generated_gifs):
    ImageWrite(image, i*FLAGS.save_steps)
  



def run_generator_once(saver, checkpoint_path, model):
  print(checkpoint_path)
  start_time = time.time()
  with tf.Session() as sess:
    tf.logging.info("Loading model from checkpoint: %s", checkpoint_path)
    saver.restore(sess, checkpoint_path)
    tf.logging.info("Successfully loaded checkpoint: %s",
                    os.path.basename(checkpoint_path))

    gray_1d = sess.run(model.X_samples)
    gray_1d *= 255.
    gray_2d = np.reshape(gray_1d, [FLAGS.batch_size, 28, 28])
    color_2d = []
    for i, gray in enumerate(gray_2d):
      color_image = cv2.merge([gray, gray, gray])
      color_image = np.reshape(color_image, [1, 28, 28, 3])
      if i == 0:
        color_2d = color_image
      else:
        color_2d = np.concatenate((color_2d, color_image), axis=0)

    generated_images = color_2d
    print(generated_images.shape)

    duration = time.time() - start_time
    print("Loading time: %.3f" % duration)

  return generated_images




def main(_):
  if not FLAGS.checkpoint_path:
    raise ValueError('You must supply the checkpoint_path with --checkpoint_path')


  with tf.Graph().as_default():
    start_time = time.time()

    # Build the generative model.
    model = vae.VAE(mode="generate")
    model.build()
    
    # Set up the Saver for saving and restoring model checkpoints.
    saver = tf.train.Saver()

    if not FLAGS.make_gif:
      if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
        if FLAGS.checkpoint_step == -1:
          checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
          checkpoint_step = int(checkpoint_path.split('-')[1])
        else:
          checkpoint_path = os.path.join(FLAGS.checkpoint_path, 'model.ckpt-%d' % FLAGS.checkpoint_step)
          checkpoint_step = FLAGS.checkpoint_step

        if os.path.basename(checkpoint_path) + '.data-00000-of-00001' in os.listdir(FLAGS.checkpoint_path):
          print(os.path.basename(checkpoint_path))
        else:
          raise ValueError("No checkpoint file found in: %s" % checkpoint_path)
      else:
        raise ValueError("checkpoint_path must be folder path")

      generated_images = run_generator_once(saver, checkpoint_path, model)
      squared_images = make_squared_image(generated_images)

      ImageWrite(squared_images, checkpoint_step)

    else:
      # Find all checkpoint_path
      if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
        checkpoint_filenames = []
        for filename in os.listdir(FLAGS.checkpoint_path):
          if '.data-00000-of-00001' in filename:
            filename = filename.split(".")[1].split("ckpt-")[1]
            checkpoint_filenames.append(filename)
      else:
        raise ValueError("checkpoint_path must be folder path")

      checkpoint_filenames.sort(key=int)
      for i, filename in enumerate(checkpoint_filenames):
        filename = 'model.ckpt-' + filename
        checkpoint_filenames[i] = filename

      generated_gifs = []
      for checkpoint_path in checkpoint_filenames:
        checkpoint_path = os.path.join(FLAGS.checkpoint_path, checkpoint_path)
        generated_images = run_generator_once(saver, checkpoint_path, model)
        squared_images = make_squared_image(generated_images)
        generated_gifs.append(squared_images)

      GIFWrite(generated_gifs)


    print('complete generating image...')




if __name__ == '__main__':
  tf.app.run()
