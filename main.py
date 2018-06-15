# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import random
import os
from scipy.misc import imsave

from model import VAE
from data_manager import many_example_random_pair, single_factor_changed_one_example_generated_pair, DataManager

tf.app.flags.DEFINE_integer("epoch_size", 2000, "epoch size")
tf.app.flags.DEFINE_integer("batch_size", 64, "batch size")
tf.app.flags.DEFINE_float("gamma", 100.0, "gamma param for latent loss")
tf.app.flags.DEFINE_float("capacity_limit", 20.0,
        "encoding capacity limit param for latent loss")
tf.app.flags.DEFINE_integer("capacity_change_duration", 100000,
        "encoding capacity change duration")
tf.app.flags.DEFINE_float("learning_rate", 5e-4, "learning rate")
tf.app.flags.DEFINE_string("checkpoint_dir", "checkpoints", "checkpoint directory")
tf.app.flags.DEFINE_string("log_file", "./log", "log file directory")
tf.app.flags.DEFINE_boolean("training", True, "training or not")
tf.app.flags.DEFINE_integer("numlatent", 10, "number of latent variable batchse")
tf.app.flags.DEFINE_boolean("latentgen", True, "generating latent variables or not")
tf.app.flags.DEFINE_string("mode", "betavae", "for switching between VAE and BetaVAE")
tf.app.flags.DEFINE_integer("z_dim", 10, "Latent dimension")
tf.app.flags.DEFINE_string("reconstr", "reconstr_img", "Directory for storing reconstructed images")
tf.app.flags.DEFINE_float("perturbration", 1e-1, "size of pertubration for performing axis walks")

flags = tf.app.flags.FLAGS

def train(sess,
        model,
        manager,
        saver):
  print("Strating training")
  summary_writer = tf.summary.FileWriter(flags.log_file, sess.graph)

  n_samples = manager.sample_size

  reconstruct_check_images = manager.get_random_images(10)

  indices = list(range(n_samples))

  step = 0
  model.init_run(sess)
  # Training cycle
  for epoch in range(flags.epoch_size):
      # Shuffle image indices
    random.shuffle(indices)
    print("Running epoch : " + str(epoch))
    avg_cost = 0.0
    total_batch = n_samples // flags.batch_size

    # Loop over all batches
    for i in range(total_batch):
        # Generate image batch
      batch_indices = indices[flags.batch_size*i : flags.batch_size*(i+1)]
      batch_xs = manager.get_images(batch_indices)

      # Fit training using batch data
      reconstr_loss, latent_loss, summary_str = model.partial_fit(sess, batch_xs, step)
      summary_writer.add_summary(summary_str, step)
      step += 1

    # Disentangle check
    # disentangle_check(sess, model, manager)

    # Save checkpoint
    saver.save(sess, flags.checkpoint_dir + '/' + 'checkpoint', global_step = step)


def autoencoded_sample_save(sess, model, manager, flags, directory="default", num_samples=6400):
  n_samples = manager.sample_size

  indices = list(range(n_samples))
  batch_size = flags.batch_size
  step = 0
  model.init_run(sess)
  random.shuffle(indices)
  total_batch = num_samples // flags.batch_size
  x_tot = np.zeros([num_samples,64,64,2])
  for i in range(total_batch):
    images, _ = manager.sample_batch(flags.batch_size, train=False)
    x_reconstruct = np.concatenate([model.reconstruct(sess, images[:,:,:,0].reshape([flags.batch_size, 4096])), model.reconstruct(sess, images[:,:,:,1].reshape([flags.batch_size, 4096]))], axis=-1)
    x_tot[(i*batch_size):((i+1)*batch_size)] = x_reconstruct

  if not os.path.exists(flags.reconstr):
      os.mkdir(flags.reconstr)
  if not os.path.exists(os.path.join(flags.reconstr, directory)):
      os.mkdir(os.path.join(flags.reconstr, directory))

  np.savez_compressed("{1}/{0}/sample".format(step, directory, flags.reconstr),x_tot)

def axis_walk_save(sess, model, images, step=0, directory="default"):
  x_reconstruct = np.array(model.axis_walk(sess, images))
  print(x_reconstruct.shape)
  x_reconstruct = x_reconstruct.reshape([model.K, model.z_dim, images.shape[0],64,64])
  print(x_reconstruct.shape)
  x_reconstruct = np.swapaxes(x_reconstruct, 0,1)
  x_reconstruct = np.swapaxes(x_reconstruct, 1,2)
  x_reconstruct = np.swapaxes(x_reconstruct, 2,3)
#  x_reconstruct = np.swapaxes(x_reconstruct, 3,4)
  x_reconstruct = x_reconstruct.reshape([model.z_dim, images.shape[0], 64,model.K*64])
  images = model.reconstruct(sess, images)
  if not os.path.exists(flags.reconstr):
      os.mkdir(flags.reconstr)
  if not os.path.exists(os.path.join(flags.reconstr, directory)):
      os.mkdir(os.path.join(flags.reconstr, directory))
  for i in range(len(images)):
    org_img = images[i].reshape(64, 64)
    org_img = org_img.astype(np.float32)
    for j in range(model.z_dim):
        reconstr_img = x_reconstruct[j,i]
        if not os.path.exists(os.path.join(flags.reconstr, str(j))):
            os.mkdir(os.path.join(flags.reconstr, str(j)))
        reconstr_img = np.concatenate([org_img, reconstr_img], axis=1)
        imsave("{0}/{2}/reconstr_{1}.png".format(flags.reconstr,i,j),reconstr_img)
  x_reconstruct = x_reconstruct.reshape([model.z_dim, images.shape[0],64,model.K,64])
  x_reconstruct = np.swapaxes(x_reconstruct, 0, 1)
  x_reconstruct = np.swapaxes(x_reconstruct, 2, 3)
  x_reconstruct = x_reconstruct.reshape([images.shape[0],model.z_dim, model.K, 64,64])
  print(x_reconstruct.shape)
  images = np.squeeze(np.array (images))
  images = np.expand_dims(images, 1)
  images = np.repeat(images, repeats=model.z_dim, axis=1)
  images = np.expand_dims(images, 2)
  print(images.shape)
  x_reconstruct = np.concatenate([images, x_reconstruct], axis=2)
  np.save("{2}/{1}/reconstructed_{0}".format(step, directory, flags.reconstr),x_reconstruct)


def latentgen(sess, model, images):
    latents = model.generate_latent(sess, images)
    np.savetxt("latent-var.tsv", latents, delimiter="\t")



def disentangle_check(sess, model, manager, save_original=False):
  img = manager.get_image(shape=1, scale=2, orientation=5)
  if save_original:
      imsave("original.png", img.reshape(64, 64).astype(np.float32))

  batch_xs = [img]
  z_mean, z_log_sigma_sq = model.transform(sess, batch_xs)
  z_sigma_sq = np.exp(z_log_sigma_sq)[0]

  # Print variance
  zss_str = ""
  for i,zss in enumerate(z_sigma_sq):
      str = "z{0}={1:.4f}".format(i,zss)
  zss_str += str + ", "
  print(zss_str)

  # Save disentangled images
  z_m = z_mean[0]
  n_z = flags.z_dim

  if not os.path.exists("disentangle_img"):
      os.mkdir("disentangle_img")

  for target_z_index in range(n_z):
      for ri in range(n_z):
          value = -3.0 + (6.0 / 9.0) * ri
      z_mean2 = np.zeros((1, n_z))
      for i in range(n_z):
        if( i == target_z_index ):
          z_mean2[0][i] = value
        else:
          z_mean2[0][i] = z_m[i]
      reconstr_img = model.generate(sess, z_mean2)
      rimg = reconstr_img[0].reshape(64, 64)
      imsave("disentangle_img/check_z{0}_{1}.png".format(target_z_index,ri), rimg)


def load_checkpoints(sess):
  saver = tf.train.Saver()
  checkpoint = tf.train.get_checkpoint_state(flags.checkpoint_dir)
  if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess, checkpoint.model_checkpoint_path)
    print("loaded checkpoint: {0}".format(checkpoint.model_checkpoint_path))
  else:
    print("Could not find old checkpoint")
    if not os.path.exists(flags.checkpoint_dir):
        os.mkdir(flags.checkpoint_dir)
  return saver

def save_labels(labels):
    np.savetxt("metadata-var.tsv", labels, delimiter='\t', header='Class1\tClass2\tClass3\tClass4\tClass5\tClass6')


def main(argv):
  sess = tf.Session()

  model = VAE(gamma=flags.gamma,
          capacity_limit=flags.capacity_limit,
          capacity_change_duration=flags.capacity_change_duration,
          learning_rate=flags.learning_rate,
          mode=flags.mode,
          z_dim = flags.z_dim,
          perturb_val=flags.perturbration)

  sess.run(tf.global_variables_initializer())

  saver = load_checkpoints(sess)

  if flags.training:
    print("Loading data")
    manager = many_example_random_pair()
    manager_alt = single_factor_changed_one_example_generated_pair()
    manager.load()
    print("Data load complete")
      # Train
    train(sess, model, manager, saver)
  elif flags.latentgen :
    manager_alt = single_factor_changed_one_example_generated_pair()
    manager.load()
    print("Latent generation")
    images, labels = manager.get_random_images_wl(flags.numlatent * flags.batch_size)
    latentgen(sess, model, images)
    save_labels(labels)
  else:
    print("Loading data")
    manager = DataManager()
    manager.load()
    print("Data load complete")
    model.init_run(sess)
#    autoencoded_sample_save(manager=manager_alt, flags=flags, directory="default_autoencded_data", sess=sess, model=model)
    for i in range(1):
        images = manager.get_random_images(64)
        axis_walk_save( sess, model , images, step=i+1)


if __name__ == '__main__':
    tf.app.run()
