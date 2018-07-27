from __future__ import print_function
from scipy.misc import imsave
import os
import tensorflow as tf
import numpy as np
import sys
from data_manager import many_example_random_pair, single_factor_changed_one_example_generated_pair
# TODO : add data manager

class FactorVAE(object):
    def __init__(self):
        self.z_dim = 6
        self.gamma = 35
        self.input_ph, self.enc_mean, self.enc_logvar, self.z_sample, self.dec_mean, self.dec_stoch = self._autoencoder_init()
        self.dec_img = tf.nn.sigmoid(self.dec_stoch)
        self.recon_loss, self.auto_encoder_loss, self.disc_loss = self._loss_init()
        self.ae_train_step, self.disc_train_step = self._optimizer_init()

        self.sess = tf.Session()
        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)
        self.sess.run(tf.global_variables_initializer())


    def train(self, data_manager):
        print("Beginning training", file=sys.stderr)
        it=0
        while it < 300000:
            it += 1
            self.sess.run(self.ae_train_step, {self.input_ph: data_manager.get_random_images(64).reshape([64,64,64,1])})
            self.sess.run(self.disc_train_step, {self.input_ph: data_manager.get_random_images().reshape([64,64,64,1])})
            if it % 100 == 0:
                self.step = it
                batch = data_manager.get_random_images().reshape([64,64,64,1])
                ae_train_loss = self.sess.run(self.auto_encoder_loss, {self.input_ph: batch})
                recon_train_loss = self.sess.run(self.recon_loss, {self.input_ph: batch})
                disc_train_loss = self.sess.run(self.disc_loss, {self.input_ph: batch})
                print("Iteration %i: \n    Autoencoder loss (train) %f\n    Reconstruction loss (train) %f\n    Discriminator loss (train) %f" % (it, ae_train_loss, recon_train_loss, disc_train_loss), file=open('train.log','a'))

                ae_test_loss = self.sess.run(self.auto_encoder_loss, {self.input_ph: data_manager.get_images([i for i in range(500)]).reshape([500,64,64,1])})
                recon_test_loss = self.sess.run(self.recon_loss, {self.input_ph: data_manager.get_images([i for i in range(500)]).reshape([500,64,64,1])})
                disc_test_loss = self.sess.run(self.disc_loss, {self.input_ph: data_manager.get_images([i for i in range(500)]).reshape([500,64,64,1])})
                print("    Autoencoder loss (test) %f\n    Reconstruction loss (test) %f\n    Discriminator loss (test) %f" % (ae_test_loss, recon_test_loss, disc_test_loss), file=open('train.log','a'))
            if it % 1000 == 0 :
                self.save_images(data_manager)

            if it % 10000 == 0:
                model_path = os.path.join(flags.checkpoint_dir, "model")
                save_path = self.saver.save(self.sess, model_path, global_step=it)
                self.save_images(data_manager)
                print("Model saved to: %s" % save_path, file=open('train.log','a'))

    def load_latest_checkpoint(self):
        self.step = 100000000
        self.saver.restore(self.sess, tf.train.latest_checkpoint(flags.checkpoint_dir))

    def make_plots(self):
        pass

    def _autoencoder_init(self):
        # make placeholder for feeding in data during training and evaluation
        input_ph = tf.placeholder(shape=[None, 64, 64, 1], dtype=tf.float32, name="input")
        # define the encoder network
        e_mean, e_logvar = self._encoder_init(input_ph)
        # reparameterisation trick
        eps = tf.random_normal(shape=tf.shape(e_mean))
        z_sample = e_mean + (tf.exp(e_logvar / 2) * eps)
        # define decoder network. d_stoch is decoding of random sample
        # from posterior, d_mean is decoding of mean of posterior
        d_stoch = self._decoder_init(inputs=z_sample)
        d_mean  = self._decoder_init(inputs=e_mean, reuse=True)

        return input_ph, e_mean, e_logvar, z_sample, d_mean, d_stoch

    def _encoder_init(self, inputs):
        with tf.variable_scope("encoder"):
            e_1 = tf.layers.conv2d(inputs=inputs, filters=32, kernel_size=4, strides=2, activation=tf.nn.relu, name="e_1", padding="same")
            e_2 = tf.layers.conv2d(inputs=e_1, filters=32, kernel_size=4, strides=2, activation=tf.nn.relu, name="e_2", padding="same")
            e_3 = tf.layers.conv2d(inputs=e_2, filters=64, kernel_size=4, strides=2, activation=tf.nn.relu, name="e_3", padding="same")
            e_4 = tf.layers.conv2d(inputs=e_3, filters=64, kernel_size=4, strides=2, activation=tf.nn.relu, name="e_4", padding="same")
            e_4_reshape = tf.reshape(e_4, shape=[-1] + [np.prod(e_4.get_shape().as_list()[1:])])
            e_5 = tf.layers.dense(inputs=e_4_reshape, units=128, name="e_5")
            e_mean = tf.layers.dense(inputs=e_5, units=self.z_dim, name="e_mean")
            e_logvar = tf.layers.dense(inputs=e_5, units=self.z_dim, name="e_logvar")

        return e_mean, e_logvar

    def _decoder_init(self, inputs, reuse=False):
        with tf.variable_scope("decoder"):
            d_1 = tf.layers.dense(inputs=inputs, units=128, activation=tf.nn.relu, name="d_1", reuse=reuse)
            d_2 = tf.layers.dense(inputs=d_1, units=1024, activation=tf.nn.relu, name="d_2", reuse=reuse)
            d_2_reshape = tf.reshape(d_2, shape=[-1, 4, 4, 64])
            d_3a = tf.layers.conv2d_transpose(inputs=d_2_reshape, filters=64, kernel_size=4, strides=2, activation=tf.nn.relu, name="d_3a", reuse=reuse, padding="same")
            d_3b = tf.layers.conv2d_transpose(inputs=d_3a, filters=64, kernel_size=4, strides=2, activation=tf.nn.relu, name="d_3b", reuse=reuse, padding="same")
            d_4 = tf.layers.conv2d_transpose(inputs=d_3b, filters=32, kernel_size=4, strides=2, activation=tf.nn.relu, name="d_4", reuse=reuse, padding="same")
            d_out = tf.layers.conv2d_transpose(inputs=d_4, filters=1, kernel_size=4, strides=2, name="d_out", reuse=reuse, padding="same")
            # In the FactorVAE paper, they say (Table 1) that they only use 3 "upconv" layers.
            # My code above deviates from that because I was getting d_out.shape=[?, 46, 46, 1]
            # with only 3 layers and default padding="valid", and [?, 32, 32, 1] with padding="same"
        return d_out

    def _discriminator_init(self, inputs, reuse=False):
        with tf.variable_scope("discriminator"):
            disc_1 = tf.layers.dense(inputs=inputs, units=1000, activation=tf.nn.leaky_relu, name="disc_1", reuse=reuse)
            disc_2 = tf.layers.dense(inputs=disc_1, units=1000, activation=tf.nn.leaky_relu, name="disc_2", reuse=reuse)
            disc_3 = tf.layers.dense(inputs=disc_2, units=1000, activation=tf.nn.leaky_relu, name="disc_3", reuse=reuse)
            disc_4 = tf.layers.dense(inputs=disc_3, units=1000, activation=tf.nn.leaky_relu, name="disc_4", reuse=reuse)
            disc_5 = tf.layers.dense(inputs=disc_4, units=1000, activation=tf.nn.leaky_relu, name="disc_5", reuse=reuse)
            disc_6 = tf.layers.dense(inputs=disc_5, units=1000, activation=tf.nn.leaky_relu, name="disc_6", reuse=reuse)

            logits = tf.layers.dense(inputs=disc_6, units=2, name="disc_logits", reuse=reuse)
            probabilities = tf.nn.softmax(logits)

        return logits, probabilities

    def save_images(self, data_manager):
        input_img = data_manager.get_random_images().reshape([64,64,64,1])
        reconstr_img = self.sess.run(self.dec_img, {
            self.input_ph : input_img
            })
        if not os.path.exists(flags.reconstr):
            os.mkdir(flags.reconstr)
        if not os.path.exists(os.path.join(flags.reconstr, str(self.step))):
            os.mkdir(os.path.join(flags.reconstr,str(self.step)))
        for i in range(len(input_img)):
            org_img = input_img[i].reshape(64,64).astype(np.float32)
            rimg = reconstr_img[i].reshape(64,64).astype(np.float32)
            imsave("{0}/{1}/org_{2}.png".format(flags.reconstr, self.step,i),org_img)
            imsave("{0}/{1}/reconstr_{2}.png".format(flags.reconstr, self.step,i),rimg)

    def _loss_init(self):
        ### Regulariser part of loss has two parts: KL divergence and Total Correlation
        ## KL part:
        KL_divergence = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.exp(tf.square(self.enc_logvar)) - tf.square(self.enc_logvar) + tf.square(self.enc_mean),axis=1) - self.z_dim)

        ## Total Correlation part:
        # permuted samples from q(z)
        real_samples = self.z_sample
        permuted_rows = []
        for i in range(real_samples.get_shape()[1]):
            permuted_rows.append(tf.random_shuffle(real_samples[:, i]))
        permuted_samples = tf.stack(permuted_rows, axis=1)

        # define discriminator network to distinguish between real and permuted q(z)
        logits_real, probs_real = self._discriminator_init(real_samples)
        logits_permuted, probs_permuted = self._discriminator_init(permuted_samples, reuse=True)

        # FactorVAE paper has gamma * log(D(z) / (1- D(z))) in Algorithm 2, where D(z) is probability of being real
        # Let PT be probability of being true, PF be probability of being false. Then we want log(PT/PF)
        # Since PT = exp(logit_T) / [exp(logit_T) + exp(logit_F)]
        # and  PF = exp(logit_F) / [exp(logit_T) + exp(logit_F)], we have that
        # log(PT/PF) = logit_T - logit_F
        tc_regulariser = self.gamma * tf.reduce_mean(logits_real[:, 0]  - logits_real[:, 1], axis=0)

        total_regulariser = KL_divergence + tc_regulariser

        ### Reconstruction loss is bernoulli
        im = self.input_ph
        im_flat = tf.reshape(im, shape=[-1, 64*64*1])
        logits = self.dec_stoch
        logits_flat = tf.reshape(logits, shape=[-1, 64*64*1])
        recon_loss = tf.reduce_mean(tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_flat,
                                                    labels=im_flat)),
                                                    name="recon_loss")

        auto_encoder_loss = tf.add(recon_loss,total_regulariser, name="auto_encoder_loss")

        ### Loss for discriminator
        disc_loss = tf.add(0.5 * tf.reduce_mean(tf.log(probs_real[:, 0])), 0.5 * tf.reduce_mean(tf.log(probs_permuted[:, 1])), name="disc_loss")

        return recon_loss, auto_encoder_loss, disc_loss


    def _optimizer_init(self):
        enc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
        dec_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder')
        disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

        ae_train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.auto_encoder_loss, var_list=enc_vars+dec_vars)
        disc_train_step = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(-self.disc_loss, var_list=disc_vars)

        return ae_train_step, disc_train_step


tf.app.flags.DEFINE_string("reconstr", "reconstr_img", "Directory for storing reconstructed images")
tf.app.flags.DEFINE_string("mode", "train", "Running mode of the program")
tf.app.flags.DEFINE_string("checkpoint_dir","checkpoints", "Directory for storing checkpoints")

flags = tf.app.flags.FLAGS

if __name__ == "__main__":
    vae = FactorVAE()
    data_manager = many_example_random_pair()
    data_manager.load()
    if flags.mode == "train":
        vae.train(data_manager)
    elif flags.mode == "load":
        vae.load_latest_checkpoint()
        vae.save_images(data_manager)
