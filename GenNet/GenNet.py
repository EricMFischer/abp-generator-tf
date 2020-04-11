from __future__ import division

import os
import math
import numpy as np
import tensorflow as tf

import ops
import datasets

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class GenNet(object):
    def __init__(self, sess, config):
        self.sess = sess
        self.batch_size = config.batch_size # 11
        self.image_size = config.image_size # 64

        self.g_lr = config.g_lr # 0.001
        self.beta1 = config.beta1 # 0.5
        self.delta = config.delta # 0.3 (Langevin step size)
        self.sigma = config.sigma # 0.3
        self.sample_steps = config.sample_steps # 30
        self.z_dim = config.z_dim # 2

        self.num_epochs = config.num_epochs # 600
        self.data_path = os.path.join(config.data_path, config.category) # './Image/lion_tiger'
        self.log_step = config.log_step # 20

        self.output_dir = os.path.join(config.output_dir, config.category) # './output/lion_tiger'
        self.log_dir = os.path.join(self.output_dir, 'log') # './output/lion_tiger/log'
        self.sample_dir = os.path.join(self.output_dir, 'sample') # './output/lion_tiger/sample'
        self.model_dir = os.path.join(self.output_dir, 'checkpoints') # './output/lion_tiger/checkpoints'

        if tf.gfile.Exists(self.log_dir):
            tf.gfile.DeleteRecursively(self.log_dir)
        tf.gfile.MakeDirs(self.log_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)

        self.obs = tf.placeholder(shape=[None, self.image_size, self.image_size, 3], dtype=tf.float32, name='obs') # [None,64,64,3]
        # self.target_output = tf.placeholder(shape=[None, self.image_size, self.image_size, 3], dtype=tf.float32, name='target_output') # [None,64,64,3]
        self.z = tf.placeholder(shape=[None, self.z_dim], dtype=tf.float32, name='z') # [None,2]

        self.build_model()

    def build_model(self):
        ####################################################
        # Define the learning process. Record the loss.
        ####################################################
        self.target_output = self.generator(self.z)
        self.loss = self.loss_fn(self.obs, self.target_output)

        tf.summary.scalar('loss', self.loss)
        summary_op = tf.summary.merge_all()

        # generated and reconstructed images
        self.gen_imgs = self.generator(self.init_z(), reuse=True, is_training=False)
        self.recon_imgs = self.generator(self.z, reuse=True, is_training=False)

        # A tf.layers.batch_normalization layer for example would create some Ops that need to be run
        # every training step (update moving average and variance of the variables).
        # tf.GraphKeys.UPDATE_OPS is a collection of these variables. With the tf.control_dependencies
        # block, these Ops will get executed before the training op is run.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.g_lr, beta1=self.beta1).minimize(self.loss)

        self.lmc_sampling = self.langevin_dynamics(self.z)

        print('done building model')

    def loss_fn(self, x, gen_x):
        reduce_sum = tf.reduce_sum(1. / (2 * self.sigma**2) * tf.square(x - gen_x), axis=[1,2,3])
        return tf.reduce_mean(reduce_sum, axis=0)

    def get_required_conv_out_size(self, size, stride):
        return int(math.ceil(float(size) / float(stride)))

    def generator(self, inputs, reuse=False, is_training=True):
        ####################################################
        # Define the structure of generator, you may use the
        # generator structure of DCGAN. ops.py defines some
        # layers that you may use.
        ####################################################
        with tf.variable_scope('generator', reuse=reuse):
            size = self.image_size # 64
            s2 = self.get_required_conv_out_size(size, 2) # 32
            s4 = self.get_required_conv_out_size(s2, 2) # 16
            s8 = self.get_required_conv_out_size(s4, 2) # 8
            s16 = self.get_required_conv_out_size(s8, 2) # 4
            img_ch = 3

            # project z and reshape
            fc1 = ops.linear(inputs, self.image_size * 8 * s16**2, 'fc') # [11,8192]
            fc1_reshaped = tf.reshape(fc1, [-1, s16, s16, self.image_size * 8])
            h0 = ops.leaky_relu(fc1_reshaped) #[11,4,4,512]

            conv1 = ops.convt2d(h0, [self.batch_size, s8, s8, self.image_size * 4], name='g_h1')
            h1 = ops.leaky_relu(conv1) # [11,8,8,256]

            conv2 = ops.convt2d(h1, [self.batch_size, s4, s4, self.image_size * 2], name='g_h2')
            h2 = ops.leaky_relu(conv2) # [11,16,16,128]

            conv3 = ops.convt2d(h2, [self.batch_size, s2, s2, self.image_size * 1], name='g_h3')
            h3 = ops.leaky_relu(conv3) # [11,32,32,64]

            h4 = ops.convt2d(h3, [self.batch_size, size, size, img_ch], name='g_h4') # [11,64,64,3]
            return h4

    def langevin_dynamics(self, z_arg): # needs to be named z_arg?
        ####################################################
        # Define Langevin dynamics sampling operation.
        # To define multiple sampling steps, you may use
        # tf.while_loop to define a loop on computation graph.
        # The return should be the updated z.
        ####################################################
        # steps = tf.constant(0)
        cond = lambda steps, z: tf.less(steps, self.sample_steps)

        def langevin_step(steps, z): # z: [11,2]
            target_output = self.generator(z, reuse=True)
            recon_loss = tf.reduce_mean(tf.reduce_sum(1. / (2 * self.sigma**2) * tf.square(self.obs - target_output), axis=[1,2,3]), axis=0)
            dz_recon_loss = tf.gradients(ys=recon_loss, xs=z, name='dz_recon_loss')[0]
            dz_log_joint = -1 * (dz_recon_loss + z)
            dz_log_joint_term = (self.delta**2 / 2) * dz_log_joint

            noise = tf.random_normal(shape=tf.shape(z), mean=0., stddev=1.)
            noise_term = self.delta * noise

            updated_z = z + dz_log_joint_term + noise_term
            return steps + 1, updated_z

        # z = tf.while_loop(cond, langevin_step, [steps, z])[1].eval()
        # return z
        with tf.name_scope('langevin_dynamics'):
            steps = tf.constant(0)
            steps, z = tf.while_loop(cond=cond, body=langevin_step, loop_vars=[steps, z_arg])
            return z

    def init_z(self):
        return np.random.normal(loc=0., scale=1., size=[self.batch_size, self.z_dim]).astype('float32')

    def train(self):
        # Prepare training data
        train_data = datasets.DataSet(self.data_path, image_size=self.image_size)
        train_data = train_data.to_range(-1, 1) # [11,64,64,3]

        num_batches = int(math.ceil(len(train_data) / self.batch_size)) # 1
        summary_op = tf.summary.merge_all()

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        saver = tf.train.Saver(max_to_keep=50) # saves and restores variables
        writer = tf.summary.FileWriter(self.log_dir, self.sess.graph) # './output/lion_tiger/log'
        self.sess.graph.finalize() # marks graph as read-only to prevent memory leak

        print('Starting training ...')

        ####################################################
        # Train the model here. Print the loss term at each
        # epoch to monitor the training process. You may use
        # save_images() in ./datasets.py to save images. At
        # each log_step, record model in self.model_dir,
        # reconstructed images and synthesized images in
        # self.sample_dir, loss in self.log_dir (using writer).
        ####################################################
        loss_history = []

        feed_init = {self.z: self.init_z(), self.obs: train_data}
        feed_dict_lmc = feed_init
        feed_dict_train = feed_init

        for epoch in range(self.num_epochs): # 600
            z = self.sess.run(self.lmc_sampling, feed_dict=feed_dict_lmc) # [11,2]

            _, loss = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict_train)
            loss_history.append(loss)

            feed_dict_lmc.update({self.z: z})
            feed_dict_train.update({self.z: z})

            # print the loss each epoch
            print('[Epoch %d] Loss: %.3f' % (epoch + 1, loss))

            current_epoch = epoch + 1
            if (current_epoch % self.log_step):
                # save model in self.model_dir
                model_f = self.model_dir + '/%d_model' % current_epoch
                saver.save(self.sess, model_f, global_step=current_epoch)

                # save reconstructed images in self.sample_dir
                recon_imgs = self.sess.run(self.recon_imgs, feed_dict={self.z: z})
                recon_imgs_f = self.sample_dir + '/%d_recon' % current_epoch
                datasets.save_images(datasets.merge_images(recon_imgs), recon_imgs_f)

                # save generated images (using random z) in self.sample_dir
                gen_imgs = self.sess.run(self.gen_imgs)
                gen_imgs_f = self.sample_dir + '/%d_gen' % current_epoch
                datasets.save_images(datasets.merge_images(gen_imgs), gen_imgs_f)

                # save loss in self.log_dir (using writer)
                loss_summary = self.sess.run(summary_op)
                writer.add_summary(loss_summary, global_step=current_epoch)

        # after training is done, generate images with linearly interpolated latent factors from (-2,2) to (-2,2)
        # ex: interpolate 8 points from (-2,2) for each dimension of z, which lends an 8x8 panel of images
        print('Interplation to do')

        # plot loss over iteration
        plt.plot(loss_history)
        plt.show()
