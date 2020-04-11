##### dataset_svhn.py

import scipy.io as sio
import numpy as np
from matplotlib import pyplot as plt
import os


class Dataset:
    def __init__(self):
        self.name = "abstract"
        self.data_dims = []
        self.width = -1
        self.height = -1
        self.train_size = -1
        self.test_size = -1
        self.range = [0.0, 1.0]

    """ Get next training batch """
    def next_batch(self, batch_size):
        self.handle_unsupported_op()
        return None

    def next_test_batch(self, batch_size):
        self.handle_unsupported_op()
        return None

    def display(self, image):
        return image

    """ After reset, the same batches are output with the same calling order of next_batch or next_test_batch"""
    def reset(self):
        self.handle_unsupported_op()

    def handle_unsupported_op(self):
        print("Unsupported Operation")
        raise(Exception("Unsupported Operation"))



class SVHNDataset(Dataset):
    def __init__(self, db_path='', use_extra=True):
        Dataset.__init__(self)
        print("Loading files")
        self.data_dims = [32, 32, 3]
        self.range = [0.0, 1.0]
        self.name = "svhn"
        self.train_file = os.path.join(db_path, "train_32x32.mat")
        self.extra_file = os.path.join(db_path, "extra_32x32.mat")
        self.test_file = os.path.join(db_path, "test_32x32.mat")
        if use_extra:
            self.train_file = self.extra_file

        # Load training images
        if os.path.isfile(self.train_file):
            mat = sio.loadmat(self.train_file)
            self.train_image = mat['X'].astype(np.float32)
            self.train_label = mat['y']
            self.train_image = np.clip(self.train_image / 255.0, a_min=0.0, a_max=1.0)
        else:
            print("SVHN dataset train files not found")
            exit(-1)
        self.train_batch_ptr = 0
        self.train_size = self.train_image.shape[-1]

        if os.path.isfile(self.test_file):
            mat = sio.loadmat(self.test_file)
            self.test_image = mat['X'].astype(np.float32)
            self.test_label = mat['y']
            self.test_image = np.clip(self.test_image / 255.0, a_min=0.0, a_max=1.0)
        else:
            print("SVHN dataset test files not found")
            exit(-1)
        self.test_batch_ptr = 0
        self.test_size = self.test_image.shape[-1]
        print("SVHN loaded into memory")

    def next_batch(self, batch_size):
        prev_batch_ptr = self.train_batch_ptr
        self.train_batch_ptr += batch_size
        if self.train_batch_ptr > self.train_image.shape[-1]:       # Note the ordering of dimensions
            self.train_batch_ptr = batch_size
            prev_batch_ptr = 0
        return np.transpose(self.train_image[:, :, :, prev_batch_ptr:self.train_batch_ptr], (3, 0, 1, 2))

    def batch_by_index(self, batch_start, batch_end):
        return np.transpose(self.train_image[:, :, :, batch_start:batch_end], (3, 0, 1, 2))

    def next_test_batch(self, batch_size):
        prev_batch_ptr = self.test_batch_ptr
        self.test_batch_ptr += batch_size
        if self.test_batch_ptr > self.test_image.shape[-1]:
            self.test_batch_ptr = batch_size
            prev_batch_ptr = 0
        return np.transpose(self.test_image[:, :, :, prev_batch_ptr:self.test_batch_ptr], (3, 0, 1, 2))

    def display(self, image):
        return np.clip(image, 0.0, 1.0)

    def reset(self):
        self.train_batch_ptr = 0
        self.test_batch_ptr = 0


##### abstract_network.py

import tensorflow as tf
import numpy as np
import math
import glob
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
# from tensorflow.examples.tutorials.mnist import input_data
import os, sys, shutil, re

def lrelu(x, rate=0.1):
    # return tf.nn.relu(x)
    return tf.maximum(tf.minimum(x * rate, 0), x)

conv2d = tf.contrib.layers.convolution2d
conv2d_t = tf.contrib.layers.convolution2d_transpose
fc_layer = tf.contrib.layers.fully_connected


def conv2d_bn_lrelu(inputs, num_outputs, kernel_size, stride, is_training=True):
    conv = tf.contrib.layers.convolution2d(inputs, num_outputs, kernel_size, stride,
                                           weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                           weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5),
                                           activation_fn=tf.identity)
    conv = tf.contrib.layers.batch_norm(conv, is_training=is_training)
    conv = lrelu(conv)
    return conv


def conv2d_t_bn_relu(inputs, num_outputs, kernel_size, stride, is_training=True):
    conv = tf.contrib.layers.convolution2d_transpose(inputs, num_outputs, kernel_size, stride,
                                                     weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                                     weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5),
                                                     activation_fn=tf.identity)
    conv = tf.contrib.layers.batch_norm(conv, is_training=is_training)
    conv = lrelu(conv)
    return conv


def conv2d_t_bn(inputs, num_outputs, kernel_size, stride, is_training=True):
    conv = tf.contrib.layers.convolution2d_transpose(inputs, num_outputs, kernel_size, stride,
                                                     weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                                     weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5),
                                                     activation_fn=tf.identity, scope=None)
    conv = tf.contrib.layers.batch_norm(conv, is_training=is_training)
    return conv


def fc_bn_lrelu(inputs, num_outputs, is_training=True):
    fc = tf.contrib.layers.fully_connected(inputs, num_outputs,
                                           weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                           weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5),
                                           activation_fn=tf.identity)
    fc = tf.contrib.layers.batch_norm(fc, is_training=is_training)
    fc = lrelu(fc)
    return fc


def fc_bn_relu(inputs, num_outputs, is_training=True):
    fc = tf.contrib.layers.fully_connected(inputs, num_outputs,
                                           weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                           weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5),
                                           activation_fn=tf.identity)
    fc = tf.contrib.layers.batch_norm(fc, is_training=is_training)
    fc = tf.nn.relu(fc)
    return fc


def compute_kernel(x, y):
    x_size = tf.shape(x)[0]
    y_size = tf.shape(y)[0]
    dim = tf.shape(x)[1]
    tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
    tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
    return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))


def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)


class Network:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.learning_rate_placeholder = tf.placeholder(shape=[], dtype=tf.float32, name="lr_placeholder")

        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
        # A unique name should be given to each instance of subclasses during initialization
        self.name = "default"

        # These should be updated accordingly
        self.iteration = 0
        self.learning_rate = 0.0
        self.read_only = False

        self.do_generate_samples = False
        self.do_generate_conditional_samples = False
        self.do_generate_manifold_samples = False

    def make_model_path(self):
        if not os.path.isdir("models"):
            os.mkdir("models")
        if not os.path.isdir("models/" + self.name):
            os.mkdir("models/" + self.name)

    def print_network(self):
        self.make_model_path()
        if os.path.isdir("models/" + self.name):
            for f in os.listdir("models/" + self.name):
                if re.search(r"events.out*", f):
                    os.remove(os.path.join("models/" + self.name, f))
        self.writer = tf.summary.FileWriter("models/" + self.name, self.sess.graph)
        self.writer.flush()

    """ Save network, if network file already exists back it up to models/old folder. Only one back up will be created
    for each network """
    def save_network(self):
        if not self.read_only:
            # Saver and Summary ops cannot run in GPU
            with tf.device('/cpu:0'):
                saver = tf.train.Saver()
            self.make_model_path()
            if not os.path.isdir("models/old"):
                os.mkdir("models/old")
            file_name = "models/" + self.name + "/" + self.name + ".ckpt"
            if os.path.isfile(file_name):
                os.rename(file_name, "models/old/" + self.name + ".ckpt")
            saver.save(self.sess, file_name)

    """ Either initialize or load network from file.
    Always run this at end of initialization for every subclass to initialize Variables properly """
    def init_network(self, restart=False):
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        if restart:
            return
        file_name = "models/" + self.name + "/" + self.name + ".ckpt"
        if len(glob.glob(file_name + '*')) != 0:
            saver = tf.train.Saver()
            try:
                saver.restore(self.sess, file_name)
                print("Successfully restored model")
            except:
                print("Warning: network load failed, reinitializing all variables", sys.exc_info()[0])
                self.sess.run(tf.global_variables_initializer())
        else:
            print("No checkpoint file found, Initializing model from random")

    """ This function should train on the given batch and return the training loss """
    def train(self, batch_input, batch_target, labels=None):
        return None

    """ This function should take the input and return the reconstructed images """
    def test(self, batch_input, labels=None):
        return None





##### visualize.py

import os
import scipy.misc as misc

class Visualizer:
    def __init__(self, network):
        plt.ion()
        plt.show()
        self.fig = None
        self.network = network
        self.name = "default"
        self.save_epoch = 0
        self.array_save_epoch = 0

    def visualize(self, **args):
        pass

    def fig_to_file(self, fig=None):
        if fig is None:
            fig = self.fig
        img_folder = "models/" + self.network.name + "/" + self.name
        if not os.path.isdir(img_folder):
            os.makedirs(img_folder)

        fig_name = "current.png"
        fig.savefig(os.path.join(img_folder, fig_name))
        fig_name = "epoch%d" % self.save_epoch + ".png"
        fig.savefig(os.path.join(img_folder, fig_name))
        self.save_epoch += 1

    def arr_to_file(self, arr):
        img_folder = "models/" + self.network.name + "/" + self.name
        if not os.path.isdir(img_folder):
            os.makedirs(img_folder)

        if arr.shape[-1] == 1:
            misc.imsave(os.path.join(img_folder, 'current.png'), arr[:, :, 0])
            misc.imsave(os.path.join(img_folder, 'epoch%d.png' % self.save_epoch), arr[:, :, 0])
        else:
            misc.imsave(os.path.join(img_folder, 'current.png'), arr)
            misc.imsave(os.path.join(img_folder, 'epoch%d.png' % self.save_epoch), arr)
        self.save_epoch += 1


class ConditionalSampleVisualizer(Visualizer):
    """ sess should be the session where the visualized network run,
    visualized_variable should be a [batch_size, *] tensor, and title is the title of plotted graph """
    def __init__(self, network, dataset):
        Visualizer.__init__(self, network)
        self.dataset = dataset
        self.name = "conditional_samples"

    def visualize(self, layers, num_rows=4, use_gui=False):
        if use_gui and self.fig is None:
            self.fig, self.ax = plt.subplots(1, len(layers))
        latent_code = self.network.random_latent_code()

        canvas_list = []
        for i, layer in enumerate(layers):
            samples = np.zeros([num_rows*num_rows]+self.dataset.data_dims)
            samples_ptr = 0
            while samples_ptr < num_rows * num_rows:
                # Generate a few samples each time. Too many samples with distribution on latent code
                # different from training time breaks batch norm
                new_samples, _ = self.network.generate_conditional_samples(layer, latent_code)
                next_ptr = samples_ptr + new_samples.shape[0]
                if next_ptr > num_rows * num_rows:
                    next_ptr = num_rows * num_rows

                samples[samples_ptr:next_ptr] = new_samples[0:next_ptr-samples_ptr]
                samples_ptr = next_ptr

            # Plot the samples in a grid
            if samples is not None:
                samples = self.dataset.display(samples)
                width = samples.shape[1]
                height = samples.shape[2]
                channel = samples.shape[3]
                canvas = np.zeros((width * num_rows, height * num_rows, channel))
                for img_index1 in range(num_rows):
                    for img_index2 in range(num_rows):
                        canvas[img_index1 * width:(img_index1 + 1) * width,
                            img_index2 * height:(img_index2 + 1) * height, :] = \
                            samples[img_index1 * num_rows + img_index2, :, :, :]
                if use_gui:
                    self.ax[i].cla()
                    if channel == 1:
                        self.ax[i].imshow(canvas[:, :, 0], cmap=plt.get_cmap('Greys'))
                    else:
                        self.ax[i].imshow(canvas)
                    self.ax[i].xaxis.set_visible(False)
                    self.ax[i].yaxis.set_visible(False)
                if i != 0:
                    if canvas.shape[-1] == 1:
                        canvas_list.append(np.zeros((width * num_rows, 20, channel)))
                    else:
                        canvas_list.append(np.ones((width * num_rows, 20, channel)))
                canvas_list.append(canvas)
            else:
                print("Warning: no samples generated during visualization")
        # np.save('samples', canvas)
        canvas = np.concatenate(canvas_list, axis=1)
        self.arr_to_file(canvas)
        if use_gui:
            self.fig.suptitle('Conditional Samples for %s' % self.network.name)
            plt.draw()
            plt.pause(0.01)


class SampleVisualizer(Visualizer):
    """ sess should be the session where the visualized network run,
    visualized_variable should be a [batch_size, *] tensor, and title is the title of plotted graph """
    def __init__(self, network, dataset):
        Visualizer.__init__(self, network)
        # self.fig.suptitle("Samples generated by " + str(network.name))
        self.dataset = dataset
        self.name = "samples"

    def visualize(self, num_rows=10, use_gui=False):
        if use_gui and self.fig is None:
            self.fig, self.ax = plt.subplots()
        samples = self.network.generate_samples()
        if samples is not None:
            samples = self.dataset.display(samples)
            width = samples.shape[1]
            height = samples.shape[2]
            channel = samples.shape[3]
            canvas = np.zeros((width * num_rows, height * num_rows, channel))
            for img_index1 in range(num_rows):
                for img_index2 in range(num_rows):
                    canvas[img_index1*width:(img_index1+1)*width, img_index2*height:(img_index2+1)*height, :] = \
                        samples[img_index1*num_rows+img_index2, :, :, :]
            self.arr_to_file(canvas)

            if use_gui:
                self.ax.cla()
                if channel == 1:
                    self.ax.imshow(canvas[:, :, 0], cmap=plt.get_cmap('Greys'))
                else:
                    self.ax.imshow(canvas)
                self.ax.xaxis.set_visible(False)
                self.ax.yaxis.set_visible(False)

                self.fig.suptitle('Samples for %s' % self.network.name)
                plt.draw()
                plt.pause(0.01)


class ManifoldSampleVisualizer(Visualizer):
    def __init__(self, network, dataset):
        Visualizer.__init__(self, network)
        # self.fig.suptitle("Samples generated by " + str(network.name))
        self.dataset = dataset
        self.name = "manifold_samples"

    def visualize(self, layers, num_rows=4, use_gui=False):
        if use_gui and self.fig is None:
            self.fig, self.ax = plt.subplots(1, len(layers))
        canvas_list = []
        for i, layer in enumerate(layers):
            samples = np.zeros([num_rows*num_rows]+self.dataset.data_dims)
            samples_ptr = 0
            latent_code_x = np.tile(np.reshape(np.linspace(-2.0, 2.0, num=num_rows), (1, num_rows)), (num_rows, 1))
            latent_code_y = latent_code_x.transpose()
            latent_code = np.reshape(np.stack([latent_code_x, latent_code_y], axis=-1), (-1, 2))
            while samples_ptr < num_rows * num_rows:
                new_samples = self.network.generate_manifold_samples(layer, latent_code)
                latent_code = latent_code[new_samples.shape[0]:]
                next_ptr = samples_ptr + new_samples.shape[0]
                if next_ptr > num_rows * num_rows:
                    next_ptr = num_rows * num_rows

                samples[samples_ptr:next_ptr] = new_samples[0:next_ptr-samples_ptr]
                samples_ptr = next_ptr
            if samples is not None:
                width = samples.shape[1]
                height = samples.shape[2]
                channel = samples.shape[3]
                canvas = np.zeros((width * num_rows, height * num_rows, channel))
                for img_index1 in range(num_rows):
                    for img_index2 in range(num_rows):
                        canvas[img_index1 * width:(img_index1 + 1) * width,
                        img_index2 * height:(img_index2 + 1) * height, :] = \
                            self.dataset.display(samples[img_index1 * num_rows + img_index2, :, :, :])
                if use_gui:
                    self.ax[i].cla()
                    if channel == 1:
                        self.ax[i].imshow(canvas[:, :, 0], cmap=plt.get_cmap('Greys'))
                    else:
                        self.ax[i].imshow(canvas)
                    self.ax[i].xaxis.set_visible(False)
                    self.ax[i].yaxis.set_visible(False)
                if i != 0:
                    if canvas.shape[-1] == 1:
                        canvas_list.append(np.zeros((width * num_rows, 20, channel)))
                    else:
                        canvas_list.append(np.ones((width * num_rows, 20, channel)))
                canvas_list.append(canvas)
        canvas = np.concatenate(canvas_list, axis=1)
        self.arr_to_file(canvas)

        if use_gui:
            self.fig.suptitle('Manifold Samples for %s' % self.network.name)
            plt.draw()
            plt.pause(0.01)



##### trainer.py

import time


class NoisyTrainer:
    def __init__(self, network, dataset, args):
        self.network = network
        self.dataset = dataset
        self.args = args
        self.batch_size = args.batch_size
        self.data_dims = self.dataset.data_dims
        self.train_with_mask = False
        self.train_discrete = False

        self.fig, self.ax = None, None
        self.network = network
        self.test_reconstruction_error = True

    def get_noisy_input(self, original):
        if not self.args.denoise_train:
            return original

        # Add salt and pepper noise
        noisy_input = np.multiply(original, np.random.binomial(n=1, p=0.9, size=[self.batch_size] + self.data_dims)) + \
                      np.random.binomial(n=1, p=0.1, size=[self.batch_size] + self.data_dims)

        # Add Gaussian noise
        noisy_input += np.random.normal(scale=0.1, size=[self.batch_size] + self.dataset.data_dims)

        # Activate following code to remove entire window of content. Not recommended
        # removed_width = random.randint(10, int(round(self.data_dims[0]/1.5)))
        # removed_height = random.randint(10, int(round(self.data_dims[1]/1.5)))
        # removed_left = random.randint(0, self.data_dims[0] - removed_width - 1)
        # removed_right = removed_left + removed_width
        # removed_top = random.randint(0, self.data_dims[1] - removed_height - 1)
        # removed_bottom = removed_top + removed_height
        # if random.random() > 0.5:
        #     noisy_input[:, removed_left:removed_right, removed_top:removed_bottom, :] = \
        #         np.zeros((self.batch_size, removed_width, removed_height, self.data_dims[-1]), dtype=np.float)
        # else:
        #     noisy_input[:, removed_left:removed_right, removed_top:removed_bottom, :] = \
        #         np.ones((self.batch_size, removed_width, removed_height, self.data_dims[-1]), dtype=np.float)

        return np.clip(noisy_input, a_min=self.dataset.range[0], a_max=self.dataset.range[1])

    def train(self):
        # Visualization
        if self.network.do_generate_samples:
            sample_visualizer = SampleVisualizer(self.network, self.dataset)
        if self.network.do_generate_conditional_samples:
            sample_visualizer_conditional = ConditionalSampleVisualizer(self.network, self.dataset)
        if self.network.do_generate_manifold_samples:
            sample_visualizer_manifold = ManifoldSampleVisualizer(self.network, self.dataset)

        iteration = 0
        # z_lmc = np.random.normal(size=[self.batch_size, 25])
        while True:
            z_lmc = np.random.normal(size=[self.batch_size, 25])
            iter_time = time.time()
            images = self.dataset.next_batch(self.batch_size)
            noisy_input = self.get_noisy_input(images)
            recon_loss, z_lmc = self.network.train(noisy_input, images, z_lmc)
            # recon_loss = self.network.train(noisy_input, images)

            if iteration % 20 == 0:
                print("Iteration %d: Reconstruction loss %f, time per iter %fs" %
                      (iteration, recon_loss, time.time() - iter_time))
                print(z_lmc[:2, :])

            if iteration % self.args.vis_frequency == 0:
                # test_error = self.test(iteration//self.args.vis_frequency, 5)
                # print("Reconstruction error @%d per pixel: " % iteration, test_error)

                layers = [layer for layer in self.network.random_latent_code()]
                layers.sort()
                print("Visualizing %s" % layers)
                if self.network.do_generate_samples:
                    sample_visualizer.visualize(num_rows=10, use_gui=self.args.use_gui)
                if self.network.do_generate_conditional_samples:
                    sample_visualizer_conditional.visualize(layers=layers, num_rows=10, use_gui=self.args.use_gui)
                if self.network.do_generate_manifold_samples:
                    sample_visualizer_manifold.visualize(layers=layers, num_rows=30, use_gui=self.args.use_gui)
            iteration += 1

    def visualize(self):
        layers = [layer for layer in self.network.random_latent_code()]
        layers.sort()

        # Visualization
        if self.network.do_generate_samples:
            sample_visualizer = SampleVisualizer(self.network, self.dataset)
            sample_visualizer.visualize(num_rows=10, use_gui=self.args.use_gui)
        if self.network.do_generate_conditional_samples:
            sample_visualizer_conditional = ConditionalSampleVisualizer(self.network, self.dataset)
            sample_visualizer_conditional.visualize(layers=layers, num_rows=10, use_gui=self.args.use_gui)
        if self.network.do_generate_manifold_samples:
            sample_visualizer_manifold = ManifoldSampleVisualizer(self.network, self.dataset)
            sample_visualizer_manifold.visualize(layers=layers, num_rows=30, use_gui=self.args.use_gui)

    """ Returns reconstruction error per pixel """

    def test(self, epoch, num_batch=3):
        error = 0.0
        for test_iter in range(num_batch):
            test_image = self.dataset.next_test_batch(self.batch_size)
            noisy_test_image = self.get_noisy_input(test_image)
            reconstruction = self.network.test(noisy_test_image)
            error += np.sum(np.square(reconstruction - test_image)) / np.prod(self.data_dims[:2]) / self.batch_size
            if test_iter == 0 and self.args.plot_reconstruction:
                # Plot the original image, noisy image, and reconstructed image
                self.plot_reconstruction(epoch, test_image, noisy_test_image, reconstruction)
        return error / num_batch

    def plot_reconstruction(self, epoch, test_image, noisy_image, reconstruction, num_plot=3):
        if test_image.shape[-1] == 1:  # Black background for mnist, white for color images
            canvas = np.zeros((num_plot * self.data_dims[0], 3 * self.data_dims[1] + 20, self.data_dims[2]))
        else:
            canvas = np.ones((num_plot * self.data_dims[0], 3 * self.data_dims[1] + 20, self.data_dims[2]))
        for img_index in range(num_plot):
            canvas[img_index * self.data_dims[0]:(img_index + 1) * self.data_dims[0], 0:self.data_dims[1]] = \
                self.dataset.display(test_image[img_index, :, :])
            canvas[img_index * self.data_dims[0]:(img_index + 1) * self.data_dims[0], self.data_dims[1] + 10:self.data_dims[1] * 2 + 10] = \
                self.dataset.display(noisy_image[img_index, :, :])
            canvas[img_index * self.data_dims[0]:(img_index + 1) * self.data_dims[0], self.data_dims[1] * 2 + 20:] = \
                self.dataset.display(reconstruction[img_index, :, :])

        img_folder = "models/" + self.network.name + "/reconstruction"
        if not os.path.isdir(img_folder):
            os.makedirs(img_folder)

        if canvas.shape[-1] == 1:
            misc.imsave(os.path.join(img_folder, 'current.png'), canvas[:, :, 0])
            misc.imsave(os.path.join(img_folder, 'epoch%d.png' % epoch), canvas[:, :, 0])
        else:
            misc.imsave(os.path.join(img_folder, 'current.png'), canvas)
            misc.imsave(os.path.join(img_folder, 'epoch%d.png' % epoch), canvas)

        if self.args.use_gui:
            if self.fig is None:
                self.fig, self.ax = plt.subplots()
                self.fig.suptitle("Reconstruction of " + str(self.network.name))
            self.ax.cla()
            if canvas.shape[-1] == 1:
                self.ax.imshow(canvas[:, :, 0], cmap=plt.get_cmap('Greys'))
            else:
                self.ax.imshow(canvas)
            plt.draw()
            plt.pause(1)



#### vladder.py

class VLadder(Network):
    def __init__(self, dataset, name=None, reg='kl', batch_size=100, restart=True):
        Network.__init__(self, dataset, batch_size)
        if name is None or name == '':
            self.name = "vladder_%s" % dataset.name
        else:
            self.name = name
        self.dataset = dataset
        self.lmc_step = 20
        self.sigma2 = 0.3
        self.delta2 = 0.3
        self.batch_size = batch_size
        self.data_dims = self.dataset.data_dims
        self.latent_noise = False
        self.restart = restart

        self.fs = [self.data_dims[0], self.data_dims[0] // 2, self.data_dims[0] // 4, self.data_dims[0] // 8,
                   self.data_dims[0] // 16]
        self.reg = reg
        if self.reg != 'kl' and self.reg != 'mmd':
            print("Unknown regularization, supported: kl, mmd")

        # Configurations
        if self.name == "vladder_celebA":
            self.cs = [3, 64, 128, 256, 512, 1024]
            self.ladder0_dim = 10
            self.ladder1_dim = 10
            self.ladder2_dim = 10
            self.ladder3_dim = 10
            self.num_layers = 4
            loss_ratio = 0.5
            layers = LargeLayers(self)
            self.do_generate_conditional_samples = True
            self.do_generate_samples = True
        elif self.name == "vladder_lsun":
            self.cs = [3, 64, 128, 256, 512, 1024]
            self.ladder0_dim = 20
            self.ladder1_dim = 20
            self.ladder2_dim = 20
            self.ladder3_dim = 40
            self.num_layers = 4
            loss_ratio = 0.5
            layers = LargeLayers(self)
            self.do_generate_conditional_samples = True
        elif self.name == "vladder_svhn":
            self.cs = [3, 64, 128, 256, 1024]
            self.ladder0_dim = 5
            self.ladder1_dim = 5
            self.ladder2_dim = 5
            self.ladder3_dim = 10
            self.z_dim = self.ladder0_dim + self.ladder1_dim + self.ladder2_dim + self.ladder3_dim
            self.z_sub_dim = [5, 5, 5, 10]
            self.num_layers = 4
            loss_ratio = 8.0
            self.layers = MediumLayers(self)
            self.do_generate_conditional_samples = True
        elif self.name == "vladder_mnist":
            self.cs = [1, 64, 128, 1024]
            self.ladder0_dim = 2
            self.ladder1_dim = 2
            self.ladder2_dim = 2
            self.num_layers = 3
            loss_ratio = 8.0
            self.error_scale = 8.0
            layers = SmallLayers(self)
            self.do_generate_manifold_samples = True
        else:
            print("Unknown architecture name %s" % self.name)
            exit(-1)
        self.self = self

        self.input_placeholder = tf.placeholder(shape=[None] + self.data_dims, dtype=tf.float32, name="input_placeholder")
        self.target_placeholder = tf.placeholder(shape=[None] + self.data_dims, dtype=tf.float32, name="target_placeholder")
        self.is_training = tf.placeholder(tf.bool, name='phase')
        self.z = tf.placeholder(shape=[None, self.z_dim], dtype=tf.float32, name='z')

        self.iteration = 0

        self.build_model()

    def build_model(self):

        self.ladders = {}
        self.ladder3_placeholder = tf.placeholder(shape=(None, self.ladder3_dim), dtype=tf.float32, name="prior_ladder3")
        self.iladder3_sample = tf.placeholder(shape=(None, self.ladder3_dim), dtype=tf.float32, name="posterior_ladder3")
        self.ladders['ladder3'] = [self.ladder3_placeholder, self.ladder3_dim]

        self.ladder2_placeholder = tf.placeholder(shape=(None, self.ladder2_dim), dtype=tf.float32, name="prior_ladder2")
        self.iladder2_sample = tf.placeholder(shape=(None, self.ladder2_dim), dtype=tf.float32, name="posterior_ladder2")
        self.ladders['ladder2'] = [self.ladder2_placeholder, self.ladder2_dim]

        self.ladder1_placeholder = tf.placeholder(shape=(None, self.ladder1_dim), dtype=tf.float32, name="prior_ladder1")
        self.iladder1_sample = tf.placeholder(shape=(None, self.ladder1_dim), dtype=tf.float32, name="posterior_ladder1")
        self.ladders['ladder1'] = [self.ladder1_placeholder, self.ladder1_dim]

        self.ladder0_placeholder = tf.placeholder(shape=(None, self.ladder0_dim), dtype=tf.float32, name="prior_ladder0")
        self.iladder0_sample = tf.placeholder(shape=(None, self.ladder0_dim), dtype=tf.float32, name="posterior_ladder0")
        self.ladders['ladder0'] = [self.ladder0_placeholder, self.ladder0_dim]

        self.toutput, self.goutput = self.generator(self.ladders, tf.stop_gradient(self.iladder0_sample),
                                                    tf.stop_gradient(self.iladder1_sample), tf.stop_gradient(self.iladder2_sample),
                                                    tf.stop_gradient(self.iladder3_sample), is_training=self.is_training)

        # Loss and training operators
        self.loss = tf.reduce_mean(tf.reduce_sum(
            1.0 / (2 * self.sigma2 * self.sigma2) * tf.square(self.toutput - self.target_placeholder), axis=[1, 2, 3]), axis=0)

        tf.summary.scalar("loss", self.loss)

        self.merged_summary = tf.summary.merge_all()

        # self.lmc_sampling = self.lmc(self.z)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(self.loss)

        self.lmc_sampling = self.lmc(self.z)

        # Set restart=True to not ignore previous checkpoint and restart training
        self.init_network(restart=self.restart)
        self.print_network()
        # Set read_only=True to not overwrite previous checkpoint
        self.read_only = False

    def lmc(self, z_arg):
        def cond(i, z):
            return tf.less(i, self.lmc_step)

        def body(i, z):
            noise = tf.random_normal(shape=tf.shape(z), name='noise')
            iladder0_sample, \
            iladder1_sample, \
            iladder2_sample, \
            iladder3_sample = tf.split(z, self.z_sub_dim, 1)
            # print(tf.shape(iladder3_sample))

            toutput = self.generator(self.ladders, iladder0_sample,
                                     iladder1_sample, iladder2_sample, iladder3_sample, is_training=True, reuse=True)
            gen_loss = tf.reduce_mean(tf.reduce_sum(
                1.0 / (2 * self.sigma2 * self.sigma2) * tf.square(toutput - self.target_placeholder), axis=[1, 2, 3]), axis=0)
            grad = tf.gradients(gen_loss, z, name='grad_gen')[0]
            z = z - 0.5 * self.delta2 * self.delta2 * (z + grad) + self.delta2 * noise
            return tf.add(i, 1), z

        with tf.name_scope("lmc"):
            i = tf.constant(0)
            i, z = tf.while_loop(cond, body, [i, z_arg])
            return z

    def generator(self, ladders, iladder0_sample, iladder1_sample, iladder2_sample, iladder3_sample, is_training, reuse=False):
        if self.num_layers >= 4 and self.ladder3_dim > 0:
            tlatent3_state = self.layers.generative3(None, iladder3_sample, is_training=is_training, reuse=reuse)
            glatent3_state = self.layers.generative3(None, ladders['ladder3'][0], reuse=True, is_training=False)
        else:
            tlatent3_state, glatent3_state = None, None

        if self.num_layers >= 3 and self.ladder2_dim > 0:
            tlatent2_state = self.layers.generative2(tlatent3_state, iladder2_sample, is_training=is_training, reuse=reuse)
            glatent2_state = self.layers.generative2(glatent3_state, ladders['ladder2'][0], reuse=True, is_training=False)
        elif tlatent3_state is not None:
            tlatent2_state = self.layers.generative2(tlatent3_state, None, is_training=is_training, reuse=reuse)
            glatent2_state = self.layers.generative2(glatent3_state, None, reuse=True, is_training=False)
        else:
            tlatent2_state, glatent2_state = None, None

        if self.num_layers >= 2 and self.ladder1_dim > 0:
            tlatent1_state = self.layers.generative1(tlatent2_state, iladder1_sample, is_training=is_training, reuse=reuse)
            glatent1_state = self.layers.generative1(glatent2_state, ladders['ladder1'][0], reuse=True, is_training=False)
        elif tlatent2_state is not None:
            tlatent1_state = self.layers.generative1(tlatent2_state, None, is_training=is_training, reuse=reuse)
            glatent1_state = self.layers.generative1(glatent2_state, None, reuse=True, is_training=False)
        else:
            tlatent1_state, glatent1_state = None, None

        if self.ladder0_dim > 0:
            toutput = self.layers.generative0(tlatent1_state, iladder0_sample, is_training=is_training, reuse=reuse)
            goutput = self.layers.generative0(glatent1_state, ladders['ladder0'][0], reuse=True, is_training=False)
        elif tlatent1_state is not None:
            toutput = self.layers.generative0(tlatent1_state, None, is_training=is_training, reuse=reuse)
            goutput = self.layers.generative0(glatent1_state, None, reuse=True, is_training=False)
        else:
            print("Error: no active ladder")
            exit(0)

        return toutput, goutput

    def train(self, batch_input, batch_target, z_lmc, label=None):
        self.iteration += 1

        # These are used for batch norm updates of generative model
        codes = {key: np.random.normal(size=[self.batch_size, self.ladders[key][1]]) for key in self.ladders}
        feed_dict_lmc = {self.ladders[key][0]: codes[key] for key in self.ladders}
        feed_dict_train = {self.ladders[key][0]: codes[key] for key in self.ladders}

        feed_dict_lmc.update({
            # self.input_placeholder: batch_input,
            self.target_placeholder: batch_target,
            self.is_training: True
        })

        feed_dict_train.update({
            # self.input_placeholder: batch_input,
            self.target_placeholder: batch_target,
            self.is_training: True
        })

        feed_dict_lmc.update({self.z: z_lmc})
        # print(z_lmc[:2])
        # idx = self.sess.run(self.idx, feed_dict=feed_dict_lmc)
        # print(idx)
        z_lmc = self.sess.run(self.lmc_sampling, feed_dict=feed_dict_lmc)
        # print(z_lmc[:2])

        z_lmc_split = np.split(z_lmc, [5, 10, 15], axis=1)

        feed_dict_train.update(
            {self.iladder0_sample: z_lmc_split[0],
             self.iladder1_sample: z_lmc_split[1],
             self.iladder2_sample: z_lmc_split[2],
             self.iladder3_sample: z_lmc_split[3]
             })

        _, loss = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict_train)
        if self.iteration % 2000 == 0:
            self.save_network()
        if self.iteration % 20 == 0:
            summary = self.sess.run(self.merged_summary, feed_dict=feed_dict_train)
            self.writer.add_summary(summary, self.iteration)

        print(self.iteration)
        return loss, z_lmc

    def test(self, batch_input, label=None):
        train_return = self.sess.run(self.toutput,
                                     feed_dict={self.input_placeholder: batch_input, self.is_training: False})
        return train_return

    def inference(self, batch_input):
        tensor_handle = [self.ladders[key][2] for key in self.ladders]
        tensor_value = self.sess.run(tensor_handle, feed_dict={self.input_placeholder: batch_input, self.is_training: False})
        return {name: value for name, value in zip(self.ladders, tensor_value)}

    def generate(self, codes):
        feed_dict = {self.ladders[key][0]: codes[key] for key in self.ladders}
        feed_dict[self.is_training] = False
        output = self.sess.run(self.goutput, feed_dict=feed_dict)
        return output

    def random_latent_code(self):
        return {key: np.random.normal(size=[self.ladders[key][1]]) for key in self.ladders}

    def generate_conditional_samples(self, condition_layer, condition_code):
        codes = {key: np.random.normal(size=[self.batch_size, self.ladders[key][1]]) for key in self.ladders}

        # To avoid breaking batch normalization the fixed codes must be inserted at random locations
        random_indexes = np.random.choice(range(self.batch_size), size=8, replace=False)
        for key in codes:
            if condition_layer != key:
                codes[key][random_indexes] = condition_code[key]

        feed_dict = {self.ladders[key][0]: codes[key] for key in self.ladders}
        feed_dict[self.is_training] = False
        output = self.sess.run(self.goutput, feed_dict=feed_dict)
        return output[random_indexes], condition_code

    def generate_samples(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        codes = {key: np.random.normal(size=[batch_size, self.ladders[key][1]]) for key in self.ladders}
        feed_dict = {self.ladders[key][0]: codes[key] for key in self.ladders}
        feed_dict[self.is_training] = False
        output = self.sess.run(self.goutput, feed_dict=feed_dict)
        return output

    def generate_manifold_samples(self, external_layer, external_code):
        codes = {key: np.random.normal(size=[external_code.shape[0], self.ladders[key][1]]) for key in self.ladders}

        # To avoid breaking batch normalization fixed code must be inserted at random locations
        # num_insertions = 8
        # if num_insertions > external_code.shape[0]:
        #     num_insertions = external_code.shape[0]
        # random_indexes = np.random.choice(range(self.batch_size), size=num_insertions, replace=False)
        codes[external_layer] = external_code
        feed_dict = {self.ladders[key][0]: codes[key] for key in self.ladders}
        feed_dict[self.is_training] = False
        output = self.sess.run(self.goutput, feed_dict=feed_dict)
        return output



#### vladder_medium.py

class MediumLayers:
    """ Definition of layers for a medium sized ladder network """
    def __init__(self, network):
        self.network = network

    def inference0(self, input_x, is_training=True):
        with tf.variable_scope("inference0"):
            conv1 = conv2d_bn_lrelu(input_x, self.network.cs[1], [4, 4], 2, is_training)
            conv2 = conv2d_bn_lrelu(conv1, self.network.cs[1], [4, 4], 1, is_training)
            return conv2

    def ladder0(self, input_x, is_training=True):
        with tf.variable_scope("ladder0"):
            conv1 = conv2d_bn_lrelu(input_x, self.network.cs[1], [4, 4], 2, is_training)
            conv2 = conv2d_bn_lrelu(conv1, self.network.cs[1], [4, 4], 1, is_training)
            conv2 = tf.reshape(conv2, [-1, np.prod(conv2.get_shape().as_list()[1:])])
            fc1_mean = tf.contrib.layers.fully_connected(conv2, self.network.ladder0_dim, activation_fn=tf.identity)
            fc1_stddev = tf.contrib.layers.fully_connected(conv2, self.network.ladder0_dim, activation_fn=tf.sigmoid)
            return fc1_mean, fc1_stddev

    def inference1(self, latent1, is_training=True):
        with tf.variable_scope("inference1"):
            conv3 = conv2d_bn_lrelu(latent1, self.network.cs[2], [4, 4], 2, is_training)
            conv4 = conv2d_bn_lrelu(conv3, self.network.cs[2], [4, 4], 1, is_training)
            return conv4

    def ladder1(self, latent1, is_training=True):
        with tf.variable_scope("ladder1"):
            conv3 = conv2d_bn_lrelu(latent1, self.network.cs[2], [4, 4], 2, is_training)
            conv4 = conv2d_bn_lrelu(conv3, self.network.cs[2], [4, 4], 1, is_training)
            conv4 = tf.reshape(conv4, [-1, np.prod(conv4.get_shape().as_list()[1:])])
            fc1_mean = tf.contrib.layers.fully_connected(conv4, self.network.ladder1_dim, activation_fn=tf.identity)
            fc1_stddev = tf.contrib.layers.fully_connected(conv4, self.network.ladder1_dim, activation_fn=tf.sigmoid)
            return fc1_mean, fc1_stddev

    def inference2(self, latent2, is_training=True):
        with tf.variable_scope("inference2"):
            conv5 = conv2d_bn_lrelu(latent2, self.network.cs[3], [4, 4], 2, is_training)
            conv6 = conv2d_bn_lrelu(conv5, self.network.cs[3], [4, 4], 1, is_training)
            conv6 = tf.reshape(conv6, [-1, np.prod(conv6.get_shape().as_list()[1:])])
            return conv6

    def ladder2(self, latent2, is_training=True):
        with tf.variable_scope("ladder2"):
            conv5 = conv2d_bn_lrelu(latent2, self.network.cs[3], [4, 4], 2, is_training)
            conv6 = conv2d_bn_lrelu(conv5, self.network.cs[3], [4, 4], 1, is_training)
            conv6 = tf.reshape(conv6, [-1, np.prod(conv6.get_shape().as_list()[1:])])
            fc2_mean = tf.contrib.layers.fully_connected(conv6, self.network.ladder2_dim, activation_fn=tf.identity)
            fc2_stddev = tf.contrib.layers.fully_connected(conv6, self.network.ladder2_dim, activation_fn=tf.sigmoid)
            return fc2_mean, fc2_stddev

    def ladder3(self, latent3, is_training=True):
        with tf.variable_scope("ladder3"):
            fc1 = fc_bn_lrelu(latent3, self.network.cs[4], is_training)
            fc2 = fc_bn_lrelu(fc1, self.network.cs[4], is_training)
            fc3_mean = tf.contrib.layers.fully_connected(fc2, self.network.ladder3_dim, activation_fn=tf.identity)
            fc3_stddev = tf.contrib.layers.fully_connected(fc2, self.network.ladder3_dim, activation_fn=tf.sigmoid)
            return fc3_mean, fc3_stddev

    def generative0(self, latent1, ladder0, reuse=tf.AUTO_REUSE, is_training=True):
        with tf.variable_scope("generative0", reuse=tf.AUTO_REUSE):
        # with tf.variable_scope("generative0") as gs:
        #     if reuse:
        #         gs.reuse_variables()
            if ladder0 is not None:
                ladder0 = fc_bn_relu(ladder0, int(self.network.fs[1] * self.network.fs[1] * self.network.cs[1]), is_training)
                ladder0 = tf.reshape(ladder0, [-1, self.network.fs[1], self.network.fs[1], self.network.cs[1]])
                if latent1 is not None:
                    latent1 = tf.concat(values=[latent1, ladder0], axis=3)
                else:
                    latent1 = ladder0
            elif latent1 is None:
                print("Generative layer must have input")
                exit(0)
            conv1 = conv2d_t_bn_relu(latent1, self.network.cs[1], [4, 4], 2, is_training)
            output = tf.contrib.layers.convolution2d_transpose(conv1, self.network.data_dims[2], [4, 4], 1,
                                                               activation_fn=tf.sigmoid)
            output = (self.network.dataset.range[1] - self.network.dataset.range[0]) * output + self.network.dataset.range[0]
            return output

    def generative1(self, latent2, ladder1, reuse=tf.AUTO_REUSE, is_training=True):
        with tf.variable_scope("generative1", reuse=tf.AUTO_REUSE):
        # with tf.variable_scope("generative1") as gs:
        #     if reuse:
        #         gs.reuse_variables()
            if ladder1 is not None:
                ladder1 = fc_bn_relu(ladder1, int(self.network.fs[2] * self.network.fs[2] * self.network.cs[2]), is_training)
                ladder1 = tf.reshape(ladder1, [-1, self.network.fs[2], self.network.fs[2], self.network.cs[2]])
                if latent2 is not None:
                    latent2 = tf.concat(values=[latent2, ladder1], axis=3)
                else:
                    latent2 = ladder1
            elif latent2 is None:
                print("Generative layer must have input")
                exit(0)
            conv1 = conv2d_t_bn_relu(latent2, self.network.cs[2], [4, 4], 2, is_training)
            conv2 = conv2d_t_bn_relu(conv1, self.network.cs[1], [4, 4], 1, is_training)
            return conv2

    def generative2(self, latent3, ladder2, reuse=tf.AUTO_REUSE, is_training=True):
        with tf.variable_scope("generative2", reuse=tf.AUTO_REUSE):
        # with tf.variable_scope("generative2") as gs:
        #     if reuse:
        #         gs.reuse_variables()
            if ladder2 is not None:
                if latent3 is not None:
                    latent3 = tf.concat(values=[latent3, ladder2], axis=1)
                else:
                    latent3 = ladder2
            elif latent3 is None:
                print("Generative layer must have input")
                exit(0)
            fc1 = fc_bn_relu(latent3, int(self.network.fs[3] * self.network.fs[3] * self.network.cs[3]), is_training)
            fc1 = tf.reshape(fc1, tf.stack([tf.shape(fc1)[0], self.network.fs[3], self.network.fs[3], self.network.cs[3]]))
            conv1 = conv2d_t_bn_relu(fc1, self.network.cs[3], [4, 4], 2, is_training)
            conv2 = conv2d_t_bn_relu(conv1, self.network.cs[2], [4, 4], 1, is_training)
            return conv2

    def generative3(self, latent4, ladder3, reuse=tf.AUTO_REUSE, is_training=True):
        with tf.variable_scope("generative3", reuse=tf.AUTO_REUSE):
        # with tf.variable_scope("generative3") as gs:
        #     if reuse:
        #         gs.reuse_variables()
            fc1 = fc_bn_relu(ladder3, self.network.cs[4], is_training)
            fc2 = fc_bn_relu(fc1, self.network.cs[4], is_training)
            fc3 = fc_bn_relu(fc2, self.network.cs[4], is_training)
            return fc3




#### main.py


# Tested configs
# --dataset=mnist --gpus=2 --denoise_train --plot_reconstruction
# --dataset=mnist --gpus=1 --denoise_train --plot_reconstruction --use_gui
# --dataset=svhn --denoise_train --plot_reconstruction --gpus=2 --db_path=dataset/svhn
# --dataset=celebA --denoise_train --plot_reconstruction --gpus=0 --db_path=/ssd_data/CelebA
# --dataset=mnist --gpus=2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--no_train', type=bool, default=False)
parser.add_argument('--gpus', type=str, default='0')
parser.add_argument('--dataset', type=str, default='svhn')
parser.add_argument('--netname', type=str, default='vladder_svhn')
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--db_path', type=str, default='./dataset/svhn')
parser.add_argument('--reg', type=str, default='kl')
parser.add_argument('--denoise_train', dest='denoise_train', action='store_true',
                    help='Use denoise training by adding Gaussian/salt and pepper noise')
parser.add_argument('--plot_reconstruction', dest='plot_reconstruction', action='store_true',
                    help='Plot reconstruction')
parser.add_argument('--use_gui', dest='use_gui', action='store_true', default=False,
                    help='Display the results with a GUI window')
parser.add_argument('--vis_frequency', type=int, default=100,
                    help='How many train batches before we perform visualization')
args = parser.parse_args()

import matplotlib
if not args.use_gui:
    matplotlib.use('Agg')
else:
    from matplotlib import pyplot as plt
    plt.ion()
    plt.show()

import os

if args.gpus is not '':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

if args.dataset == 'mnist':
    dataset = MnistDataset()
elif args.dataset == 'lsun':
    dataset = LSUNDataset(db_path=args.db_path)
elif args.dataset == 'celebA':
    dataset = CelebADataset(db_path=args.db_path)
elif args.dataset == 'svhn':
    dataset = SVHNDataset(db_path=args.db_path)
else:
    print("Unknown dataset")
    exit(-1)

model = VLadder(dataset, name=args.netname, reg=args.reg, batch_size=args.batch_size, restart=not args.no_train)
trainer = NoisyTrainer(model, dataset, args)
if args.no_train:
    trainer.visualize()
else:
    trainer.train()
