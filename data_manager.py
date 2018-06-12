# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import scipy
import cv2
import os


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/home/pkhosla/MNIST_data/", one_hot=True)

def center_img(img):
    maxval = np.max(img)
    minval = np.min(img)
    img = (img - np.ones(img.shape)*minval) / maxval
    return img

def cast_to_32(img):
    mode = 0
    if (len(list(img.shape)) == 2):
        mode = 1
        img = img.reshape([1] + list(img.shape))
    t = np.zeros([img.shape[0],32,32])
    off1 = (32 - img.shape[1]) // 2
    off2 = (32 - img.shape[2]) // 2
    t[:,off1:off1+img.shape[1],off2:off2+img.shape[2]] = img
    if mode == 1:
        t = t.reshape((32,32))
    return t

def cast_to_64(img, translate=[0,0], rotate=None, resize=None):
    if rotate != None :
        img = rotate_at_angle(img, rotate)
    if resize != None and resize > 0 :
        img = resize_to_scale(img, resize)
    mode = 0
    if (len(list(img.shape)) == 2):
        mode = 1
        img = img.reshape([1] + list(img.shape))
    t = np.zeros([img.shape[0],64,64])
    off1 = int(translate[0]) # (32 - img.shape[1]) // 2
    off2 = int(translate[1]) # (32 - img.shape[2]) // 2
    t[:,off1:off1+img.shape[1],off2:off2+img.shape[2]] = img
    if mode == 1:
        t = t.reshape((64,64))
    return t

def resize_to_scale(img,size):
    mode = 0
    if (len(list(img.shape)) == 2):
        mode = 1
        img = img.reshape([1] + list(img.shape))
    if int(size) != 0 :
        size_dims = map(lambda x : x // int(size), img.shape[1:])
        t = np.array(map(lambda x : cast_to_32(cv2.resize(x , dsize=tuple(size_dims))), img))
    if mode == 1:
        t = t.reshape((32,32))
    return t

def rotate_at_angle(img,angle):
    mode = 0
    if (len(list(img.shape)) == 2):
        mode = 1
        img = img.reshape([1] + list(img.shape))
    t = np.array(map(lambda x : center_img(scipy.misc.imrotate(x, angle)), img))
    if mode == 1:
        t = t.reshape((32,32))
    return t

class DataManager(object):
  def load(self):
    # Load dataset
    dataset_zip = np.load('/home/pkhosla/data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz',
                          encoding = 'latin1')

    # print('Keys in the dataset:', dataset_zip.keys())
    #  ['metadata', 'imgs', 'latents_classes', 'latents_values']
    print("Setting variables")
    self.imgs       = dataset_zip['imgs']
    self.latents_values  = dataset_zip['latents_values']
    self.latents_classes = dataset_zip['latents_classes']
    metadata        = dataset_zip['metadata'][()]

    # Define number of values per latents and functions to convert to indices
    latents_sizes = metadata['latents_sizes']
    # [ 1,  3,  6, 40, 32, 32]
    # color, shape, scale, orientation, posX, posY

    self.n_samples = latents_sizes[::-1].cumprod()[-1]
    # 737280

    self.latents_bases = np.concatenate((latents_sizes[::-1].cumprod()[::-1][1:],
                                         np.array([1,])))
    self.info = dict(
                capacity=int(.5 * len(self.imgs)),
                min_after_dequeue = int(.2 * len(self.imgs)),
                n_files=len(self.imgs)
            )
    # [737280, 245760, 40960, 1024, 32, 1]

  @property
  def sample_size(self):
    return self.n_samples

  def get_image(self, shape=0, scale=0, orientation=0, x=0, y=0):
    latents = [0, shape, scale, orientation, x, y]
    index = np.dot(latents, self.latents_bases).astype(int)
    return self.get_images([index])[0]

  def get_images(self, indices):
    images = []
    for index in indices:
      img = self.imgs[index]
      img = img.reshape(4096)
      images.append(img)
    return np.array(images)

  def get_classes(self, indices):
    labels = []
    for index in indices:
      lab = self.latents_classes[index]
      labels.append(lab)
    return labels


  def get_random_images_wl(self, size): # return images with labels
    indices = [np.random.randint(self.n_samples) for i in range(size)]
    return self.get_images(indices), self.get_classes(indices)

  def get_random_images(self, size):
    indices = [np.random.randint(self.n_samples) for i in range(size)]
    return self.get_images(indices)

  def randomly_modify_labels(self, label_initial):
      label_set = np.zeros([label_initial.shape[0],4])
      label_initial = np.copy(label_final)
      for i in range(label_final.shape[0]):
          rand = np.random.randint(1,5)
          if rand == 1 :
            scale = np.random.randint(0,6)
            label_set[0] = 1
            label_final[1] = 0.5 + (0.1*scale)
          elif rand == 2:
            orientation = np.random.randint(1,5)
            label_set[1] = 1
            label_final[2] = (np.pi*orientation*2.0) / (40.0)
          elif rand == 3 :
            x = np.random.randint(0,33) / 32.0
            label_final[3] = x
            label_set[2] = 1
          else :
            y = np.random.randint(0,33) / 32.0
            label_final[4] = y
            label_set[3] = 1
      return label_final, label_set

  def sample_batch(self, batch_size, train=True):
      image_initial, label_initial = self.get_random_images_wl(self, batch_size)
      image_initial = np.expand_dims(image_initial, axis=-1)
      label_final, label_set = self.randomly_modify_labels(label_initial)
      image_final = np.expand_dims(np.array(map(lambda x : self.get_image(scale=t[1], orientation=t[2], x=t[3], y=t[4]), label_final)), axis=-1)
      return np.concatenate([image_initial, image_final], axis=-1) , label_set

class many_example_random_pair(object):
    """
    Perform random operations on them
    During training, predict the difference in labels given a random image pair
    Better in general but might be over-reaching
    """
    def load(self):
        train_image_list = mnist.train.images.reshape([len(mnist.train.images), 28,28])
        train_image_list = cast_to_32(train_image_list)
        train_label_list = mnist.train.labels
        test_image_list = mnist.test.images.reshape([len(mnist.test.images), 28,28])
        test_image_list = cast_to_32(test_image_list)
        test_label_list = mnist.test.labels
        self.datadir = "/home/pkhosla/data"
        self.train_image_list, self.train_label_list = self._create_set(train_image_list, train_label_list)
        self.test_image_list, self.test_label_list = self._create_set(test_image_list, test_label_list, train=False)
        self.train_sample_size = self.train_image_list.shape[0]
        self.test_sample_size = self.test_image_list.shape[0]
        self.info = dict(
                capacity=int(.5 * len(self.train_image_list)),
                min_after_dequeue = int(.2 * len(self.train_image_list)),
                n_files=len(self.train_image_list)
            )

    def _create_set(self, train_image_list, train_label_list, train=True):
        if train :
            if os.path.exists(os.path.join(self.datadir, "train_data_image.npz")):
                train_image_list = np.load(os.path.join(self.datadir, "train_data_image.npz"))
                train_label_list = np.load(os.path.join(self.datadir, "train_data_label.npz"))
                return train_image_list["arr_0"], train_label_list["arr_0"]
        else :
            if os.path.exists(os.path.join(self.datadir, "test_data_image.npz")):
                test_image_list = np.load(os.path.join(self.datadir, "test_data_image.npz"))
                test_label_list = np.load(os.path.join(self.datadir, "test_data_label.npz"))
                return test_image_list["arr_0"], test_label_list["arr_0"]
        train_image_list, train_label_list = self.rotate_randomly(train_image_list, train_label_list, target_size=2*train_image_list.shape[0])
        print(train_image_list.shape)
        train_image_list, train_label_list = self.resize_randomly(train_image_list, train_label_list, target_size=2*train_image_list.shape[0])
        print(train_image_list.shape)
        train_image_list, train_label_list = self.translate_randomly(train_image_list, train_label_list, target_size=2*train_image_list.shape[0])
        print(train_image_list.shape)
        if train :
            np.savez_compressed(os.path.join(self.datadir, "train_data_image.npz"), train_image_list)
            np.savez_compressed(os.path.join(self.datadir, "train_data_label.npz"), train_label_list)
        else:
            np.savez_compressed(os.path.join(self.datadir, "test_data_image.npz"), train_image_list)
            np.savez_compressed(os.path.join(self.datadir, "test_data_label.npz"), train_label_list)
        return train_image_list, train_label_list

    def translate_randomly(self, image_list, label_list, target_size=100):
        label_list_target = np.zeros([target_size] + [label_list.shape[-1]+2])
        image_list_target = np.zeros([target_size] + map(lambda x : x*2,list(image_list.shape[1:])))
        label_list_target[:image_list.shape[0], :label_list.shape[-1]] = label_list
        image_list_target[:image_list.shape[0], :image_list.shape[1], :image_list.shape[2]] = image_list
        copy_index=image_list.shape[0]
        while (copy_index < target_size):
            print(copy_index, end="\r")
            if copy_index % 10000 == 0 :
                print()
            index = np.random.randint(0, image_list.shape[0])
            translate_1 = np.random.randint(0, image_list.shape[1])
            translate_2 = np.random.randint(0, image_list.shape[2])
            image_list_target[copy_index][translate_1:translate_1+image_list.shape[1], translate_2:translate_2+image_list.shape[2]] = np.copy(image_list[index])
            label_list_target[copy_index][:label_list.shape[-1]] = np.copy(label_list[index])
            label_list_target[copy_index][-2] = translate_1
            label_list_target[copy_index][-1] = translate_2
            copy_index+=1
        return image_list_target, label_list_target
    def resize_randomly(self, image_list, label_list, target_size=100):
        label_list_target = np.zeros([target_size] + list(label_list.shape[1:-1]) + [label_list.shape[-1]+1])
        image_list_target = np.zeros([target_size] + list(image_list.shape[1:]))
        label_list_target[:image_list.shape[0], :label_list.shape[-1]] = np.copy(label_list)
        image_list_target[:image_list.shape[0]] = np.copy(image_list)
        copy_index=image_list.shape[0]
        while (copy_index < target_size):
            index = np.random.randint(0, image_list.shape[0])
            size = np.random.randint(1, 5)
            size_dims = map(lambda x : x // size, image_list.shape[1:])
            image_list_target[copy_index] = cast_to_32(cv2.resize(image_list[index], dsize=tuple(size_dims)))
            label_list_target[copy_index][:label_list.shape[-1]] = np.copy(label_list[index])
            label_list_target[copy_index][-1] = size
            copy_index+=1
        return image_list_target, label_list_target
    def rotate_randomly(self, image_list, label_list, target_size=100):
        label_list_target = np.zeros([target_size] + list(label_list.shape[1:-1]) + [label_list.shape[-1]+1])
        image_list_target = np.zeros([target_size] + list(image_list.shape[1:]))
        label_list_target[:image_list.shape[0], :label_list.shape[-1]] = np.copy(label_list)
        image_list_target[:image_list.shape[0]] = np.copy(image_list)
        copy_index=image_list.shape[0]
        while (copy_index < target_size):
            index = np.random.randint(0, image_list.shape[0])
            angle = np.random.randint(0, 360)
            image_list_target[copy_index] = center_img(scipy.misc.imrotate(image_list[index], angle))
            label_list_target[copy_index][:label_list.shape[-1]] = np.copy(label_list[index])
            label_list_target[copy_index][-1] = angle
            copy_index+=1
        return image_list_target, label_list_target
    @property
    def sample_size(self):
        return self.train_sample_size

    def get_images(self, indices, train=True):
        if train :
            self.base_list = self.train_image_list
        images = []
        for index in indices:
            img = self.base_list[index]
            img = img.reshape(4096)
            images.append(img)
        return np.array(images)

    def get_classes(self, indices, train=True):
        if train :
            self.base_label_list = self.train_image_list
        labels = []
        for index in indices:
            lab = self.base_label_list[index]
            labels.append(lab)
        return labels


    def get_random_images_wl(self, size): # return images with labels
        indices = [np.random.randint(self.sample_size) for i in range(size)]
        return self.get_images(indices), self.get_classes(indices)

    def get_random_images(self, size=64):
        indices = [np.random.randint(self.sample_size) for i in range(size)]
        return np.array(self.get_images(indices))

class single_factor_changed_one_example_generated_pair(object):
    """
    Complete MNIST data
    During training, choose an image, apply random transformation and give it to the algorithm to predict
    Much better for axis-walk only
    """
    def __init__(self, train=7, val=2, test=1):
        perm = np.random.permutation([i for i in range(10)])
        train_index_list = []
        test_index_list = []
        val_index_list = []
        for i in range(10):
            for j in range(mnist.train.images.shape[0]):
                if (mnist.train.labels[j][perm[i]] == 1):
                    if i < train :
                        train_index_list.append(j)
                    elif i < train + test :
                        test_index_list.append(j)
                    else :
                        val_index_list.append(j)
                    break
        self. base_list = mnist.train.images
        train_image_list = np.array(map(lambda x : mnist.train.images[x], train_index_list)).reshape([len(train_index_list), 28,28])
        train_image_list = cast_to_32(train_image_list)
        train_label_list = np.array(map(lambda x : np.append(mnist.train.labels[x],x), train_index_list))
        test_image_list = np.array(map(lambda x : mnist.test.images[x], test_index_list)).reshape([len(test_index_list), 28,28])
        test_image_list = cast_to_32(test_image_list)
        test_label_list = np.array(map(lambda x : np.append(mnist.test.labels[x], x), test_index_list))
        self.train_image_list, self.train_label_list = self._create_set(train_image_list, train_label_list)
        self.test_image_list, self.test_label_list = self._create_set(test_image_list, test_label_list)
        self.train_sample_size = self.train_image_list.shape[0]
        self.test_sample_size = self.test_image_list.shape[0]

    def _create_set(self, train_image_list, train_label_list):
        train_image_list, train_label_list = self.rotate_randomly(train_image_list, train_label_list, target_size=20*train_image_list.shape[0])
        train_image_list, train_label_list = self.resize_randomly(train_image_list, train_label_list, target_size=20*train_image_list.shape[0])
        train_image_list, train_label_list = self.translate_randomly(train_image_list, train_label_list, target_size=20*train_image_list.shape[0])
        perm = np.random.permutation(zip(train_image_list, train_label_list))
        train_image_list, train_label_list = np.array(map(lambda x : x[0], perm)),np.array(map(lambda x : x[1], perm))
        return train_image_list, train_label_list

    @property
    def sample_size(self):
        return self.train_sample_size
    def translate_randomly(self, image_list, label_list, target_size=100, x_or_y=None, rotate=0, resize=0):
        base_list = self.base_list
        if image_list.shape[1] == 64 :
            image_list = np.array(map(lambda x : cast_to_32(base_list[int(x[10])].reshape((28,28))), label_list))
            if rotate != 0 :
                image_list = np.array(map(lambda x : rotate_at_angle(x, rotate), image_list))
            if resize != 0 :
                image_list = np.array(map(lambda x : resize_to_scale(x, resize), image_list))
        label_list_target = np.zeros([target_size] + [label_list.shape[-1]+2])
        image_list_target = np.zeros([target_size] + map(lambda x : x*2,list(image_list.shape[1:])))
        copy_index=0 #image_list.shape[0]
        while (copy_index < target_size):
            index = np.random.randint(0, image_list.shape[0])
            translate_1 = np.random.randint(0, image_list.shape[1])
            translate_2 = np.random.randint(0, image_list.shape[2])
            if x_or_y != None :
                if x_or_y == "x" :
                    translate_2 = int(label_list[index,-1])
                elif x_or_y == "y":
                    translate_1 = int(label_list[index, -2])
                else :
                    print("Invalid translate direction")
                    exit(1)
            image_list_target[copy_index][translate_1:translate_1+image_list.shape[1], translate_2:translate_2+image_list.shape[2]] = np.copy(image_list[index])
            label_list_target[copy_index][:label_list.shape[-1]] = np.copy(label_list[index])
            label_list_target[copy_index][-2] = translate_1
            label_list_target[copy_index][-1] = translate_2
            copy_index+=1
        return image_list_target, label_list_target

    def resize_randomly(self, image_list, label_list, target_size=100):
        base_list = self.base_list
        if image_list.shape[1] == 64 :
            image_list = np.array(map(lambda x : cast_to_32(base_list[int(x[10])].reshape((28,28))), label_list))
        label_list_target = np.zeros([target_size] + list(label_list.shape[1:-1]) + [label_list.shape[-1]+1])
        image_list_target = np.zeros([target_size] + list(image_list.shape[1:]))
        copy_index=0 #image_list.shape[0]
        while (copy_index < target_size):
            index = np.random.randint(0, image_list.shape[0])
            size = np.random.randint(1, 5)
            size_dims = map(lambda x : x // size, image_list.shape[1:])
            image_list_target[copy_index] = cast_to_32(cv2.resize(image_list[index], dsize=tuple(size_dims)))
            label_list_target[copy_index][:label_list.shape[-1]] = np.copy(label_list[index])
            label_list_target[copy_index][-1] = size
            copy_index+=1
        return image_list_target, label_list_target

    def rotate_randomly(self, image_list, label_list, target_size=100):
        base_list = self.base_list
        if image_list.shape[1] == 64 :
            image_list = np.array(map(lambda x : cast_to_32(base_list[int(x[10])].reshape((28,28))), label_list))
        label_list_target = np.zeros([target_size] + list(label_list.shape[1:-1]) + [label_list.shape[-1]+1])
        image_list_target = np.zeros([target_size] + list(image_list.shape[1:]))
        copy_index=0 # image_list.shape[0]
        while (copy_index < target_size):
            index = np.random.randint(0, image_list.shape[0])
            angle = np.random.randint(0, 360)
            image_list_target[copy_index] = center_img(scipy.misc.imrotate(image_list[index], angle))
            label_list_target[copy_index][:label_list.shape[-1]] = np.copy(label_list[index])
            label_list_target[copy_index][-1] = angle
            copy_index+=1
        return image_list_target, label_list_target

    def sample_batch(self, batch_size=64, train=True):
        if train :
            perm = [i for i in range(self.train_image_list.shape[0])]
        else :
            perm = [i for i in range(self.test_image_list.shape[0])]
        for i in range(7): # most random permutation achieved in 7 shuffles
            perm = np.random.permutation(perm)
        curr_index = 0
        if train :
            self.base_list = mnist.train.images
            return self.get_sample(self.train_image_list, self.train_label_list, perm[:batch_size]) # image_batch, label_batch
        else :
            self.base_list = mnist.test.images
            return self.get_sample(self.test_image_list, self.test_label_list, perm[:batch_size]) # image_batch, label_batch

    def random_change(self, img, label):
        img = img.reshape([1]  + list(img.shape))
        label = label.reshape([1]  + list(label.shape))
        label_old = np.copy(label)
        label_set = np.zeros([5])
        change_type = np.random.randint(1,5) # change_type =  1 (rotation), 2(translation_x), 3 (translation_y) , 4 (rotation)
        if change_type == 2 :
            img, label = self.translate_randomly(img, label, target_size=1, x_or_y="x", rotate=label[0][-4], resize=label[0][-3] ) # need to ensure other generative factors are constant
            label_set[1] = np.ones(label_set[1].shape) #label[0][-2:] - label_old[0][-2:]  # translation is given in the last 2 co-ordinates of the labels, every transformation extends the label space by required dimensions
        elif change_type == 3 :
            img, label = self.translate_randomly(img, label, target_size=1, x_or_y="y", rotate=label[0][-4], resize=label[0][-3])
            label_set[2] = np.ones(label_set[2].shape) #label[0][-2:] - label_old[0][-2:]
        elif change_type == 1 :
            img, label = self.resize_randomly(img, label, target_size=1)
            img = cast_to_64(img.reshape([32,32]), translate = label_old[0][-2:], rotate=label[0][-4])
            label_set[0] = np.ones(label_set[0].shape) #label[0][-1] - label_old[0][-3]
        elif change_type == 4 :
            img, label = self.rotate_randomly(img, label, target_size=1)
            img = cast_to_64(img.reshape([32,32]), translate=label_old[0][-2:], resize=label[0][-3]) # all functions are made for lists, so in order to work with single images they have to be cast accordingly
            label_set[3] = np.ones(label_set[3].shape) # label[0][-1] - label_old[0][-4]
        return img.reshape([64,64]), label_set


    def get_sample(self, image_list, label_list, index_list):
        image_batch = np.zeros([len(index_list)] + map(lambda x : x, list(self.train_image_list.shape[1:])) + [2])
        label_batch = np.zeros([len(index_list)] + [5])  # list(self.train_label_list.shape[1:]))
        curr_index = 0
        for i in index_list:
            image_batch[curr_index,:,:,0] = image_list[i]
            x = list(self.random_change(image_list[i], label_list[i]))
            image_batch[curr_index,:,:,1], label_batch[curr_index] = x[0], x[1] #self.random_change(image_list[i], label_list[i])
            curr_index += 1
        return image_batch.reshape(list(image_batch.shape)), label_batch

if __name__ == "__main__" :
    example = one_example_random_pair()

