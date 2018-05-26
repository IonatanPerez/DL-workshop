import os
from skimage import io, transform
import numpy as np
import glob
import config as cfg
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf

#########################################################
#                       Generic                         #
#########################################################

def shuffle_dataset(x, y):
    N = x.shape[0]
    idxs = np.arange(N)
    np.random.shuffle(idxs)
    return x[idxs], y[idxs]


def get_one_hot_encoding(labels):
    N, = labels.shape
    M = len(np.unique(labels))
    
    ret = np.zeros((N, M))
    ret[np.arange(len(labels)), labels] = 1

    return ret

def split(pics, labels, p):
    N = pics.shape[0]
    N_train = int(p*N)

    idxs = np.arange(N)

    idxs_train = np.random.choice(idxs, N_train, replace=False)
    idxs_val = np.setdiff1d(idxs, idxs_train, assume_unique=True)

    return pics[idxs_train], labels[idxs_train], pics[idxs_val], labels[idxs_val]    


def get_small_dataset(x, y, p=0.1):
    N = x.shape[0]

    new_N = int(p * N)

    idxs = np.arange(N)
    idxs = np.random.choice(idxs, new_N, replace=False)

    new_x = x[idxs]
    new_y = y[idxs]

    return new_x, new_y

#########################################################
#             Simpsons Problem Specific                 #
#########################################################

def load_characters(data_dir, min_imgs=800):
    i = 0
    map_characters = {}
    for char in os.listdir(data_dir):
        full_path = os.path.join(data_dir, char)
        if os.path.isfile(full_path) or char[0] == '.':
            continue

        pictures = glob.glob(os.path.join(full_path, "*.jpg"))
        if len(pictures) < min_imgs:
            continue

        map_characters[i] = char
        i+=1

    return map_characters


def load_pictures(data_dir, map_characters, max_per_classs=None):
    pics = []
    labels = []
    for l, c in map_characters.items():
        pictures = glob.glob(os.path.join(data_dir, "{}".format(c), "*"))
        for idx in tqdm(range(len(pictures)), desc="Loading {}".format(c)):

            if max_per_classs is not None and idx >= max_per_classs:
                break

            img = io.imread(pictures[idx])
            img = transform.resize(img, (cfg.IMG_SIZE, cfg.IMG_SIZE), mode='constant', preserve_range=True).astype(np.uint8)
            img = img.astype(np.uint8)
            pics.append(img)
            labels.append(l)
    pics = np.array(pics, dtype=np.uint8)
    labels = get_one_hot_encoding(np.array(labels, dtype=np.uint8))

    return shuffle_dataset(pics, labels)


def show_random_characters(pics, labels, map_characters):
    N, _, _, _ = pics.shape
    fig=plt.figure(figsize=(20, 20))
    columns = 3
    rows = 3
    for i in range(1, columns*rows +1):
        idx = np.random.choice(range(N)) 
        img = pics[idx]
        character = map_characters[np.argmax(labels[idx])]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
        plt.title(character)
    plt.show()


#########################################################
#                MNIST Problem Specific                 #
#########################################################


def load_mnist():
    H, W = 28, 28

    mnist = tf.contrib.learn.datasets.load_dataset("mnist")

    pics_train = mnist.train.images
    pics_train = pics_train.reshape((-1, H, W, 1))

    labels_train = get_one_hot_encoding(np.asarray(mnist.train.labels, dtype=np.int32))

    pics_test = mnist.test.images
    pics_test = pics_test.reshape((-1, H, W, 1))

    labels_test = get_one_hot_encoding(np.asarray(mnist.test.labels, dtype=np.int32))

    return pics_train, labels_train, pics_test, labels_test


def show_random_mnist(pics, labels):
    N, _, _, _  = pics.shape
    fig=plt.figure(figsize=(10, 10))
    columns = 3
    rows = 3
    for i in range(1, columns*rows +1):
        idx = np.random.choice(range(N)) 
        img = pics[idx]
        fig.add_subplot(rows, columns, i)
        plt.imshow(np.squeeze(img), cmap='gray')
        plt.axis('off')
        plt.title(np.argmax(labels[idx]))
    plt.show()
