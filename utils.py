import os
from skimage import io, transform
import numpy as np
import glob
import config as cfg
from tqdm import tqdm
import matplotlib.pyplot as plt


def load_characters(min_imgs=800):
    i = 0
    map_characters = {}
    for char in os.listdir(cfg.DATA_DIR):
        full_path = os.path.join(cfg.DATA_DIR, char)
        if os.path.isfile(full_path) or char[0] == '.':
            continue

        pictures = glob.glob(os.path.join(full_path, "*.jpg"))
        if len(pictures) < min_imgs:
            continue

        map_characters[i] = char
        i+=1

    return map_characters


def load_pictures(map_characters, max_per_classs=None):
    pics = []
    labels = []
    for l, c in map_characters.items():
        pictures = glob.glob(os.path.join(cfg.DATA_DIR, "{}".format(c), "*"))
        for idx in tqdm(range(len(pictures)), desc="Loading {}".format(c)):

            if max_per_classs is not None and idx >= max_per_classs:
                break

            img = io.imread(pictures[idx])
            img = transform.resize(img, (cfg.IMG_SIZE, cfg.IMG_SIZE), mode='constant', preserve_range=True).astype(np.uint8)
            img = img.astype(np.uint8)
            pics.append(img)
            labels.append(l)
    return np.array(pics, dtype=np.uint8), np.array(labels, dtype=np.uint8) 


def show_random(pics, labels, map_characters):
    N, _, _, _ = pics.shape
    fig=plt.figure(figsize=(20, 20))
    columns = 3
    rows = 3
    for i in range(1, columns*rows +1):
        idx = np.random.choice(range(N)) 
        img = pics[idx]
        character = map_characters[labels[idx]]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
        plt.title(character)
    plt.show()


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

