import numpy as np
import cv2
import tensorflow as tf
import os
import sys

import matplotlib.pyplot as plt
import time


MNIST_MODEL = "./trained_models/simpsons/"
H, W = 128, 128


def load_trained_model():

    sess=tf.Session()    
    #First let's load meta graph and restore weights
    saver = tf.train.import_meta_graph(os.path.join(MNIST_MODEL, "model.ckpt.meta"))
    saver.restore(sess, os.path.join(MNIST_MODEL, "model.ckpt"))

    graph = tf.get_default_graph()

    x = graph.get_tensor_by_name("x:0")
    out = graph.get_tensor_by_name("out/BiasAdd:0")

    return sess, x, out


def softmax(_in):
    return np.exp(_in) / np.sum(np.exp(_in))


def classify_character(img, sess, x, out):
    img_h, img_w, _ = img.shape

    img = cv2.reshape(img, (H, W))

    graph_out, p = sess.run(out, feed_dict={x: np.reshape(img, (1, H, W))})

    # TODO -> find character

    cv2.putText(img, str(character), (0, 0), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))

    return img


def run(vid, sess, x, out):

    FPS = 24

    cap = cv2.VideoCapture(vid)

    cv2.namedWindow("Frame", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Frame",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

    now = time.time()
    while(True):
        ret, frame = cap.read()

        h, w, c = frame.shape

        img = classify_character(frame, sess, x, out)

        cv2.imshow("Frame", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        interval = time.time() - now
        if interval < 1/FPS:
            time.sleep(1/FPS - interval)
        now = time.time()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    vid = sys.argv[1]
    # sess, x, out = load_trained_model()
    sess, x, out = None, None, None
    run(vid, sess, x, out)
