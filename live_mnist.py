import numpy as np
import cv2
import tensorflow as tf
import os

import matplotlib.pyplot as plt


MNIST_MODEL = "./trained_models/mnist/"
H, W = 28, 28

EPSILON = 0.1

MIN_AREA = 750

def load_trained_model():

    sess=tf.Session()    
    #First let's load meta graph and restore weights
    saver = tf.train.import_meta_graph(os.path.join(MNIST_MODEL, "model.ckpt.meta"))
    saver.restore(sess, os.path.join(MNIST_MODEL, "model.ckpt"))

    graph = tf.get_default_graph()

    x = graph.get_tensor_by_name("x:0")
    out = graph.get_tensor_by_name("out/BiasAdd:0")

    return sess, x, out


def find_digits(img, reader):
    img_copy = img.copy()

    img_h, img_w, _ = img_copy.shape

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (15, 15), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 15, 2)

    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if cv2.contourArea(cnt) > MIN_AREA:
            [x,y,w,h] = cv2.boundingRect(cnt)

            if w > h*0.3 and w < h:

                eps_h = int(EPSILON * h)
                eps_w = int(EPSILON * w)
                roi = gray[max(y - eps_h, 0): min(y + h + eps_h, img_h - 1), max(x - eps_w, 0): min(x + w + eps_w, img_w - 1)]
                digit = cv2.resize(roi, (H, W), interpolation = cv2.INTER_AREA)

                digit, prob = reader(digit)

                if prob > 0.5:
                    cv2.putText(img_copy, str(digit), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))
                    cv2.rectangle(img_copy, (x,y), (x+w,y+h), (0, 0, 255), 2)

    return img_copy

def softmax(_in):
    return np.exp(_in) / np.sum(np.exp(_in))

def reader(img, sess, x, out):
    img = img.copy()
    img = 255 - img

    ret, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    img[img <= ret] = 0

    img_input = np.reshape(img, (1, H, W, 1))
    img_input = img_input / np.max(img_input)

    graph_out, = sess.run([out], feed_dict={x: img_input})

    graph_out = np.squeeze(graph_out)

    char = np.argmax(graph_out)
    prob = max(softmax(graph_out))

    return char, prob


def run(sess, x, out):
    cap = cv2.VideoCapture(0)

    cv2.namedWindow("Frame", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Frame",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

    while(True):
        ret, frame = cap.read()

        h, w, c = frame.shape

        notated_img = find_digits(frame, lambda img: reader(img, sess, x, out))

        cv2.imshow("Frame", notated_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    sess, x, out = load_trained_model()
    run(sess, x, out)
