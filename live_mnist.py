import numpy as np
import cv2
import tensorflow as tf
import os
import sys

import matplotlib.pyplot as plt

# Disable tensorflow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

MNIST_MODEL = "./trained_models/mnist/"

H, W = 28, 28
EPSILON = 0.25
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

    digit_roi = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if cv2.contourArea(cnt) > MIN_AREA:
            [x,y,w,h] = cv2.boundingRect(cnt)

            if w > h*0.3 and w < h:

                eps_h = int(EPSILON * h)
                eps_w = int(EPSILON * w)
                roi = gray[y: y + h, x: x + w]

                digit, prob, digit_roi = reader(roi)

                if prob > 0.1:
                    cv2.putText(img_copy, str(digit), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0))
                    cv2.rectangle(img_copy, (x,y), (x+w,y+h), (0, 0, 255), 2)


    return img_copy, digit_roi

def softmax(_in):
    return np.exp(_in) / np.sum(np.exp(_in))

def reader(img, sess, x, out):
    img = img.copy()
    img = 255 - img

    img_h, img_w = img.shape

    ret, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img[img <= ret] = 0

    img = (img - np.min(img)) / (np.max(img) - np.min(img))

    if img_h > img_w:
        pad = img_h - img_w
        img = np.pad(img, ((0,0), (pad//2, pad//2)), 'constant', constant_values=0)
    else:
        pad = img_w - img_h
        img = np.pad(img, ((pad//2 , pad//2), (0,0)), 'constant', constant_values=0)

    img_h, img_w = img.shape

    pad_w = int(EPSILON * img_w)
    pad_h = int(EPSILON * img_h)

    img = np.pad(img, ((pad_h , pad_h), (pad_w,pad_w)), 'constant', constant_values=0)

    img = cv2.resize(img, (H, W), cv2.INTER_NEAREST)

    img_input = np.reshape(img, (1, H, W, 1))

    graph_out, = sess.run([out], feed_dict={x: img_input})

    graph_out = np.squeeze(graph_out)

    char = np.argmax(graph_out)
    prob = max(softmax(graph_out))

    return char, prob, img


def run(i, sess, x, out):
    cap = cv2.VideoCapture(i)

    cv2.namedWindow("Frame", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Frame",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

    while(True):
        ret, frame = cap.read()

        h, w, c = frame.shape

        notated_img, _ = find_digits(frame, lambda img: reader(img, sess, x, out))

        cv2.imshow("Frame", notated_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def test_camera():
    cameras = []
    for i in range(3):
        cap = cv2.VideoCapture(i)
        ret, frame = cap.read()
        try:
            h, w, c = frame.shape
        except AttributeError:
            continue
        cap.release()
        cameras.append(i)

    return cameras

if __name__ == "__main__":
    cameras = test_camera()
    if len(cameras) == 0:
        print("Webcam not detected.")
        sys.exit(1)

    if len(cameras) == 1:
        print("Using webcam # {}".format(cameras[0]))
        i = cameras[0]
    else:
        i = int(input("{} cameras detected. Provide an index of the one to use ({} to {}): ".format(len(cameras), 0, len(cameras)-1)))

    sess, x, out = load_trained_model()
    run(i, sess, x, out)
