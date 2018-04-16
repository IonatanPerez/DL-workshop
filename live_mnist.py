import numpy as np
import cv2
import tensorflow as tf
import os

import matplotlib.pyplot as plt


MNIST_MODEL = "./trained_models/mnist/"

def load_trained_model():

    sess=tf.Session()    
    #First let's load meta graph and restore weights
    saver = tf.train.import_meta_graph(os.path.join(MNIST_MODEL, "model.ckpt.meta"))
    saver.restore(sess, os.path.join(MNIST_MODEL, "model.ckpt"))

    graph = tf.get_default_graph()

    x = graph.get_tensor_by_name("x:0")
    out = graph.get_tensor_by_name("out/BiasAdd:0")

    return sess, x, out


def run(sess, x, out):
    cap = cv2.VideoCapture(0)
    cap.set(3, 512)
    cap.set(4, 512)

    H, W = 28, 28

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        img_small = cv2.resize(img, (H, W), interpolation = cv2.INTER_AREA) 
        img_big = cv2.resize(img, (10*H, 10*W), interpolation = cv2.INTER_AREA) 

        img_small = 255 - img_small

        ret, _ = cv2.threshold(img_small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        img_small[img_small <= ret] = 0

        img_input = np.reshape(img_small, (1, H, W, 1))
        img_input = img_input / np.max(img_input)

        graph_out, = sess.run([out], feed_dict={x: img_input})

        char = np.argmax(np.squeeze(graph_out))

        cv2.putText(img_big, str(char), (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))

        cv2.imshow("Frame", img_big)
        cv2.imshow("Network input", img_small)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    sess, x, out = load_trained_model()
    run(sess, x, out)
