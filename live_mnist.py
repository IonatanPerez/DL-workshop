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

    H, W = 28, 28
    cv2.namedWindow("Frame", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Frame",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

    cv2.namedWindow("Network input", cv2.WINDOW_NORMAL)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        h, w, c = frame.shape

        # Set squared frame
        margin = (w - h) // 2
        frame = frame[:, margin : h + margin, :]

        # Our operations on the frame come here
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        img = cv2.resize(img, (H, W), interpolation = cv2.INTER_AREA) 

        img = 255 - img

        ret, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        img[img <= ret] = 0

        img_input = np.reshape(img, (1, H, W, 1))
        img_input = img_input / np.max(img_input)

        graph_out, = sess.run([out], feed_dict={x: img_input})

        char = np.argmax(np.squeeze(graph_out))

        cv2.putText(frame, "Detected: {}".format(str(char)), (0, h-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))

        cv2.imshow("Frame", frame)
        cv2.imshow("Network input", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    sess, x, out = load_trained_model()
    run(sess, x, out)
