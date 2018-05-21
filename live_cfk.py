import numpy as np
import cv2
import tensorflow as tf
import os
import pickle


CFK_MODEL = "./trained_models/lstm_cfk/"

def load_trained_model():

    sess=tf.Session()    
    #First let's load meta graph and restore weights
    saver = tf.train.import_meta_graph(os.path.join(CFK_MODEL, "model.ckpt.meta"))
    saver.restore(sess, os.path.join(CFK_MODEL, "model.ckpt"))

    graph = tf.get_default_graph()

    x = graph.get_tensor_by_name("x:0")

    seq_len = graph.get_tensor_by_name("seq_len:0")

    init_state_c_1 = graph.get_tensor_by_name("init_state_c_1:0")
    init_state_h_1 = graph.get_tensor_by_name("init_state_h_1:0")

    init_state_c_2 = graph.get_tensor_by_name("init_state_c_2:0")
    init_state_h_2 = graph.get_tensor_by_name("init_state_h_2:0")

    states_0_c = graph.get_tensor_by_name("states_0_c:0")
    states_0_h = graph.get_tensor_by_name("states_0_h:0")

    states_1_c = graph.get_tensor_by_name("states_1_c:0")
    states_1_h = graph.get_tensor_by_name("states_1_h:0")

    out = graph.get_tensor_by_name("out/BiasAdd:0")
    out_softmax = graph.get_tensor_by_name("out_softmax:0")
    
    model =  x, seq_len, init_state_c_1, init_state_h_1, init_state_c_2, init_state_h_2, states_0_c, states_0_h, states_1_c, states_1_h, out, out_softmax

    return sess, model


def load_vocab():
    return pickle.load(open("./vocab.dump", "rb"))


def write_inline(char):
    print(char, end="", flush=True)


def run(sess, model, vocab, max_=1000, T=None, f=write_inline, n_hidden=512, times=32):
    char_to_ix, ix_to_char = vocab
    input_size = len(ix_to_char)

    x, seq_len, init_state_c_1, init_state_h_1, init_state_c_2, init_state_h_2, states_0_c, states_0_h, states_1_c, states_1_h, out, out_softmax = model

    states_0 = tf.nn.rnn_cell.LSTMStateTuple(states_0_c, states_0_h)
    states_1 = tf.nn.rnn_cell.LSTMStateTuple(states_1_c, states_1_h)

    states = (states_0, states_1)


    pred = "<START>"
    
    c_1 = np.zeros((1, n_hidden))
    h_1 = np.zeros((1, n_hidden))
    
    c_2 = np.zeros((1, n_hidden))
    h_2 = np.zeros((1, n_hidden))
    
    i = 0
    while True:
        
        in_ = np.zeros(shape=(1, times, input_size), dtype=np.uint)
        in_[0, 0, char_to_ix[pred]] = 1

        if T is None:
            net_out, net_states = sess.run([out_softmax, states], feed_dict={x: in_, init_state_c_1: c_1, init_state_h_1: h_1, init_state_c_2: c_2, init_state_h_2: h_2, seq_len: np.ones(shape=(1,))})
            c_1, h_1 = net_states[0].c, net_states[0].h
            c_2, h_2 = net_states[1].c, net_states[1].h
            p = np.squeeze(net_out)[0]
        else:
            net_out, net_states = sess.run([out, states], feed_dict={x: in_, init_state_c_1: c_1, init_state_h_1: h_1, init_state_c_2: c_2, init_state_h_2: h_2, seq_len: np.ones(shape=(1,))})
            c_1, h_1 = net_states[0].c, net_states[0].h
            c_2, h_2 = net_states[1].c, net_states[1].h
            p = np.squeeze(net_out)[0]
            p = np.exp(p/T) / np.sum(np.exp(p/T))
            
        char_out = ix_to_char[int(np.random.choice(np.arange(input_size), p=p))]

        f(char_out)

        pred = char_out

        i+=1
                                                                         
        if char_out == '\n' or i > max_:
            break
        

if __name__ == "__main__":
    sess, model = load_trained_model()
    vocab = load_vocab()
    run(sess, model, vocab)
