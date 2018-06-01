import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm
import random

# Disable tensorflow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

MIN_PARAGRAPH_LEN = 5

def load_data(_dir):
    ret = []
    for each in os.listdir(_dir):
        full_path = os.path.join(_dir, each)
        if each.endswith("txt"):
            with open(full_path, "rb") as f:
                aux = f.read().decode("utf-8").split('\n\n')
                for paragraph in aux:
                    paragraph = paragraph.strip('\n')
                    paragraph += '\n'
                    if len(paragraph) < MIN_PARAGRAPH_LEN:
                        continue
                    ret.append(paragraph)
    return ret


def preprocess(paragraphs):
    chars = set()
    
    for each in paragraphs:
        chars.update(set(each))
    
    char_to_ix = { ch:i for i,ch in enumerate(chars) }
    ix_to_char = { i:ch for i,ch in enumerate(chars) }
    
    aux = len(char_to_ix)
    char_to_ix["<START>"] = aux
    ix_to_char[aux] = "<START>"
    
    vocab_size = len(char_to_ix)

    max_p = max([len(i) for i in paragraphs]) + 1 # Plus one because of the START token
    
    ret = np.zeros(shape=(len(paragraphs), max_p, vocab_size), dtype=np.uint8)
    lens = np.zeros(shape=len(paragraphs), dtype=np.uint8)

    for idx, each in tqdm(enumerate(paragraphs)):
        lens[idx] = len(each) + 1
        for i in range(max_p - len(each) - 1):
            each += '\n'

        aux = np.zeros(shape=(max_p, vocab_size))
        aux[0][char_to_ix["<START>"]] = 1
        for i, c in enumerate(each):
            aux[i+1][char_to_ix[c]] = 1
        ret[idx] = aux
        
    return ret, lens, char_to_ix, ix_to_char


def get_model(times, input_size, n_hidden):
    tf.reset_default_graph()

    init = tf.contrib.layers.xavier_initializer()
    x = tf.placeholder(tf.float32, shape=(None, times, input_size), name="x")
    y = tf.placeholder(tf.float32, shape=(None, times, input_size))
    seq_len = tf.placeholder(tf.int64, shape=(None), name="seq_len")

    x_2 = tf.unstack(x, axis=1)

    init_state_c_1 = tf.placeholder(tf.float32, shape=[None, n_hidden], name="init_state_c_1")
    init_state_h_1 = tf.placeholder(tf.float32, shape=[None, n_hidden], name="init_state_h_1")

    init_state_c_2 = tf.placeholder(tf.float32, shape=[None, n_hidden], name="init_state_c_2")
    init_state_h_2 = tf.placeholder(tf.float32, shape=[None, n_hidden], name="init_state_h_2")

    cell_1 = tf.contrib.rnn.BasicLSTMCell(n_hidden)
    cell_2 = tf.contrib.rnn.BasicLSTMCell(n_hidden)

    cell = tf.contrib.rnn.MultiRNNCell([cell_1, cell_2])
        
    t_1 = tf.contrib.rnn.LSTMStateTuple(init_state_c_1, init_state_h_1)
    t_2 = tf.contrib.rnn.LSTMStateTuple(init_state_c_2, init_state_h_2)

    outputs, states = tf.contrib.rnn.static_rnn(cell, x_2, dtype=tf.float32, sequence_length=seq_len, initial_state=(t_1, t_2))

    states_0 = tf.nn.rnn_cell.LSTMStateTuple(tf.identity(states[0][0], name="states_0_c"), tf.identity(states[0][1], name="states_0_h"))
    states_1 = tf.nn.rnn_cell.LSTMStateTuple(tf.identity(states[1][0], name="states_1_c"), tf.identity(states[1][1], name="states_1_h"))

    states = (states_0, states_1)

    outputs_2 = tf.stack(outputs, axis=1)

    out = tf.layers.dense(outputs_2, units=input_size, kernel_initializer=init, name="out")

    out_softmax = tf.nn.softmax(out, name="out_softmax")

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y))

    upd = tf.train.AdamOptimizer().minimize(loss)

    return init, x, y, seq_len, init_state_c_1, init_state_h_1, init_state_c_2, init_state_h_2, outputs, states, out, out_softmax, loss, upd


def test(sess, model, times, input_size, n_hidden, max_=1000, T=None):
    init, x, y, seq_len, init_state_c_1, init_state_h_1, init_state_c_2, init_state_h_2, outputs, states, out, out_softmax, loss, upd = model

    pred = "<START>"
    
    c_1 = np.zeros((1, n_hidden))
    h_1 = np.zeros((1, n_hidden))
    
    c_2 = np.zeros((1, n_hidden))
    h_2 = np.zeros((1, n_hidden))
    
    ret = []
        
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
        ret.append(char_out)

        pred = char_out
                                                                         
        if char_out == '\n' or len(ret) > max_:
            break
        
    return ret                                                                       


def infinite_train(model, times, n_hidden, input_size, batch_size, print_each):
    init, x, y, seq_len, init_state_c_1, init_state_h_1, init_state_c_2, init_state_h_2, outputs, states, out, out_softmax, loss, upd = model

    N, M, V = data.shape

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    zeros = np.zeros(shape=(batch_size))
    times_minus_one = (times - 1) * np.ones(shape=(batch_size))

    counter = 0
    e = 0
    while True:
        idxs = np.random.choice(N, batch_size, replace=False)
        batch = data[idxs]
        batch_lens = lens[idxs].astype(np.int32)
        
        ts = (M-1) // times # + 1
        
        c_1 = np.zeros((batch_size, n_hidden))
        h_1 = np.zeros((batch_size, n_hidden))

        c_2 = np.zeros((batch_size, n_hidden))
        h_2 = np.zeros((batch_size, n_hidden))
        
        if e % print_each == 0:
            print("Epoch # {}:".format(e))
            print("".join(test(sess, model, times, input_size, n_hidden, max_=500)))
            print()
        
        for t in range(ts):
            batch_x = batch[:, t*times:times*(t+1), :]
            batch_y = batch[:, t*times+1:times*(t+1)+1, :]
            
            batch_lens_aux = batch_lens -  (times * t)
            
            batch_lens_aux = np.maximum(zeros, batch_lens_aux)
            batch_lens_aux = np.minimum(times_minus_one, batch_lens_aux)
            
            batch_lens_aux = batch_lens_aux.astype(np.uint8)
            
            non_zero_idxs = batch_lens_aux > 0
            batch_lens_aux = batch_lens_aux[non_zero_idxs]

            batch_x = batch_x[non_zero_idxs, :, :]
            batch_y = batch_y[non_zero_idxs, :, :]
            c_l_1 = c_1[non_zero_idxs]
            h_l_1 = h_1[non_zero_idxs]
            
            c_l_2 = c_2[non_zero_idxs]
            h_l_2 = h_2[non_zero_idxs]
            
            if np.all(batch_lens_aux == 0):
                break
        
               
            states_, _ = sess.run([states, upd], feed_dict={x: batch_x, y: batch_y, init_state_c_1: c_l_1, init_state_h_1: h_l_1, init_state_c_2: c_l_2, init_state_h_2: h_l_2, seq_len: batch_lens_aux})
            
            counter += 1
            
            c_1[non_zero_idxs] = states_[0].c
            h_1[non_zero_idxs] = states_[0].h

            c_2[non_zero_idxs] = states_[1].c
            h_2[non_zero_idxs] = states_[1].h

        e += 1
        

if __name__ == "__main__":
    print("Cargando la data. Esto puede tardar unos segundos...")
    ps = load_data("./speeches/")
    data, lens, char_to_ix, ix_to_char = preprocess(ps)

    BATCH_SIZE = 64
    INPUT_SIZE = len(ix_to_char)
    TIMES = 32
    N_HIDDEN = 512

    PRINT_EACH = 10

    print("Empezando el entrenamiento. Presione enter para arrancar. En cualquier momento, presion Ctrl+C para terminar el proceso.")
    input()

    model = get_model(TIMES, INPUT_SIZE, N_HIDDEN)

    infinite_train(model, TIMES, N_HIDDEN, INPUT_SIZE, BATCH_SIZE, PRINT_EACH)
