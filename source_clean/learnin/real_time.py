import tensorflow as tf
import numpy as np
import datetime
import socket
import threading
import time
from time import sleep

################# Do not edit here ###################
num_layers = 0
hidden_neuron = 0
direction = 0
dataX_test = []
dataY_test = []
Cell_Type = ""

global data_received_start
global sended_data
global input_data
global send_signal
global first_time
global time_now
global ready_to_go
global few_data

######################################################

ready_to_go = 1
send_signal = 0
data_received_start = 0
result = ''
seq_length = 10  # number of frames for recognition
day = datetime.datetime.now().date()
sended_data = []

class Config(object):
    def __init__(self, X_test):
        # Input data
        self.n_steps = len(X_test[0]) # number of frames for recognition
        self.n_inputs = len(X_test[0][0])  # Number of data in one frame

        # Training Configuration
        self.batch_size = len(X_test)
        self.n_classes = 2  # Number of Classes (11 gestures)

        # hidden neurons in the model
        self.n_hidden = hidden_neuron

        # bidirectional RNN requires doubled hidden number for 'output' weight variable
        if Cell_Type == "Bidirectional_RNN":
            direction = 2
        else:
            direction = 1

        # Define the variables
        self.W = {
            'hidden': tf.Variable(tf.random_normal([self.n_inputs, self.n_hidden])),
            'output': tf.Variable(tf.random_normal([direction*self.n_hidden, self.n_classes]))}
        self.biases = {
            'hidden': tf.Variable(tf.random_normal([self.n_hidden], mean=1.0)),
            'output': tf.Variable(tf.random_normal([self.n_classes]))}

def recieved_data_to_array(data):
    global data_received_start
    global sended_data
    global input_data
    global send_signal
    global first_time
    global ready_to_go
    global few_data

    if ready_to_go == 1:
        few_data = 0

        print(data)

        converted = data.split('\t')
        np_converted = np.array(converted[:-1])
        sended_data.append(np_converted)

        if len(sended_data) < seq_length:
            first_time = time.time()

        elif len(sended_data) == seq_length:
            sended_data = [sended_data]
            input_data = np.array(sended_data)
            if data_received_start == 0:
                data_received_start = 1
            send_signal = 1
            sended_data.clear()

        else:
            sended_data.clear()

def socket_listener(IP, portnum):
    sever_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sever_socket.bind((IP, portnum))
    sever_socket.listen(0)
    print("Socket is ready")
    while True:
        client_socket, addr = sever_socket.accept()
        data = client_socket.recv(65535)
        data = data.decode()
        recieved_data_to_array(data)
    sever_socket.close()


def socket_Sender(strdata):
    clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    clientSocket.connect(('127.0.0.1', 3001))
    clientSocket.send(strdata.encode())
    # clientSocket.close()

def LSTM_Network(_X, config):

    # Change of data shape for RNN structure and ReLU application
    _X = tf.transpose(_X, [1, 0, 2])
    _X = tf.reshape(_X, [-1, config.n_inputs])
    _X = tf.nn.relu(tf.matmul(_X, config.W['hidden']) + config.biases['hidden'])
    _X = tf.split(_X, config.n_steps, 0)

    # Type of RNN Architecture
    if Cell_Type == "LSTM":
        cells = [tf.contrib.rnn.BasicLSTMCell(config.n_hidden) for _ in range(num_layers)]
    elif Cell_Type == "GRU":
        cells = [tf.contrib.rnn.GRUCell(config.n_hidden) for _ in range(num_layers)]
    elif Cell_Type == "RNN":
        cells = [tf.contrib.rnn.BasicRNNCell(config.n_hidden) for _ in range(num_layers)]
    elif Cell_Type == "Bidirectional_RNN":
        cells = [tf.contrib.rnn.BasicRNNCell(config.n_hidden) for _ in range(num_layers)]
    elif Cell_Type == "AttentionCellWrapper":
        cells = [tf.contrib.rnn.AttentionCellWrapper(tf.contrib.rnn.BasicLSTMCell(config.n_hidden), 20, state_is_tuple=True) for _ in range(num_layers)]

    # Make a Deep-Layer RNN Model
    cells = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

    # Make the output tensor and state tensor
    # Bidirectional RNN requires forward state and backward state
    if Cell_Type == "Bidirectional_RNN":
        outputs, fw_states, bw_states = tf.contrib.rnn.static_bidirectional_rnn(cells, cells, _X, dtype=tf.float32)
    else:
        outputs, states = tf.contrib.rnn.static_rnn(cells, _X, dtype=tf.float32)

    lstm_last_output = outputs[-1]

    # Return the calculated value
    return tf.matmul(lstm_last_output, config.W['output']) + config.biases['output']

# Training Session
def result_show(address):
    global send_signal
    global ready_to_go
    global few_data
    config = Config(input_data)

    # Placeholder is required to put the data
    X = tf.placeholder(tf.float32, [None, config.n_steps, config.n_inputs])

    # Return the predicted value
    pred_Y = LSTM_Network(X, config)

    sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False))
    init = tf.global_variables_initializer()
    sess.run(init)

    saver = tf.train.Saver()
    saver.restore(sess, address)

    send_signal = 0

    few_data = 0

    while True:
        global time_now
        sleep(0.1)
        time_now = time.time()

        if time_now-first_time> 2:
            if few_data == 0:
                few_data = 1
                sended_data.clear()
                print("Ready to Go!")

        if send_signal == 1:
            ready_to_go = 0
            send_signal = 0
            print("")
            print("##########################################################")
            print("")
            before_time = time.time()
            pred_out= sess.run(pred_Y, feed_dict={X: input_data})

            result = str(pred_out[0][0]) + '\t' + str(pred_out[0][1])

            print("Predict: ", result)

            socket_Sender(result)
            print("")
            after_time = time.time()
            print("Calculation Time:",(after_time*1000-before_time*1000), 'ms')
            print("")
            sleep(2)
            ready_to_go = 1

    sess.close()

t = threading.Thread(target=socket_listener, args=('127.0.0.1', 8888))
t.start()

hidden_neuron = 80                      # how broad the layers are
num_layers = 15                           # How deep the layers are
Cell_Type = "GRU"                       # Your Cell Type

t2 = threading.Thread(target=result_show, args=('realtime_variables/'+Cell_Type+'_pingpong_weight',))


while data_received_start == 0:
    sleep(0.1)

t2.start()