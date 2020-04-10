import tensorflow as tf
import numpy as np
import math
import os
import datetime
import random

training_epochs = 0
lr = 0
hidden_neuron = 0
num_layers = 0
num_of_data = 0
training_count = 0
global few_data_count
few_data_count = 0

dataX_train = []
dataY_train = []
dataX_test = []
dataY_test = []

Prediction = []

seq_length = 15

Cell_Type = ""

day = datetime.datetime.now().date()

classnum = 2

class Config(object):
    def __init__(self, X_train, X_test, lr, hidden_neuron):

        # Input data
        self.train_count = len(X_train)  # number of training datasets
        self.test_data_count = len(X_test)  # number of test datasets
        self.n_steps = len(X_train[0])  # number of frames for recognition
        self.n_inputs = len(X_train[0][0])  # Number of data in one frame

        # Training Configuration
        self.learning_rate = lr
        self.lambda_loss_amount = 0.0001
        self.training_epochs = training_epochs
        self.batch_size = int(len(X_train)/2)
        self.n_classes = classnum

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

def Datasets_Import(data_num):

    global few_data_count
    test_num = num_of_data/5
    training_num = num_of_data-test_num

    Address_train = '../pingpong/data/data_' + str(data_num) + '.txt'
    load_data = np.loadtxt(Address_train, delimiter='\t')
    load_data = np.around(load_data, decimals=3)

    mirror_matrix = np.array([1,-1,1])
    mirrored_data = load_data * mirror_matrix

    if len(load_data) > seq_length:

        x_, y_ = load_data[:seq_length, :-1], load_data[[-1], :-1]
        mirrored_x_, mirrored_y_ = mirrored_data[:seq_length, :-1], mirrored_data[[-1], :-1]

        if training_count < training_num:
            dataX_train.append(x_)
            dataY_train.append(y_)
            dataX_train.append(mirrored_x_)
            dataY_train.append(mirrored_y_)
        else:
            dataX_test.append(x_)
            dataY_test.append(y_)

    else:
        few_data_count = few_data_count + 1
        # print("Too Few Data at data", data_num)

def LSTM_Network(_X, config):
    _X = tf.transpose(_X, [1, 0, 2])
    _X = tf.reshape(_X, [-1, config.n_inputs])
    _X = tf.nn.relu(tf.matmul(_X, config.W['hidden']) + config.biases['hidden'])
    _X = tf.split(_X, config.n_steps, 0)

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


    cells = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

    if Cell_Type == "Bidirectional_LSTM":
        outputs, fw_states, bw_states = tf.contrib.rnn.static_bidirectional_rnn(cells, cells, _X, dtype=tf.float32)

    else:
        outputs, states = tf.contrib.rnn.static_rnn(cells, _X, dtype=tf.float32)

    lstm_last_output = outputs[-1]

    return tf.matmul(lstm_last_output, config.W['output']) + config.biases['output']

def training(input_dataX_train, input_dataX_test):

    tf.reset_default_graph()

    X_train2 = np.array(input_dataX_train)
    Y_train2 = np.array(dataY_train)

    X_test2 = np.array(input_dataX_test)
    Y_test2 = np.array(dataY_test)

    X_train = X_train2[:, :, :]
    Y_train = Y_train2[:, :, :]
    X_test = X_test2[:, :, :]
    Y_test = Y_test2[:, :, :]


    print('[INFO] Number of training sets = {}'.format(len(X_train)))
    print('[INFO] Number of test sets = {}'.format(len(X_test)))


    config = Config(X_train, X_test, lr, hidden_neuron)

    Y_train = Y_train.reshape(-1, config.n_classes)
    Y_test = Y_test.reshape(-1, config.n_classes)


    X = tf.placeholder(tf.float32, [None, config.n_steps, config.n_inputs])
    Y = tf.placeholder(tf.float32, [None, config.n_classes])

    pred_Y = LSTM_Network(X, config)

    # lambda_loss_amount = 0.00001
    #
    # l2 = lambda_loss_amount * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
    # cost = tf.losses.mean_squared_error(Y, pred_Y) + l2

    cost = tf.losses.mean_squared_error(Y, pred_Y)

    optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(cost)

    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=config.learning_rate).minimize(cost)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33)

    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    sess.run(init)

    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')

    # Start training for each batch and loop epochs

    Prediction = []

    target_error = 0.1

    lowest_error = 1000.0

    for i in range(config.training_epochs):
        for start, end in zip(range(0, config.train_count, config.batch_size),
                              range(config.batch_size, config.train_count + 1, config.batch_size)):
            sess.run(optimizer, feed_dict={X: X_train[start:end], Y: Y_train[start:end]})

        # Test completely at every epoch: calculate accuracy
        pred_out_train, loss = sess.run([pred_Y, cost], feed_dict={X: X_train, Y: Y_train})
        pred_out_test, loss_test = sess.run([pred_Y, cost], feed_dict={X: X_test, Y: Y_test})

        Prediction = pred_out_test

        training_error = math.sqrt(loss)
        test_error = math.sqrt(loss_test)

        print("")
        print(Cell_Type, layer, "Layer", hidden_neuron, "HN", i, "th Epoch")
        print("Test Error: ", "%0.2f" %(test_error*100), "cm")
        print("Training Error: ", "%0.2f" %(training_error*100), "cm")
        print("Lowest Error:", "%0.2f" %(lowest_error*100), "cm")

        dirName = '../pingpong/error/'+str(day)

        if not os.path.exists(dirName):
            os.mkdir(dirName)

        f = open(dirName+'/' + current_time + '_' + Cell_Type + '_' + str(num_layers)+ 'layer_' + str(hidden_neuron) + 'hidden_unit_montreal.txt', 'a')

        data = ("%f\t%d\t%d\t%f\t%f\n" % (config.learning_rate, num_layers, hidden_neuron, test_error, training_error))
        f.write(data)
        f.close()

        dirName = 'Pingpong_Trained_Weight/'+str(day) + '/' + str(num_layers) + '/' + str(hidden_neuron) + '/' + str(target_error)

        if not os.path.exists(dirName):
            os.makedirs(dirName)

        # Weight Save
        if test_error <= target_error:
            saver.save(sess, dirName + '/' + Cell_Type + '_pingpong_weight')
            target_error = target_error - 0.005

        if test_error < lowest_error:
            lowest_error = test_error

    f = open('Result.txt', 'a')

    data = ("%s\t%s\t%d\t%d\t%f\t%f\t%f\t%f\n" % (
    current_time, Cell_Type, hidden_neuron, num_layers, config.learning_rate, lowest_error * 100, test_error * 100, training_error * 100))
    f.write(data)
    f.close()

    np.savetxt('Prediction/'+current_time+'_______Test.txt', Y_test, fmt='%f', delimiter='\t')
    np.savetxt('Prediction/'+current_time+'_Prediction.txt', Prediction, fmt='%f', delimiter='\t')
    np.savetxt('Prediction/'+current_time+'______Error.txt', Y_test-Prediction, fmt='%f', delimiter='\t')
    sess.close()

print("Data Import Start")
few_data_count = 0
num_of_data = 8000
data_shuffle = list(range(num_of_data))
random.shuffle(data_shuffle)
for i in data_shuffle:
    Datasets_Import(i)
    training_count = training_count + 1
print("Data Import End")
print("")
print(few_data_count, "out of total", num_of_data, "is not used!!")

##############################################################################################

layer = 15  #15
neuron = 60  #80

training_epochs = 10000
lr = 0.0001
hidden_neuron = neuron
num_layers = layer

# while True:
#     while num_layers > 0:
#         while hidden_neuron > 0:
#
#             Cell_Type = "RNN"
#             training(dataX_train, dataX_test)
#
#             Cell_Type = "LSTM"
#             training(dataX_train, dataX_test)
#
#             hidden_neuron = hidden_neuron - 10
#         hidden_neuron = neuron
#         num_layers = num_layers - 5
#     num_layers = layer



while True:
    while num_layers > 0:
        while hidden_neuron > 0:
            Cell_Type = "GRU"
            training(dataX_train, dataX_test)

            hidden_neuron = hidden_neuron - 10
        hidden_neuron = neuron
        num_layers = num_layers - 5
    num_layers = layer

##############################################################################################


# layer = 20
# neuron = 50
#
# training_epochs = 10000
# lr = 0.0001
# hidden_neuron = neuron
# num_layers = layer
#
# while True:
#     while num_layers > 0:
#         while hidden_neuron > 0:
#             Cell_Type = "LSTM"
#             training(dataX_train, dataX_test)
#
#             Cell_Type = "RNN"
#             training(dataX_train, dataX_test)
#
#             hidden_neuron = hidden_neuron - 10
#         hidden_neuron = neuron
#         num_layers = num_layers - 10
#
#     num_layers = layer