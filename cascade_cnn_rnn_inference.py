import pandas as pd
import pickle
import numpy as np
import time
from pylsl import StreamInlet, resolve_stream

# loading stream
print('looking for an EEG stream...')
stream = resolve_stream('type', 'EEG')
print(stream)
inlet = StreamInlet(stream[0])
print('Found stream')


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

np.random.seed(33)

conv_1_shape = '3*3*1*32'
pool_1_shape = 'None'

conv_2_shape = '3*3*1*64'
pool_2_shape = 'None'

conv_3_shape = '3*3*1*128'
pool_3_shape = 'None'

conv_4_shape = 'None'
pool_4_shape = 'None'

n_person = 108
window_size = 10
n_lstm_layers = 2
# full connected parameter
fc_size = 1024
n_fc_in = 1024
n_fc_out = 1024

dropout_prob = 0.5

calibration = 'N'
norm_type='2D'
regularization_method = 'dropout'
enable_penalty = False

input_dir = "conv_3l_win_10_108_fc_rnn2_fc_1024_N_summary_075_train_posibility_roc"
model_file = "conv_3l_win_10_108_fc_rnn2_fc_1024_N_summary_075_train_posibility_roc"

dataset_dir = "preprocessed_data/3D_CNN/raw_data/"

with open(dataset_dir+"1_108_shuffle_labels_3D_win_10.pkl", "rb") as fp:
    labels = pickle.load(fp)

one_hot_labels = np.array(list(pd.get_dummies(labels)))
print(one_hot_labels)
labels = np.asarray(pd.get_dummies(labels), dtype = np.int8)

print("**********("+time.asctime(time.localtime(time.time()))+") Define parameters and functions Begin: **********\n")

# input parameter
input_channel_num = 1

input_height = 10
input_width = 11

n_labels = 5

# training parameter
lambda_loss_amount = 0.0005
training_epochs = 1

batch_size = 1

# kernel parameter
kernel_height_1st = 3
kernel_width_1st = 3

kernel_height_2nd = 3
kernel_width_2nd = 3

kernel_height_3rd = 3
kernel_width_3rd = 3

kernel_stride = 1
conv_channel_num = 32
# pooling parameter
pooling_height = 2
pooling_width = 2

pooling_stride = 2

# algorithn parameter
learning_rate = 1e-4

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, kernel_stride):
# API: must strides[0]=strides[4]=1
    return tf.nn.conv2d(x, W, strides=[1, kernel_stride, kernel_stride, 1], padding='SAME')

def apply_conv2d(x, filter_height, filter_width, in_channels, out_channels, kernel_stride):
    weight = weight_variable([filter_height, filter_width, in_channels, out_channels])
    bias = bias_variable([out_channels]) # each feature map shares the same weight and bias
    return tf.nn.elu(tf.add(conv2d(x, weight, kernel_stride), bias))

def apply_max_pooling(x, pooling_height, pooling_width, pooling_stride):
# API: must ksize[0]=ksize[4]=1, strides[0]=strides[4]=1
    return tf.nn.max_pool(x, ksize=[1, pooling_height, pooling_width, 1], strides=[1, pooling_stride, pooling_stride, 1], padding='SAME')

def apply_fully_connect(x, x_size, fc_size):
    fc_weight = weight_variable([x_size, fc_size])
    fc_bias = bias_variable([fc_size])
    return tf.nn.elu(tf.add(tf.matmul(x, fc_weight), fc_bias))

def apply_readout(x, x_size, readout_size):
    readout_weight = weight_variable([x_size, readout_size])
    readout_bias = bias_variable([readout_size])
    return tf.add(tf.matmul(x, readout_weight), readout_bias)

print("**********("+time.asctime(time.localtime(time.time()))+") Define parameters and functions End **********\n")

print("**********("+time.asctime(time.localtime(time.time()))+") Define NN structure Begin: **********\n")

# input placeholder
X = tf.placeholder(tf.float32, shape=[None, input_height, input_width, input_channel_num], name='X')
Y = tf.placeholder(tf.float32, shape=[None, n_labels], name = 'Y')
keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
phase_train = tf.placeholder(tf.bool, name = 'phase_train')

# first CNN layer
conv_1 = apply_conv2d(X, kernel_height_1st, kernel_width_1st, input_channel_num, conv_channel_num, kernel_stride)
# pool_1 = apply_max_pooling(conv_1, pooling_height, pooling_width, pooling_stride)
print(conv_1.shape)
# second CNN layer
conv_2 = apply_conv2d(conv_1, kernel_height_2nd, kernel_width_2nd, conv_channel_num, conv_channel_num*2, kernel_stride)
# pool_2 = apply_max_pooling(conv_2, pooling_height, pooling_width, pooling_stride)
print(conv_2.shape)
# third CNN layer
conv_3 = apply_conv2d(conv_2, kernel_height_3rd, kernel_width_3rd, conv_channel_num*2, conv_channel_num*4, kernel_stride)
# fully connected layer
print(conv_3.shape)
shape = conv_3.get_shape().as_list()

pool_2_flat = tf.reshape(conv_3, [-1, shape[1]*shape[2]*shape[3]])
fc = apply_fully_connect(pool_2_flat, shape[1]*shape[2]*shape[3], fc_size)

# dropout regularizer
# Dropout (to reduce overfitting; useful when training very large neural network)
# We will turn on dropout during training & turn off during testing

fc_drop = tf.nn.dropout(fc, keep_prob)

# fc_drop size [batch_size*window_size, fc_size]
# lstm_in size [batch_size, window_size, fc_size]
lstm_in = tf.reshape(fc_drop, [-1, window_size, fc_size])

###########################################################################################
# add lstm cell to network
###########################################################################################
# define lstm cell
cells = []
for _ in range(n_lstm_layers):
    cell = tf.nn.rnn_cell.BasicLSTMCell(n_fc_in, forget_bias=1.0, state_is_tuple=True)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
    cells.append(cell)
lstm_cell = tf.nn.rnn_cell.MultiRNNCell(cells)

init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

# output ==> [batch, step, n_fc_in]
output, states = tf.nn.dynamic_rnn(lstm_cell, lstm_in, initial_state=init_state, time_major=False)

# output ==> [step, batch, n_fc_in]
# output = tf.transpose(output, [1, 0, 2])

# only need the output of last time step
# rnn_output ==> [batch, n_fc_in]
# rnn_output = tf.gather(output, int(output.get_shape()[0])-1)
# print(type(rnn_output))
###################################################################
# another output method
output = tf.unstack(tf.transpose(output, [1, 0, 2]), name = 'lstm_out')
rnn_output = output[-1]
###################################################################

###########################################################################################
# fully connected and readout
###########################################################################################
# rnn_output ==> [batch, fc_size]
shape_rnn_out = rnn_output.get_shape().as_list()
# fc_out ==> [batch_size, n_fc_out]
fc_out = apply_fully_connect(rnn_output, shape_rnn_out[1], n_fc_out)

# keep_prob = tf.placeholder(tf.float32)
fc_drop = tf.nn.dropout(fc_out, keep_prob)

# readout layer
y_ = apply_readout(fc_drop, shape_rnn_out[1], n_labels)
y_pred = tf.argmax(tf.nn.softmax(y_), 1, name = "y_pred")
y_posi = tf.nn.softmax(y_, name = "y_posi")

# l2 regularization
l2 = lambda_loss_amount * sum(
    tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
)

if enable_penalty:
    # cross entropy cost function
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=Y) + l2, name = 'loss')
else:
    # cross entropy cost function
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=Y), name = 'loss')

# get correctly predicted object and accuracy
correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(y_), 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name = 'accuracy')

print("**********("+time.asctime(time.localtime(time.time()))+") Define NN structure End **********\n")

print("**********("+time.asctime(time.localtime(time.time()))+") Train and Test NN Begin: **********\n")

# run
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


#   0,  1,  2,   3,  4,  5,  6,  7, -6, -5,  -4, -3, -2,  -1
# AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4
def convert2D(data, Y=10, X=11):
    data_2D = np.zeros([Y, X])
    data_2D[0] = (       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0)
    data_2D[1] = (       0,       0,       0,       0, data[0],       0,data[-1],       0,       0,       0,       0)
    data_2D[2] = (       0, data[1],       0, data[2],       0,       0,       0,data[-3],       0,data[-2],       0)
    data_2D[3] = (       0,       0, data[3],       0,       0,       0,       0,       0,data[-4],       0,       0)
    data_2D[4] = (       0, data[4],       0,       0,       0,       0,       0,       0,       0,data[-5],       0)
    data_2D[5] = (       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0)
    data_2D[6] = (       0, data[5],       0,       0,       0,       0,       0,       0,       0,data[-6],       0)
    data_2D[7] = (       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0)
    data_2D[8] = (       0,       0,       0,       0, data[6],       0, data[7],       0,       0,       0,       0)
    data_2D[9] = (       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0)
    return data_2D


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def color_switch(p):
    if p == 0:
        return bcolors.ENDC
    elif p == 1:
        return bcolors.OKBLUE
    elif p == 2:
        return bcolors.OKCYAN
    elif p == 3:
        return bcolors.OKGREEN
    elif p == 4:
        return bcolors.WARNING
    else:
        return bcolors.FAIL

with tf.Session(config=config) as session:
    saver = tf.train.Saver()
    saver.restore(session, "./result/"+input_dir+"/model_" + model_file)
    print('MODEL LOADED')
    # session.run(tf.global_variables_initializer())

    # loop
    # get data from stream

    # predict
    # pred = session.run([y_pred], feed_dict={X: test_batch_x, Y: test_batch_y, keep_prob: 1.0, phase_train: False})
    # print(pred)
    # print(one_hot_labels[pred])


    chunk = np.zeros([10, 10, 11])
    counter = 0


    while True:
        sample, timestamp = inlet.pull_sample()
        sample = np.asarray(sample[3:-1]) - 4100
        chunk[counter] = convert2D(sample)
        counter += 1
        if counter == 10:
            counter = 0
            prob, pred = session.run([y_posi, y_pred], feed_dict={X: chunk.reshape(10,10,11,1), keep_prob: 1.0})
            # print(color_switch(pred), one_hot_labels[pred], bcolors.ENDC, prob)
            if (prob[0][pred] > 0.7):
                print(color_switch(pred), one_hot_labels[pred], bcolors.ENDC, prob)
            else:
                print('NOT RECOGNIZED')
            chunk = np.zeros([10, 10, 11])
