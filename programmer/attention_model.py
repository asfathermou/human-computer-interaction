
# coding: utf-8

# ## import

# In[1]:


import sys
import sklearn
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import pickle
import tensorflow as tf
import numpy as np
import time
import math
import matplotlib.pyplot as plt


# ## setting

# In[2]:


conv_1_shape = '4*4*32'
pool_1_shape = 'None'

conv_2_shape = '4*4*64'
pool_2_shape = 'None'

conv_3_shape = '4*4*128'
pool_3_shape = 'None'

conv_4_shape = '1*1*13'
pool_4_shape = 'None'

window_size = 128

dropout_prob = 0.5
np.random.seed(32)

norm_type = '2D'
regularization_method = 'dropout'
enable_penalty = True


# ## Read-in

# In[4]:

file_dir = '../preprocessed_data'
cnn_suffix        =".mat_win_128_cnn_dataset.pkl"
label_suffix    =".mat_win_128_labels.pkl"
num_people = 6
test = 1
cnn_datasets=np.empty(shape=[0,128,9,9])
labels=np.empty(shape=[0,])
eog_datasets=np.empty(shape=[0,128,2])
for i in range(num_people):
    with open(file_dir + 's0'+str(i+test) + cnn_suffix,"rb") as f:
        cnn_dataset = pickle.load(f)
    with open(file_dir + 's0'+str(i+test) + label_suffix,"rb") as f:
        label = pickle.load(f)   
    cnn_datasets=np.concatenate((cnn_datasets,cnn_dataset), axis=0)
    labels=np.concatenate((labels,label),axis=0)
labels_backup = labels
# one-hot encoding
labels = np.asarray(pd.get_dummies(labels), dtype=np.int8)
print("labels shape after one-hot:", np.shape(labels))
cnn_datasets = cnn_datasets.reshape((len(cnn_datasets), window_size, 9, 9, 1))
print("cnn_dataset shape after reshape:", np.shape(cnn_datasets))


# ## Split train and test

# In[5]:


split_index = int((num_people - 1) / num_people * len(cnn_datasets))
cnn_train_x = cnn_datasets[:split_index]
cnn_test_x = cnn_datasets[split_index:]
train_y = labels[:split_index]
test_y = labels[split_index:]
index = np.array(range(len(cnn_train_x)))
np.random.shuffle(index)
cnn_train_x = cnn_train_x[index]
train_y = train_y[index]
print("training samples:{0},\ntest samples:{1},\ntraining labels:{2},\ntest labels:{3}"      .format(cnn_train_x.shape, cnn_test_x.shape, train_y.shape, test_y.shape))


# ## hyper-param

# In[6]:


# input parameter
n_input_ele = 32
n_time_step = window_size

input_channel_num = 1
input_height = 9
input_width = 9

n_labels = 2
# training parameter
lambda_loss_amount = 0.5
training_epochs = 30

batch_size = 35

# kernel parameter
kernel_height_1st = 4
kernel_width_1st = 4

kernel_height_2nd = 4
kernel_width_2nd = 4

kernel_height_3rd = 4
kernel_width_3rd = 4

kernel_height_4th = 1
kernel_width_4th = 1

kernel_stride = 1
conv_channel_num = 32

# algorithn parameter
learning_rate = 1e-3


# ## variable and layers function

# In[7]:


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W, kernel_stride):
    # API: must strides[0]=strides[4]=1
    return tf.nn.conv2d(x, W, strides=[1, kernel_stride, kernel_stride, 1], padding='SAME')

def apply_conv2d(x, filter_height, filter_width, in_channels, out_channels, kernel_stride):
    weight = weight_variable([filter_height, filter_width, in_channels, out_channels])
    bias = bias_variable([out_channels])  # each feature map shares the same weight and bias
    #print("weight shape:", np.shape(weight))
    #print("x shape:", np.shape(x))
    #tf.layers.batch_normalization()
    return tf.nn.elu(tf.layers.batch_normalization(conv2d(x, weight, kernel_stride)))

def apply_max_pooling(x, pooling_height, pooling_width, pooling_stride):
    # API: must ksize[0]=ksize[4]=1, strides[0]=strides[4]=1
    return tf.nn.max_pool(x, ksize=[1, pooling_height, pooling_width, 1],
                          strides=[1, pooling_stride, pooling_stride, 1], padding='SAME')

def apply_fully_connect(x, x_size, fc_size):
    fc_weight = weight_variable([x_size, fc_size])
    fc_bias = bias_variable([fc_size])
    return tf.nn.elu(tf.add(tf.matmul(x, fc_weight), fc_bias))

def apply_fully_connect_relu(x, x_size, fc_size):
    fc_weight = weight_variable([x_size, fc_size])
    fc_bias = bias_variable([fc_size])
    return tf.nn.relu(tf.add(tf.matmul(x, fc_weight), fc_bias))

def apply_readout(x, x_size, readout_size):
    readout_weight = weight_variable([x_size, readout_size])
    readout_bias = bias_variable([readout_size])
    return tf.add(tf.matmul(x, readout_weight), readout_bias)

#print("\n**********(" + time.asctime(time.localtime(time.time())) + ") Define parameters and functions End **********")
#print("\n**********(" + time.asctime(time.localtime(time.time())) + ") Define NN structure Begin: **********")


# ## place-holder

# In[8]:


# set placeholder
cnn_in = tf.placeholder(tf.float32, shape=[None,input_height,input_width,1], name="cnn_in")
model_output = tf.placeholder(tf.float32, shape=[None, n_labels], name="output")
Y = tf.placeholder(tf.float32, shape=[None, n_labels], name='Y')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
phase_train = tf.placeholder(tf.bool, name='phase_train')


# # Graph

# ## CNN-Part-Construction

# In[9]:


###########################################################################################
# add cnn parallel to network
###########################################################################################
# first CNN layer
with tf.name_scope("conv_1"):
    conv_1 = apply_conv2d(cnn_in, kernel_height_1st, kernel_width_1st, input_channel_num, conv_channel_num, kernel_stride)
    print("conv_1 shape:", conv_1.shape)
# second CNN layer
with tf.name_scope("conv_2"):
    conv_2 = apply_conv2d(conv_1, kernel_height_2nd, kernel_width_2nd, conv_channel_num, conv_channel_num * 2,kernel_stride)
    print("conv_2 shape:", conv_2.shape)
# third CNN layer
with tf.name_scope("conv_3"):
    conv_3 = apply_conv2d(conv_2, kernel_height_3rd, kernel_width_3rd, conv_channel_num * 2, conv_channel_num * 4,kernel_stride)
    print("conv_3 shape:", conv_3.shape)
# fourth CNN layer
with tf.name_scope("conv_4"):
    conv_4 = apply_conv2d(conv_3, kernel_height_4th, kernel_width_4th, conv_channel_num * 4, 13,kernel_stride)
    print("\nconv_4 shape:", conv_4.shape)
    
# flatten (13*9*9) cube into a 1053 vector.
shape = conv_4.get_shape().as_list()
# flatten (9*9*13) cube into a 13 vector(dimention:81)
conv_flat = tf.reshape(conv_4, [-1, 9*9*shape[3]])
cnn_out = tf.reshape(conv_flat, [-1, window_size, 9*9, shape[3]])
print("cnn out shape: ", cnn_out.shape)
"""保留13个通道，这里认为13个通道代表13个不同的高级语义信息，下面Attention模型(先不用multi-head)
分别在这13个通道上去找时间维度的关系，后进行融合再做分类"""


# ## Attention-part

# In[10]:


def multi_channel_attention(inputs):
    K = inputs
    Q = inputs
    V = inputs
    input_shape = inputs.get_shape().as_list()
    similarity = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(float(input_shape[-1]))
    outputs = tf.matmul(tf.nn.softmax(similarity), V)
    return outputs


# In[11]:


def multi_channel_LT(x, output_size, bias=False):
    x_shape = x.get_shape().as_list()
    print("shape before LT: ", x_shape)
    x = tf.reshape(x, (-1, x_shape[-1]))
    W = weight_variable(np.array([x_shape[-1], output_size]))
    if bias:
        b = bias_variable([output_size])
        outputs = tf.add(tf.matmul(x, W), b)
    else:
        outputs = tf.matmul(x, W)
    outputs = tf.reshape(outputs, (-1, x_shape[1], x_shape[2], output_size))
    return outputs


# In[12]:


def apply_att_and_LT(inputs):
    att = multi_channel_attention(inputs)
    att = tf.concat([att, inputs], axis = -1)
    att = multi_channel_LT(att, int(att.get_shape().as_list()[-1] / 2))
    #att += inputs
    print("after attention layer shape: ", att.shape)
    return att


# In[13]:


cnn_out = tf.transpose(cnn_out, (0,3,1,2))
att_1 = apply_att_and_LT(cnn_out)
att_2 = apply_att_and_LT(att_1)
att_3 = apply_att_and_LT(att_2)
att_3_shape = att_3.get_shape().as_list()
att_flatten = tf.reshape(att_3, (-1, att_3_shape[1] * att_3_shape[2] * att_3_shape[3]))
att_flatten_shape = att_flatten.get_shape().as_list()
print("after flatten shape: ", att_flatten_shape)
att_out = apply_fully_connect(att_flatten, att_flatten_shape[-1], 1024)
att_out = apply_readout(att_out, 1024, n_labels)
print("attention network output shape: ", att_out.shape)
y_ = att_out
y_pred = tf.argmax(tf.nn.softmax(y_), 1, name="y_pred")
y_posi = tf.nn.softmax(y_, name="y_posi")

# ## cost and optimizer

# In[14]:


# l2 regularization
l2 = lambda_loss_amount * sum(
    tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
)

if enable_penalty:
    # cross entropy cost function
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=Y) + l2, name='loss')
    tf.summary.scalar('cost_with_L2',cost)
else:
    # cross entropy cost function
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=Y), name='loss')
    tf.summary.scalar('cost',cost)

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# get correctly predicted object and accuracy
correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(y_), 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
tf.summary.scalar('accuracy',accuracy)

print("\n**********(" + time.asctime(time.localtime(time.time())) + ") Define NN structure End **********")

print("\n**********(" + time.asctime(time.localtime(time.time())) + ") Train and Test NN Begin: **********")




# In[15]:


batch_num_per_epoch = math.floor(cnn_train_x.shape[0]/batch_size)+ 1
accuracy_batch_size = batch_size
train_accuracy_batch_num = batch_num_per_epoch
test_accuracy_batch_num = math.floor(cnn_test_x.shape[0]/batch_size)+ 1


# # Session

# In[ ]:


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.7
merged = tf.summary.merge_all()
logdir = "my_tensorboard"
train_writer = tf.summary.FileWriter("log/"+logdir+"/train")
test_writer = tf.summary.FileWriter("log/"+logdir+"/test")
with tf.Session() as session:
    count_cost = 0
    train_count_accuracy = 0
    test_count_accuracy = 0
    session.run(tf.global_variables_initializer())
    train_accuracy_save = np.zeros(shape=[0], dtype=float)
    test_accuracy_save = np.zeros(shape=[0], dtype=float)
    test_loss_save = np.zeros(shape=[0], dtype=float)
    train_loss_save = np.zeros(shape=[0], dtype=float)
    for epoch in range(training_epochs):
        print("learning rate: ",learning_rate)
        cost_history = np.zeros(shape=[0], dtype=float)
        for b in range(batch_num_per_epoch):
            start = b * batch_size
            if (b+1) * batch_size > train_y.shape[0]:
                offset = train_y.shape[0] % batch_size
            else:
                offset = batch_size
            # print("start:",start,"end:",start+offset)
            cnn_batch = cnn_train_x[start:(start + offset), :, :, :, :]
            cnn_batch = cnn_batch.reshape(len(cnn_batch) * window_size, 9, 9, 1)
            batch_y = train_y[start:(start + offset), :]
            _ , c = session.run([optimizer, cost],
                               feed_dict={cnn_in: cnn_batch, Y: batch_y, keep_prob: 1 - dropout_prob,
                                          phase_train: True})
            cost_history = np.append(cost_history, c)
            count_cost += 1
        if (epoch % 1 == 0):
            train_accuracy = np.zeros(shape=[0], dtype=float)
            test_accuracy = np.zeros(shape=[0], dtype=float)
            test_loss = np.zeros(shape=[0], dtype=float)
            train_loss = np.zeros(shape=[0], dtype=float)

            for i in range(train_accuracy_batch_num):
                start = i* batch_size
                if (i+1)*batch_size>train_y.shape[0]:
                    offset = train_y.shape[0] % batch_size
                else:
                    offset = batch_size
                train_cnn_batch = cnn_train_x[start:(start + offset), :, :, :, :]
                train_cnn_batch = train_cnn_batch.reshape(len(train_cnn_batch) * window_size, 9, 9, 1)

                train_batch_y = train_y[start:(start + offset), :]

                tf_summary,train_a, train_c = session.run([merged,accuracy, cost],
                                               feed_dict={cnn_in: train_cnn_batch,
                                                          Y: train_batch_y, keep_prob: 1.0, phase_train: False})
                train_writer.add_summary(tf_summary,train_count_accuracy)
                train_loss = np.append(train_loss, train_c)
                train_accuracy = np.append(train_accuracy, train_a)
                train_count_accuracy += 1
            print("(" + time.asctime(time.localtime(time.time())) + ") Epoch: ", epoch + 1, " Training Cost: ",
                  np.mean(train_loss), "Training Accuracy: ", np.mean(train_accuracy))
            train_accuracy_save = np.append(train_accuracy_save, np.mean(train_accuracy))
            train_loss_save = np.append(train_loss_save, np.mean(train_loss))

            if(np.mean(train_accuracy)<0.8):
                learning_rate=1e-4
            elif(0.8<np.mean(train_accuracy)<0.85):
                learning_rate=5e-5
            elif(0.85<np.mean(train_accuracy)):
                learning_rate=5e-6

            for j in range(test_accuracy_batch_num):
                start = j * batch_size
                print(start)
                if (j+1)*batch_size>test_y.shape[0]:
                    offset = test_y.shape[0] % batch_size
                else:
                    offset = batch_size
                test_cnn_batch = cnn_test_x[start:(start + offset), :, :, :, :]
                test_cnn_batch = test_cnn_batch.reshape(len(test_cnn_batch) * window_size, 9, 9, 1)

                test_batch_y = test_y[start:(start + offset), :]

                tf_test_summary,test_a, test_c = session.run([merged,accuracy, cost],
                                             feed_dict={cnn_in: test_cnn_batch, Y: test_batch_y,
                                                        keep_prob: 1.0, phase_train: False})
                test_writer.add_summary(tf_test_summary,test_count_accuracy)
                test_accuracy = np.append(test_accuracy, test_a)
                test_loss = np.append(test_loss, test_c)
                test_count_accuracy += 1 
            print("(" + time.asctime(time.localtime(time.time())) + ") Epoch: ", epoch + 1, " Test Cost: ",
                  np.mean(test_loss), "Test Accuracy: ", np.mean(test_accuracy), "\n")
            test_accuracy_save = np.append(test_accuracy_save, np.mean(test_accuracy))
            test_loss_save = np.append(test_loss_save, np.mean(test_loss))
        # reshuffle
        index = np.array(range(0, len(train_y)))
        np.random.shuffle(index)
        cnn_train_x=cnn_train_x[index]
        train_y=train_y[index]

        # learning_rate decay
        if(np.mean(train_accuracy)<0.9):
            learning_rate=1e-3
        elif(0.9<np.mean(train_accuracy)<0.95):
            learning_rate=5e-5
        elif(0.99<np.mean(train_accuracy)):
            learning_rate=5e-6
    test_accuracy = np.zeros(shape=[0], dtype=float)
    test_loss = np.zeros(shape=[0], dtype=float)
    test_pred = np.zeros(shape=[0], dtype=float)
    test_true = np.zeros(shape=[0, 2], dtype=float)
    test_posi = np.zeros(shape=[0, 2], dtype=float)
    for k in range(test_accuracy_batch_num):
        start = k * batch_size
        if (k+1)*batch_size>test_y.shape[0]:
            offset = test_y.shape[0] % batch_size
        else:
            offset = batch_size
        test_cnn_batch = cnn_test_x[start:(start + offset), :, :, :, :]
        test_cnn_batch = test_cnn_batch.reshape(len(test_cnn_batch) * window_size, 9, 9, 1)
        test_batch_y = test_y[start:(start + offset), :]

        test_a, test_c, test_p, test_r = session.run([accuracy, cost, y_pred, y_posi],
                                                     feed_dict={cnn_in: test_cnn_batch, Y: test_batch_y, keep_prob: 1.0, phase_train: False})
        test_t = test_batch_y

        test_accuracy = np.append(test_accuracy, test_a)
        test_loss = np.append(test_loss, test_c)
        test_pred = np.append(test_pred, test_p)
        test_true = np.vstack([test_true, test_t])
        test_posi = np.vstack([test_posi, test_r])
    test_pred_1_hot = np.asarray(pd.get_dummies(test_pred), dtype=np.int8)
    test_true_list = tf.argmax(test_true, 1).eval()
    print(test_loss)
    # recall
    #test_recall = recall_score(test_true, test_pred_1_hot, average=None)
    # precision
    #test_precision = precision_score(test_true, test_pred_1_hot, average=None)
    # f1 score
    #test_f1 = f1_score(test_true, test_pred_1_hot, average=None)
     # confusion matrix
    # confusion_matrix = confusion_matrix(test_true_list, test_pred)
    #print("********************recall:", test_recall)
    #print("*****************precision:", test_precision)
    #print("******************f1_score:", test_f1)
    print("(" + time.asctime(time.localtime(time.time())) + ") Final Test Cost: ", np.mean(test_loss),
              "Final Test Accuracy: ", np.mean(test_accuracy))
    saver = tf.train.Saver()
    saver.save(session,                    "./model_attetntion")


# In[ ]:

epochs = range(1, len(train_accuracy_save) + 1)
plt.switch_backend('agg')
figure1 = plt.figure(dpi = 128)
# "bo" is for "blue dot"
plt.plot(epochs, train_accuracy_save, 'bo', label='Training accuracy')
# b is for "solid blue line"
plt.plot(epochs, test_accuracy_save, 'b', label='Test accuracy')
plt.title('Training and test accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
figure1.savefig('../picture/curve_accuracy_34.jpg')
figure2 = plt.figure(dpi = 128)
plt.plot(epochs, train_loss_save, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, test_loss_save, 'b', label='Test loss')
plt.title('Training and test loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
figure2.savefig('../picture/curve_loss_34.jpg')
