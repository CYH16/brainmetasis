import os
import pickle
import numpy as np
import tensorflow as tf
import random
from sklearn.model_selection import KFold

############################################################################################read data

tumor_data_AXC = {}
tumor_data_AX1 = {}
tumor_data_AX2 = {}

with open("X_pickle_1211/label_2015/label_2015", 'rb') as handle:
    tumor_label = pickle.load(handle)
with open("X_pickle_1211/label_2014/label_2014", 'rb') as handle:
    temp = pickle.load(handle)
    for key, value in temp.items():
        tumor_label[key] = value
with open("X_pickle_1211/label_2013/label_2013", 'rb') as handle:
    temp = pickle.load(handle)
    for key, value in temp.items():
        tumor_label[key] = value


for root, dirs, filenames in os.walk("X_pickle_1211/tumor_all/AXC"):
    for i in filenames:
        try:
            with open(root+'/'+i, 'rb') as handle:
                a = pickle.load(handle)
                tumor_data_AXC[i] = a
        except:
            pass
'''
for root, dirs, filenames in os.walk("X_pickle_1211/tumor_all/AX1"):
    for i in filenames:
        try:
            with open(root+'/'+i, 'rb') as handle:
                a = pickle.load(handle)
                tumor_data_AX1[i] = a
        except:
            pass
			
for root, dirs, filenames in os.walk("X_pickle_1211/tumor_all/AX2"):
    for i in filenames:
        try:
            with open(root+'/'+i, 'rb') as handle:
                a = pickle.load(handle)
                tumor_data_AX2[i] = a
        except:
            pass

############################################################################################AX1 AX2 intersection

tumor_label_intersection = {}
for i in tumor_label:
    if i.replace("AXC", "AX1") in tumor_data_AX1 and i.replace("AXC", "AX2") in tumor_data_AX2:
        tumor_label_intersection[i] = tumor_label[i]

############################################################################################equal
'''
tumor_label_intersection = tumor_label
label_count = 0
for i in tumor_label_intersection:
    label_count+=tumor_label_intersection[i]
print(label_count)
print(len(tumor_label_intersection)) 

resp_tumor = []
nore_tumor = []
selected_tumor = []

for tumor, label in tumor_label_intersection.items():
    if label == 1:
        resp_tumor.append(tumor)
    else:
        nore_tumor.append(tumor)

random.shuffle(resp_tumor)
selected_tumor = nore_tumor[:10]
#selected_tumor.extend(resp_tumor[:1])

selected_tumor_label = []
selected_tumor_data_AXC = []
#selected_tumor_data_AX1 = []
#selected_tumor_data_AX2 = []
for i in selected_tumor:
    selected_tumor_label.append(tumor_label_intersection[i])
    selected_tumor_data_AXC.append(tumor_data_AXC[i])
    #selected_tumor_data_AX1.append(tumor_data_AX1[i.replace("AXC", "AX1")])
    #selected_tumor_data_AX2.append(tumor_data_AX2[i.replace("AXC", "AX2")])

############################################################################################pad

#set input shape (125,125,40)

pad_selected_tumor_label = []
pad_selected_tumor_data_AXC = []
pad_selected_tumor_data_AX1 = []
pad_selected_tumor_data_AX2 = []

for i in selected_tumor_label:
    if i == 0:
        pad_selected_tumor_label.append([0,1])
    elif i == 1:
        pad_selected_tumor_label.append([1,0])

for i in selected_tumor_data_AXC:
    i = (i-np.amin(i))/(np.amax(i)-np.amin(i))
    ref = np.zeros((125,125,40))
    ref[:i.shape[0],:i.shape[1],:i.shape[2]] = i
    pad_selected_tumor_data_AXC.append(ref)
'''
for i in selected_tumor_data_AX1:
    i = (i-np.amin(i))/(np.amax(i)-np.amin(i))
    ref = np.zeros((125,125,40))
    ref[:i.shape[0],:i.shape[1],:i.shape[2]] = i
    pad_selected_tumor_data_AX1.append(ref)
	
for i in selected_tumor_data_AX2:
    i = (i-np.amin(i))/(np.amax(i)-np.amin(i))
    ref = np.zeros((125,125,40))
    ref[:i.shape[0],:i.shape[1],:i.shape[2]] = i
    pad_selected_tumor_data_AX2.append(ref)
'''
pad_selected_tumor_label = np.asarray(pad_selected_tumor_label).reshape([10,2])
pad_selected_tumor_data_AXC = np.asarray(pad_selected_tumor_data_AXC).reshape([10,125,125,40,1])
#pad_selected_tumor_data_AX1 = np.asarray(pad_selected_tumor_data_AX1).reshape([160,125,125,40,1])
#pad_selected_tumor_data_AX2 = np.asarray(pad_selected_tumor_data_AX2).reshape([160,125,125,40,1])

#pad_selected_tumor_data = np.concatenate((pad_selected_tumor_data_AXC, pad_selected_tumor_data_AX1, pad_selected_tumor_data_AX2), axis=4)
pad_selected_tumor_data = pad_selected_tumor_data_AXC
print(pad_selected_tumor_data.shape)

############################################################################################shuffle and train/test
def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

shuffle_in_unison(pad_selected_tumor_data, pad_selected_tumor_label)

train_data = pad_selected_tumor_data[2:]
train_label = pad_selected_tumor_label[2:]
test_data = pad_selected_tumor_data[:2]
test_label = pad_selected_tumor_label[:2]

############################################################################################

def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    weight_decay = tf.constant(0.005, dtype=tf.float32)
    W = tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
    return W

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1, 3, 3, 3, 1], padding='SAME')
	
def max_pool_2x2(x):
    return tf.layers.max_pooling3d(x, pool_size=[2,2,2], strides=[2,2,2], padding='SAME')

def bano(x, out_size):
    fc_mean, fc_var = tf.nn.moments(
        x,
        axes=[0, 1, 2, 3],   
    )
    scale = tf.Variable(tf.ones([out_size]))
    shift = tf.Variable(tf.zeros([out_size]))
    epsilon = 0.001
    x_bn = tf.nn.batch_normalization(x, fc_mean, fc_var, shift, scale, epsilon)
    return x_bn
	
xs = tf.placeholder(tf.float32, [None, 625000*1])   # 125x125x40x3
ys = tf.placeholder(tf.float32, [None, 2])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 125, 125, 40, 1])
# print(x_image.shape)  # [n_samples, 125,125,40,1]

## conv1 layer ##
W_conv1 = weight_variable([3,3,3, 1,2], "conv1")
b_conv1 = bias_variable([2])
h_conv1 = tf.nn.relu(conv3d(x_image, W_conv1) + b_conv1)
h_bano1 = bano(h_conv1, 2)
h_pool1 = max_pool_2x2(h_bano1)
print(h_pool1.shape)
#(?, 21, 21, 7, 2)

## conv2 layer ##
W_conv2 = weight_variable([3,3,3, 2, 4], "conv2")
b_conv2 = bias_variable([4])
h_conv2 = tf.nn.relu(conv3d(h_pool1, W_conv2) + b_conv2)
h_bano2 = bano(h_conv2, 4)
h_pool2 = max_pool_2x2(h_bano2)
print(h_pool2.shape)
#(?, 4, 4, 2, 4)

## fc1 layer ##
W_fc1 = weight_variable([4*4*2*4, 64], "fc1")
b_fc1 = bias_variable([64])
h_pool2_flat = tf.reshape(h_pool2, [-1, 4*4*2*4])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## fc2 layer ##
W_fc2 = weight_variable([64, 2], "fc2")
b_fc2 = bias_variable([2])
prediction = tf.nn.sigmoid(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# the error between prediction and real data
reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=ys, logits=prediction)
LOSS = cross_entropy# + 1*sum(reg_losses)
train_step = tf.train.AdamOptimizer(0.001).minimize(LOSS)

accuracy = tf.metrics.accuracy(labels=tf.argmax(ys, axis=1), predictions=tf.argmax(prediction, axis=1),)
auc = tf.metrics.auc(labels=tf.argmax(ys, axis=1), predictions=tf.argmax(prediction, axis=1))

sess = tf.Session()
init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init)
saver = tf.train.Saver()

for i in range(10):
    train_data_split = np.split(train_data, len(train_data))
    train_label_split = np.split(train_label, len(train_label))
    for j, mini_batch in enumerate(train_data_split):
        sess.run(train_step, feed_dict={x_image: mini_batch, ys: train_label_split[j], keep_prob: 1})
        #sess.run(train_step, feed_dict={x_image: train_x, ys: train_y, keep_prob: .5})
    if i % 1 == 0:
        print("Cycle number "+ str(i) + " :")
        print("train accuracy")
        sess.run(tf.local_variables_initializer())
        print(sess.run(accuracy, feed_dict={x_image: train_data, ys: train_label, keep_prob: 1}))	
        print("test accuracy")
        sess.run(tf.local_variables_initializer())
        print(sess.run(accuracy, feed_dict={x_image: test_data, ys: test_label, keep_prob: 1}))	
        print("-----------------------------------------------------------")

"""	
############################################################################################CV
kf = KFold(n_splits=5)
AUC = []
Sensitivity = []
Specificty = []
Dice = []
name = 0

for train_idx, val_idx in kf.split(pad_data_train, label_train):
    sess = tf.Session()
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)
    train_x = pad_data_train[train_idx]
    train_y = label_train[train_idx]
    val_x = pad_data_train[val_idx]
    val_y = label_train[val_idx]
    for i in range(100):
        pad_data_split = np.split(train_x, 4)
        label_split = np.split(train_y, 4)
        for j, mini_batch in enumerate(pad_data_split):
            sess.run(train_step, feed_dict={x_image: mini_batch, ys: label_split[j], keep_prob: .5})
        #sess.run(train_step, feed_dict={x_image: train_x, ys: train_y, keep_prob: .5})
        if i % 20 == 0:
            print("Cycle number "+ str(i) + " :")
            print("train accuracy")
            sess.run(tf.local_variables_initializer())
            print(sess.run(accuracy, feed_dict={x_image: train_x, ys: train_y, keep_prob: 1}))	
            print("validation accuracy")
            sess.run(tf.local_variables_initializer())
            print(sess.run(accuracy, feed_dict={x_image: val_x, ys: val_y, keep_prob: 1}))	
            print("-----------------------------------------------------------")
    AUC.append(sess.run(auc[1], feed_dict={x_image: val_x, ys: val_y, keep_prob: 1}))
    
    label_pred = []
    for i in sess.run(prediction, feed_dict={x_image: val_x, keep_prob: 1}):
        if i[0] >= i[1]:
            label_pred.append([1,0])
        else:
            label_pred.append([0,1])
    res_res = 0
    res_non = 0
    non_res = 0
    non_non = 0
    for i, w in enumerate(val_y):
        if int(w[0]) == 1 and label_pred[i][0] == 1: res_res+=1
        if int(w[0]) == 1 and label_pred[i][0] == 0: res_non+=1
        if int(w[0]) == 0 and label_pred[i][0] == 1: non_res+=1
        if int(w[0]) == 0 and label_pred[i][0] == 0: non_non+=1
    Sensitivity.append(res_res/(res_res+res_non))
    Specificty.append(non_non/(non_res+non_non))
    Dice.append(2*res_res/(2*res_res+non_res+res_non))
    
    save_path = saver.save(sess, "model/model_"+str(name))
    print("Model "+str(name)+" saved.")
    print("-----------------------------------------------------------")
    name += 1
############################################################################################CV result
print("###########################CV##################################")
print("AUC")
print(AUC)
print(sum(AUC)/float(len(AUC)))
print("Sensitivity")
print(Sensitivity)
print(sum(Sensitivity)/float(len(Sensitivity)))
print("Specificty")
print(Specificty)
print(sum(Specificty)/float(len(Specificty)))
print("Dice")
print(Dice)
print(sum(Dice)/float(len(Dice)))

############################################################################################test ensemble
print("###########################test##################################")
AUC = []
Sensitivity = []
Specificty = []
Dice = []

mean_pred = []
final_pred = []

sess = tf.Session()
saver = tf.train.Saver()

for i in range(0,5):
    sess = tf.Session()
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)
    saver.restore(sess, "model/model_"+str(i))
    print("Model "+str(i)+" restored.")
    AUC.append(sess.run(auc[1], feed_dict={x_image: pad_data_test, ys: label_test, keep_prob: 1}))

    label_pred = []
    for i in sess.run(prediction, feed_dict={x_image: pad_data_test, keep_prob: 1}):
        if i[0] >= i[1]:
            label_pred.append([1,0])
        else:
            label_pred.append([0,1])
    res_res = 0
    res_non = 0
    non_res = 0
    non_non = 0
    for i, w in enumerate(label_test):
        if int(w[0]) == 1 and label_pred[i][0] == 1: res_res+=1
        if int(w[0]) == 1 and label_pred[i][0] == 0: res_non+=1
        if int(w[0]) == 0 and label_pred[i][0] == 1: non_res+=1
        if int(w[0]) == 0 and label_pred[i][0] == 0: non_non+=1
    Sensitivity.append(res_res/(res_res+res_non))
    Specificty.append(non_non/(non_res+non_non))
    Dice.append(2*res_res/(2*res_res+non_res+res_non))

    mean_pred.append(np.asarray(label_pred))

mean_pred = np.mean(np.asarray(mean_pred), axis=0)
for i in mean_pred:
    if i[0] >= i[1]:
        final_pred.append([1,0])
    else:
        final_pred.append([0,1])
final_pred = np.asarray(final_pred)

############################################################################################test resulte
print("###########################ENSENBLE##################################")
res_res = 0
res_non = 0
non_res = 0
non_non = 0
for i, w in enumerate(label_test):
    if int(w[0]) == 1 and final_pred[i][0] == 1: res_res+=1
    if int(w[0]) == 1 and final_pred[i][0] == 0: res_non+=1
    if int(w[0]) == 0 and final_pred[i][0] == 1: non_res+=1
    if int(w[0]) == 0 and final_pred[i][0] == 0: non_non+=1
print("Sensitivity")
print(res_res/(res_res+res_non))
print("Specificty")
print(non_non/(non_res+non_non))
print("DICE")
print(2*res_res/(2*res_res+non_res+res_non))

print("###########################EACH MODEL##################################")
print("AUC")
print(AUC)
print(sum(AUC)/float(len(AUC)))
print("Sensitivity")
print(Sensitivity)
print(sum(Sensitivity)/float(len(Sensitivity)))
print("Specificty")
print(Specificty)
print(sum(Specificty)/float(len(Specificty)))
print("Dice")
print(Dice)
print(sum(Dice)/float(len(Dice)))
"""