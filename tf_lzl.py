import tensorflow as tf
import numpy as np
import scipy.io as sio
import os
import functools
import random

def load_data(path):
    files = os.listdir(path)
    train_X = []
    train_Y = []
    for file in files:
        name = file.split('_')[0]
        y = int(name.replace('device','')) - 1 # 设备号0-53
        data = sio.loadmat(os.path.join(path,file))  #导入数据
        X = data['a']
        # X = X[0:39200]  # 取前39200个数据 10M
        # X = X[0:36450]  # 取前39200个数据 9M
        # X = X[0:31250]  # 取前39200个数据  8M
        X = X[0:26450]  # 取前39200个数据  8M
        real = X.real
        imag = X.imag
        X = np.vstack((real,imag))  #竖直方向堆放数据
        train_X.append(X)
        train_Y.append(y)
    train_X = np.array(train_X)   # 生成train_X数组
    train_X = train_X.reshape((train_X.shape[0],train_X.shape[1]))  #train_X.shape[0]:行长度；train_X.shape[1]:列长度
    train_Y = tf.one_hot(train_Y,54,1,0)
    return train_X, train_Y

X_train ,Y_train =load_data(r'D:\Data\new Data\0dB 7M 45组\train')
X_test ,Y_test =load_data(r'D:\Data\new Data\0dB 7M 45组\test')
# sess = tf.InteractiveSession()


def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1,dtype=tf.float32)
    return tf.Variable(initial)


def bias_variable(shape):
    # 创建一个结构为shape矩阵也可以说是数组shape声明其行列，初始化所有值为0.1
    # initial = tf.constant(0.1, shape=shape)
    initial= tf.truncated_normal(shape,stddev=0.1,dtype=tf.float32)
    return tf.Variable(initial)


def conv2d(x, W):
    # 卷积遍历各方向步数为1，SAME：边缘外自动补0，遍历相乘
    # 修改步长的话改中间两个值，水平和垂直滑动值
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 声明一个占位符，None表示输入数据的数量不定，280*280
# xs = tf.placeholder(tf.float32, [None, 78400])
# xs = tf.placeholder(tf.float32, [None, 270*270])
# xs = tf.placeholder(tf.float32, [None, 250*250])
xs = tf.placeholder(tf.float32, [None, 230*230])
# 总共54个设备，对应输出分类结果
ys = tf.placeholder(tf.float32, [None, 54])
keep_prob = tf.placeholder(tf.float32)  #dropout
global_step= tf.Variable(initial_value=0,trainable=False,dtype=tf.int32,name="global_step")
# x_image = tf.reshape(xs, [-1, 280 , 280 , 1])
# x_image = tf.reshape(xs, [-1, 270 , 270 , 1])
# x_image = tf.reshape(xs, [-1, 250 , 250 , 1])
x_image = tf.reshape(xs, [-1, 230 , 230 , 1])

## 第一层卷积操作 ##
# 第一二参数值得卷积核尺寸大小，即patch，第三个参数是图像通道数，第四个参数是卷积核的数目，代表会出现多少个卷积特征图像;
W_conv1 = weight_variable([3, 3, 1, 32])
b_conv1 = bias_variable([32])

def BN(var1,index):
	mean,var=tf.nn.moments(x=var1,axes=[0,1,2,3],keep_dims=True) #这里好像一定要用keep_dims=True
		            # keep_dims: produce moments with the same dimensionality as the input.
	with tf.name_scope("layer%d_norm"%index):
		with tf.variable_scope("layer%d_norm" % index):
		    offset=tf.get_variable(name="offset",shape=[1,var1.get_shape().as_list()[-1]],dtype=tf.float32,
		                                           initializer=tf.zeros_initializer())
		    scale=tf.get_variable(name="scale",shape=[1,var1.get_shape().as_list()[-1]],dtype=tf.float32,
		                                          initializer=tf.ones_initializer())
	conv_out=tf.nn.batch_normalization(var1,mean,var,offset,scale,variance_epsilon=1e-3)
	return conv_out
    # return var1
	
h_conv1= conv2d(x_image, W_conv1) + b_conv1

h_conv1 = tf.nn.relu(BN(h_conv1,1))  # 卷积结果280*280*32
h_pool1 = max_pool_2x2(h_conv1)  #池化结果 140*140*32

## 第二层卷积操作 ##
# 32通道卷积，卷积出64个特征
w_conv2 = weight_variable([3, 3, 32, 32])
# 64个偏执数据
b_conv2 = bias_variable([32])

h_conv2 = tf.nn.relu(BN(conv2d(h_pool1, w_conv2) + b_conv2,2)) #卷积结果 140*140*64
h_pool2 = max_pool_2x2(h_conv2)     # 池化结果 70*70*64

kernel= weight_variable([3,3,32,1])
h_conv3= tf.nn.relu(BN(conv2d(h_pool2,kernel),3))

## 第三层全连接操作 ##
# 二维张量，第一个参数70*70*64的patch，也可以认为是只有一行70*70*64个数据的卷积，
# 第二个参数代表卷积个数共1024个
nums_out= int(functools.reduce(lambda x,y:x*y,h_conv3.get_shape().as_list()[1:]))
W_fc1 = weight_variable([nums_out, 128])
b_fc1 = bias_variable([128])
# 将第二层卷积池化结果reshape成只有一行70*70*64个数据
#  [n_samples, 70, 70, 64] ->> [n_samples, 70*70*64]
h_pool2_flat = tf.reshape(h_conv3, [-1, nums_out])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout操作，减少过拟合
# keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob) #对卷积结果执行dropout操作

## 第四层输出操作 ##
# 二维张量，1*1024矩阵卷积，共54个卷积，对应我们开始的ys长度为54
W_fc2 = weight_variable([128, 54])
b_fc2 = bias_variable([54])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# cross_entropy = -tf.reduce_sum(ys * tf.log(y_conv))  # 定义交叉熵为loss函数
loss= tf.nn.softmax_cross_entropy_with_logits(labels=ys,logits=y_conv)
# tf.nn.sparse_softmax_cross_entropy_with_logits()

tvars= tf.trainable_variables()
loss_L2= 0.0001 * tf.reduce_sum(input_tensor=[tf.nn.l2_loss(x1) for x1 in tvars])
loss= loss+ loss_L2 # L2正则化

loss_to_print= tf.reduce_mean(loss)
train_op_first = tf.train.AdamOptimizer(0.001).minimize(loss,global_step=global_step)

variable_averages=tf.train.ExponentialMovingAverage(decay=0.999,num_updates=global_step)
variable_averages_op=variable_averages.apply(tf.trainable_variables())
with tf.control_dependencies([train_op_first]):
    train_op= tf.group(variable_averages_op) # 应该等效于两个一起group

# tf.global_variables_initializer().run()

# # 测试数据
# # cast(x, dtype, name=None)  将x的数据格式转化成dtype
# # tf.reduce_mean 求平均值
# correct_prediction = tf.equal(tf.argmax(Y_test,1), tf.argmax(y_conv,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# tf.train.batch([example, label], batch_size=batch_size, capacity=capacity)
# [example, label]表示样本和样本标签
# batch_size是返回的一个batch样本集的样本个数
# capacity是队列中的容量。这主要是按顺序组合·    12` -+6
# 3w4`   成一个batch

# X_test.size/280/280 总的测试数量
training_epochs = 1500
batch_Size = 32
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    Y_train, Y_test= sess.run([tf.cast(Y_train,tf.float32),tf.cast(Y_test,tf.float32)])
    train_all= list(zip(X_train,Y_train))
    step  = 0
    for epoch in range(training_epochs):
        print(epoch)
        random.shuffle(train_all)
        train_split = [train_all[x:x + batch_Size] for x in range(0, len(train_all), batch_Size)]
	    # step = 0
	    # print(sess.run([bias, cnn.test]))
        for values in train_split:
            train_batch, train_batch_label = values[0][0].reshape(1,-1), values[0][1].reshape(1, -1)
            for values_in in values[1:]:
                train_batch= np.append(train_batch,values_in[0].reshape(1,-1),axis=0)
                train_batch_label = np.append(train_batch_label, values_in[1].reshape(1, -1), axis=0)
            if (step % 100 == 0) & (step != 0):
                pred,loss_to_see, _ = sess.run([y_conv,loss_to_print,train_op],
                                     feed_dict={xs: train_batch, ys: train_batch_label, keep_prob: 0.8})
                acc= np.mean(np.equal(np.argmax(pred,axis=1),np.argmax(train_batch_label,axis=1)))
                print("traning is step %d,acc is %s,loss is %s\n"%(step,acc,loss_to_see))
                acc_list,loss_to_see_list=[],[]
                for index in range(int(len(Y_test)/100)):
                    Y_test_batch= Y_test[index*100:(index+1)*100]
                    X_test_batch= X_test[index*100:(index+1)*100]
                    pred,loss_to_see= sess.run([y_conv,loss_to_print],
                                feed_dict={xs:X_test_batch,ys:Y_test_batch,keep_prob:1.0})
                    acc = np.mean(np.equal(np.argmax(pred, axis=1), np.argmax(Y_test_batch, axis=1)))
                    acc_list.append(acc)
                    loss_to_see_list.append(loss_to_see)
                print("testing is step %d,acc is %s,loss is %s\n" % (step, np.mean(acc), np.mean(loss_to_see)))
            else:
                sess.run(train_op, feed_dict={xs: train_batch, ys: train_batch_label, keep_prob: 0.8})
                # print(step)
		    # print(sess.run([bias, cnn.test]))
		    # if (step% 100==0) & (step!=0):
			 #    train_pred,_=
            step += 1

        pred, loss_to_see, _ = sess.run([y_conv, loss_to_print, train_op],
                                            feed_dict={xs: train_batch, ys: train_batch_label, keep_prob: 0.8})
        acc = np.mean(np.equal(np.argmax(pred, axis=1), np.argmax(train_batch_label, axis=1)))
        print("traning is step %d,acc is %s,loss is %s\n" % (step, acc, loss_to_see))
        acc_list, loss_to_see_list = [], []
        for index in range(int(len(Y_test) / 100)):
            Y_test_batch = Y_test[index*100:(index + 1) * 100]
            X_test_batch = X_test[index*100:(index + 1) * 100]
            pred, loss_to_see = sess.run([y_conv, loss_to_print],
                                             feed_dict={xs: X_test_batch, ys: Y_test_batch, keep_prob: 1.0})
            acc = np.mean(np.equal(np.argmax(pred, axis=1), np.argmax(Y_test_batch, axis=1)))
            acc_list.append(acc)
            loss_to_see_list.append(loss_to_see)
        print("testing is step %d,acc is %s,loss is %s\n" % (step, np.mean(acc), np.mean(loss_to_see)))



# print("Accuracy:",
#       accuracy.eval({xs: X_test,
#                      ys: Y_test}))