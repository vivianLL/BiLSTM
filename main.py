import argparse
import tensorflow as tf
import pickle
import math
import numpy as np
from sklearn.model_selection import train_test_split
from os.path import join

FLAGS = None


def load_data():
    """
    Load data from pickle
    :return: Arrays
    """
    with open(FLAGS.source_data, 'rb') as f:
        data_x = pickle.load(f)
        data_y = pickle.load(f)
        word2id = pickle.load(f)
        id2word = pickle.load(f)
        tag2id = pickle.load(f)
        id2tag = pickle.load(f)
        return data_x, data_y, word2id, id2word, tag2id, id2tag


def get_data(data_x, data_y):
    """
    Split data from loaded data
    :param data_x:
    :param data_y:
    :return: Arrays
    """
    print('Data X Length', len(data_x), 'Data Y Length', len(data_y))
    print('Data X Example', data_x[0])
    print('Data Y Example', data_y[0])

    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2, random_state=40)
    train_x, dev_x, train_y, dev_y = train_test_split(train_x, train_y, test_size=0.2, random_state=40) # 训练、交叉验证、测试：8:2:2

    print('Train X Shape', train_x.shape, 'Train Y Shape', train_y.shape)
    print('Dev X Shape', dev_x.shape, 'Dev Y Shape', dev_y.shape)
    print('Test Y Shape', test_x.shape, 'Test Y Shape', test_y.shape)
    return train_x, train_y, dev_x, dev_y, test_x, test_y


def weight(shape, stddev=0.1, mean=0):
    initial = tf.truncated_normal(shape=shape, mean=mean, stddev=stddev)
    return tf.Variable(initial)


def bias(shape, value=0.1):
    initial = tf.constant(value=value, shape=shape)
    return tf.Variable(initial)


def lstm_cell(num_units, keep_prob=0.5):
    cell = tf.nn.rnn_cell.LSTMCell(num_units, reuse=tf.AUTO_REUSE)
    return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)


def main():
    # Load data
    data_x, data_y, word2id, id2word, tag2id, id2tag = load_data()
    # Split data
    train_x, train_y, dev_x, dev_y, test_x, test_y = get_data(data_x, data_y)

    # Steps
    train_steps = math.ceil(train_x.shape[0] / FLAGS.train_batch_size)  # ceil向上取整
    dev_steps = math.ceil(dev_x.shape[0] / FLAGS.dev_batch_size)
    test_steps = math.ceil(test_x.shape[0] / FLAGS.test_batch_size)

    vocab_size = len(word2id) + 1
    print('Vocab Size', vocab_size)

    # global_step代表全局步数，比如在多少步该进行什么操作，现在神经网络训练到多少轮等等，类似于一个钟表
    global_step = tf.Variable(-1, trainable=False, name='global_step')  # 使用tensorflow在默认的图中创建节点，这个节点是一个新的变量，-1为初始值。Variable是可更改的（mutable），而Tensor是不可更改的

    # Train and dev dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))  # 函数接受一个数组并返回表示该数组切片的 tf.data.Dataset（即把array的第一维切开）
    train_dataset = train_dataset.batch(FLAGS.train_batch_size)

    dev_dataset = tf.data.Dataset.from_tensor_slices((dev_x, dev_y))
    dev_dataset = dev_dataset.batch(FLAGS.dev_batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y))
    test_dataset = test_dataset.batch(FLAGS.test_batch_size)  # 获取批量样本

    # A reinitializable iterator
    iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes) #  Iterator从不同的Dataset对象中读取数值，可重新初始化，通过from_structure()统一规格

    train_initializer = iterator.make_initializer(train_dataset) # 初始化
    dev_initializer = iterator.make_initializer(dev_dataset)
    test_initializer = iterator.make_initializer(test_dataset)

    # Input Layer
    with tf.variable_scope('inputs'):  # 打开一个已经存在的作用域，与创建/调用变量函数tf.Variable() 和tf.get_variable()搭配使用。
        x, y_label = iterator.get_next()

    # Embedding Layer
    with tf.variable_scope('embedding'):
        embedding = tf.Variable(tf.random_normal([vocab_size, FLAGS.embedding_size]), dtype=tf.float32)
    inputs = tf.nn.embedding_lookup(embedding, x)  # 根据input_ids中的id，寻找embeddings中的第id行。比如input_ids=[1,3,5]，则找出embeddings中第1，3，5行，组成一个tensor返回

    # Variables
    keep_prob = tf.placeholder(tf.float32, [])  # 可以理解为形参，用于定义过程，在执行的时候再赋具体的值，keep_prob 用来表示神经元输出的更新概率

    # RNN Layer
    # cell_fw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell(FLAGS.num_units, keep_prob) for _ in range(FLAGS.num_layer)])
    # cell_bw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell(FLAGS.num_units, keep_prob) for _ in range(FLAGS.num_layer)])
    cell_fw = [lstm_cell(FLAGS.num_units, keep_prob) for _ in range(FLAGS.num_layer)]  # 2层
    cell_bw = [lstm_cell(FLAGS.num_units, keep_prob) for _ in range(FLAGS.num_layer)]
    # initial_state_fw = cell_fw.zero_state(tf.shape(x)[0], tf.float32)
    # initial_state_bw = cell_bw.zero_state(tf.shape(x)[0], tf.float32)
    inputs = tf.unstack(inputs, FLAGS.time_step, axis=1)  # 矩阵分解,time_step即序列本身的长度，即句子最大长度max_length=32
    output, _, _ = tf.contrib.rnn.stack_bidirectional_rnn(cell_fw, cell_bw, inputs=inputs, dtype=tf.float32)  # 创建一个双向循环神经网络。堆叠2个双向rnn层
    # output_fw, _ = tf.nn.dynamic_rnn(cell_fw, inputs=inputs, initial_state=initial_state_fw)
    # output_bw, _ = tf.nn.dynamic_rnn(cell_bw, inputs=inputs, initial_state=initial_state_bw)
    # print('Output Fw, Bw', output_fw, output_bw)
    # output_bw = tf.reverse(output_bw, axis=[1])
    # output = tf.concat([output_fw, output_bw], axis=2)
    output = tf.stack(output, axis=1)  # 矩阵拼接，转化成一个Tensor
    print('Output', output)
    output = tf.reshape(output, [-1, FLAGS.num_units * 2])
    print('Output Reshape', output)

    # Output Layer 全连接
    with tf.variable_scope('outputs'):
        w = weight([FLAGS.num_units * 2, FLAGS.category_num])
        b = bias([FLAGS.category_num])
        y = tf.matmul(output, w) + b  # 矩阵相乘
        y_predict = tf.cast(tf.argmax(y, axis=1), tf.int32)  # cast张量数据类型转换，argmax针对传入函数的axis参数,去选取array中相对应轴元素值大的索引
        print('Output Y', y_predict)

    tf.summary.histogram('y_predict', y_predict)  # 显示直方图信息

    # Reshape y_label
    y_label_reshape = tf.cast(tf.reshape(y_label, [-1]), tf.int32)
    print('Y Label Reshape', y_label_reshape)

    # Prediction
    correct_prediction = tf.equal(y_predict, y_label_reshape)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy) # 显示标量信息

    print('Prediction', correct_prediction, 'Accuracy', accuracy)

    # Loss
    # logits：就是神经网络最后一层的输出,有batch的话，它的大小就是[batchsize，num_classes];labels：实际的标签。因为sparse_softmax_cross_entropy_with_logits返回一个向量，求loss需做reduce_mean，求交叉熵则求reduce_sum
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_label_reshape,
                                                                                  logits=tf.cast(y, tf.float32)))
    tf.summary.scalar('loss', cross_entropy)

    # Train
    train = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy, global_step=global_step)

    # Saver
    saver = tf.train.Saver()

    # Iterator
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Global step
    gstep = 0

    # Summaries
    summaries = tf.summary.merge_all() # 将所有summary全部保存到磁盘，以便tensorboard显示
    # writer = tf.summary.FileWriter(join(FLAGS.summaries_dir, 'train'),
    #                                sess.graph)  # 创建一个FileWrite的类对象，并将计算图写入文件

    if FLAGS.train:

        # if tf.gfile.Exists(FLAGS.summaries_dir):
        #     tf.gfile.DeleteRecursively(FLAGS.summaries_dir)

        for epoch in range(FLAGS.epoch_num):
            tf.train.global_step(sess, global_step_tensor=global_step)
            # Train
            sess.run(train_initializer)
            for step in range(int(train_steps)):
                smrs, loss, acc, gstep, _ = sess.run([summaries, cross_entropy, accuracy, global_step, train],
                                                     feed_dict={keep_prob: FLAGS.keep_prob})
                # Print log
                if step % FLAGS.steps_per_print == 0:
                    print('Global Step', gstep, 'Step', step, 'Train Loss', loss, 'Accuracy', acc)

                # Summaries for tensorboard
                if gstep % FLAGS.steps_per_summary == 0:
                    # writer.add_summary(smrs, gstep)
                    print('Write summaries to', FLAGS.summaries_dir)

            if epoch % FLAGS.epochs_per_dev == 0:
                # Dev
                sess.run(dev_initializer)
                for step in range(int(dev_steps)):
                    if step % FLAGS.steps_per_print == 0:
                        print('Dev Accuracy', sess.run(accuracy, feed_dict={keep_prob: 1}), 'Step', step)

            # # Save model
            # if epoch % FLAGS.epochs_per_save == 0:
            #     saver.save(sess, FLAGS.checkpoint_dir, global_step=gstep)

    else:
        # Load model
        ckpt = tf.train.get_checkpoint_state('ckpt')
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Restore from', ckpt.model_checkpoint_path)
        sess.run(test_initializer)

        for step in range(int(test_steps)):
            x_results, y_predict_results, acc = sess.run([x, y_predict, accuracy], feed_dict={keep_prob: 1})
            print('Test step', step, 'Accuracy', acc)
            y_predict_results = np.reshape(y_predict_results, x_results.shape)
            for i in range(len(x_results)):
                x_result, y_predict_result = list(filter(lambda x: x, x_results[i])), list(
                    filter(lambda x: x, y_predict_results[i]))
                x_text, y_predict_text = ''.join(id2word[x_result].values), ''.join(id2tag[y_predict_result].values)
                print(x_text, y_predict_text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BI LSTM')
    parser.add_argument('--train_batch_size', help='train batch size', default=50)
    parser.add_argument('--dev_batch_size', help='dev batch size', default=50)
    parser.add_argument('--test_batch_size', help='test batch size', default=500)
    parser.add_argument('--source_data', help='source size', default='./data/testdata.pkl')
    parser.add_argument('--num_layer', help='num of layer', default=2, type=int)           # bi-lstm层数
    parser.add_argument('--num_units', help='num of units', default=64, type=int)          # （隐藏层）神经元个数
    parser.add_argument('--time_step', help='time steps', default=32, type=int)
    parser.add_argument('--embedding_size', help='embedding size', default=64, type=int)   # 字向量长度
    parser.add_argument('--category_num', help='category num', default=5, type=int)        # 分类数
    parser.add_argument('--learning_rate', help='learning rate', default=0.01, type=float)
    parser.add_argument('--epoch_num', help='num of epoch', default=500, type=int)         # 1000
    parser.add_argument('--epochs_per_test', help='epochs per test', default=100, type=int)
    parser.add_argument('--epochs_per_dev', help='epochs per dev', default=2, type=int)
    parser.add_argument('--epochs_per_save', help='epochs per save', default=2, type=int)
    parser.add_argument('--steps_per_print', help='steps per print', default=100, type=int)
    parser.add_argument('--steps_per_summary', help='steps per summary', default=100, type=int)
    parser.add_argument('--keep_prob', help='train keep prob dropout', default=0.5, type=float)
    parser.add_argument('--checkpoint_dir', help='checkpoint dir', default='ckpt/model.ckpt', type=str)
    parser.add_argument('--summaries_dir', help='summaries dir', default='summaries', type=str)
    parser.add_argument('--train', help='train', default=False, type=bool)

    FLAGS, args = parser.parse_known_args()

    main()