import numpy as np
import tensorflow as tf
import datetime
from random import randint

# data parameter
pos_point = 25751
neu_point = 29074
neg_point = 32707
max_seq_length = 200
num_dimensions = 400

# model parameter
lstm_units = 64
num_classes = 3
batch_size = 512
iterations = 100000

word_vectors = np.load('wordVectors.npy')
print('Loaded the word vectors!')
ids = np.load('idsMatrix.npy')
print('Loaded the sentence ids matrix!')


def get_train_batch():
    labels = []
    arr = np.zeros([batch_size, max_seq_length], dtype='int32')
    for i in range(batch_size):
        if i % 3 == 0:
            num = randint(1, pos_point - 1)
            labels.append([1, 0, 0])
        elif i % 3 == 1:
            num = randint(pos_point, neu_point - 1)
            labels.append([0, 1, 0])
        else:
            num = randint(neu_point, neg_point - 1)
            labels.append([0, 0, 1])
        arr[i] = ids[num - 1:num]
    return arr, labels


def train():
    tf.reset_default_graph()
    labels = tf.placeholder(tf.float32, [batch_size, num_classes])
    input_data = tf.placeholder(tf.int32, [batch_size, max_seq_length])
    # data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]), dtype=tf.float32)
    data = tf.nn.embedding_lookup(word_vectors, input_data)
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_units)
    lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=0.75)
    value, _ = tf.nn.dynamic_rnn(lstm_cell, data, dtype=tf.float32)
    weight = tf.Variable(tf.truncated_normal([lstm_units, num_classes]))
    bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))
    value = tf.transpose(value, [1, 0, 2])
    last = tf.gather(value, int(value.get_shape()[0]) - 1)
    prediction = (tf.matmul(last, weight) + bias)
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    sess = tf.InteractiveSession()
    tf.summary.scalar('Loss', loss)
    tf.summary.scalar('Accuracy', accuracy)
    merged = tf.summary.merge_all()
    logdir = 'tensorboard/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '/'
    writer = tf.summary.FileWriter(logdir, sess.graph)

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    for i in range(iterations):
        next_batch, next_batch_labels = get_train_batch()
        sess.run(optimizer, {input_data: next_batch, labels: next_batch_labels})
        if i % 50 == 0:
            summary = sess.run(merged, {input_data: next_batch, labels: next_batch_labels})
            writer.add_summary(summary, i)
            print('step %s' % i)

        if i % 10000 == 0 and i != 0:
            save_path = saver.save(sess, 'new_models/pretrained_lstm.ckpt', global_step=i)
            print('saved to %s' % save_path)
    writer.close()


if __name__ == '__main__':
    train()
