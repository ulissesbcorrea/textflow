import numpy as np
import gensim
import time
import jieba
import utils

numDimensions = 400
maxSeqLength = 200
w2v = gensim.models.Word2Vec.load(utils.get_w2v_model_path('tm/tm.model'))


def build_ids():
    words_list = np.load('wordsList.npy')
    words_list = words_list.tolist()
    print('Loaded the word list!')

    pos_f_name = 'multi_pos.txt'
    neg_f_name = 'multi_neg.txt'
    neu_f_name = 'multi_neu.txt'
    pos_lines = open(utils.get_corpus_path(pos_f_name)).readlines()
    neg_lines = open(utils.get_corpus_path(neg_f_name)).readlines()
    neu_lines = open(utils.get_corpus_path(neu_f_name)).readlines()
    pos_cut_lines = list(map(lambda x: ' '.join(jieba.cut(x.replace('\n', ''))), pos_lines))
    neg_cut_lines = list(map(lambda x: ' '.join(jieba.cut(x.replace('\n', ''))), neg_lines))
    neu_cut_lines = list(map(lambda x: ' '.join(jieba.cut(x.replace('\n', ''))), neu_lines))

    sent_num = len(pos_cut_lines) + len(neg_cut_lines) + len(neu_cut_lines)
    ids = np.zeros((sent_num, maxSeqLength), dtype='int32')

    unk_id = len(words_list) - 1
    line_counter = 0
    for line in pos_cut_lines:
        index_counter = 0
        split = line.split()
        for word in split:
            try:
                ids[line_counter][index_counter] = words_list.index(word)
            except ValueError:
                ids[line_counter][index_counter] = unk_id  # Vector for unkown words
            index_counter += 1
            if index_counter >= maxSeqLength:
                break
        line_counter += 1

    pos_point = line_counter

    for line in neu_cut_lines:
        index_counter = 0
        split = line.split()
        for word in split:
            try:
                ids[line_counter][index_counter] = words_list.index(word)
            except ValueError:
                ids[line_counter][index_counter] = unk_id  # Vector for unkown words
            index_counter += 1
            if index_counter >= maxSeqLength:
                break
        line_counter += 1
    neu_point = line_counter

    for line in neg_cut_lines:
        index_counter = 0
        split = line.split()
        for word in split:
            try:
                ids[line_counter][index_counter] = words_list.index(word)
            except ValueError:
                ids[line_counter][index_counter] = unk_id  # Vector for unkown words
            index_counter += 1
            if index_counter >= maxSeqLength:
                break
        line_counter += 1
    neg_point = line_counter

    print(pos_point)  # 25751
    print(neu_point)  # 29074
    print(neg_point)  # 32707
    np.save('idsMatrix', ids)


def build_word_list(pos_f_name, neg_f_name, neu_f_name):
    words = []
    words_list = []
    word_vectors = []

    pos_lines = open(utils.get_corpus_path(pos_f_name)).readlines()
    neg_lines = open(utils.get_corpus_path(neg_f_name)).readlines()
    neu_lines = open(utils.get_corpus_path(neu_f_name)).readlines()
    pos_cut_lines = list(map(lambda x: ' '.join(jieba.cut(x.replace('\n', ''))), pos_lines))
    neg_cut_lines = list(map(lambda x: ' '.join(jieba.cut(x.replace('\n', ''))), neg_lines))
    neu_cut_lines = list(map(lambda x: ' '.join(jieba.cut(x.replace('\n', ''))), neu_lines))

    for line in pos_cut_lines:
        words.extend(line.split(' '))
    for line in neg_cut_lines:
        words.extend(line.split(' '))
    for line in neu_cut_lines:
        words.extend(line.split(' '))

    words = list(set(words))

    words_list.append('this_is_null')
    word_vectors.append(np.array([0] * 400))
    words_list.extend(words)

    for i in range(len(words)):
        if words_list[i] in w2v.vocab:
            word_vectors.append(w2v[words_list[i]])
        else:
            word_vectors.append(np.array([0] * 400))

    words_list.append('UNK')
    word_vectors.append(np.array([0] * 400))

    words_list = np.array(words_list)
    word_vectors = np.array(word_vectors, dtype='float32')

    print('Saving words list')
    np.save('wordsList', words_list)

    print('Saving word vectors')
    np.save('wordVectors', word_vectors)


def test_npy():
    words_list = np.load('wordsList.npy').tolist()
    print(words_list[len(words_list) - 1])


if __name__ == '__main__':
    start_time = time.time()
    # build_word_list('multi_pos.txt', 'multi_neg.txt', 'multi_neu.txt')
    # test_npy()
    build_ids()
    stop_time = time.time()
    print('Time used:', str(stop_time - start_time))
