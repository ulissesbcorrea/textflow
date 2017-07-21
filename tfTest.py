import numpy as np
from random import randint


numDimensions = 400
maxSeqLength = 200
batchSize = 128
lstmUnits = 64
numClasses = 3
iterations = 100000


wordsList = np.load('wordsList.npy')
print('Loaded the word list!')

wordsList = wordsList.tolist()


wordVectors = np.load('wordVectors.npy')
print('Loaded the word vectors!')

# print(wordVectors[0])
#
# print(len(wordsList))
# print(wordVectors.shape)
#
# print(wordsList[9316])




ids = np.load('idsMatrix.npy')
print('this is the ids shape:')
print(ids.shape)

pos_point = 25751
neu_point = 29074
neg_point = 32707

def getTrainBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength], dtype='int32')
    for i in range(batchSize):
        if (i % 3 == 0):
            num = randint(1, pos_point - 1)
            labels.append([1, 0, 0])
        elif (i % 3 == 1):
            num = randint(pos_point, neu_point - 1)
            labels.append([0, 1, 0])
        else:
            num = randint(neu_point, neg_point - 1)
            labels.append([0, 0, 1])
        arr[i] = ids[num]
    return arr, labels

nextBatch, nextBatchLabels = getTrainBatch()

print('this is the batch shape!')
print(nextBatch.shape)


def print_sent(sent_id):
    sent = ''
    for i in range(120):
        sent += wordsList[sent_id[i]]
    print(sent)

for i in range(10):
    print_sent(nextBatch[i])
    print(nextBatchLabels[i])