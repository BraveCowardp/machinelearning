#coding:utf-8

import numpy as np
import os
import gzip
import random
import cv2
from six.moves import urllib
import operator
from datetime import datetime

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

#下载mnist数据集，仿照tensorflow的base.py中的写法。
def maybe_download(filename, path, source_url):
    if not os.path.exists(path):
        os.makedirs(path)
    filepath = os.path.join(path, filename)
    if not os.path.exists(filepath):
        urllib.request.urlretrieve(source_url, filepath)
    return filepath

#按32位读取，主要为读校验码、图片数量、尺寸准备的
#仿照tensorflow的mnist.py写的。
def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]

#抽取图片，并按照需求，可将图片中的灰度值二值化，按照需求，可将二值化后的数据存成矩阵或者张量
#仿照tensorflow中mnist.py写的
def extract_images(input_file, is_value_binary, is_matrix):
    with gzip.open(input_file, 'rb') as zipf:
        magic = _read32(zipf)
        if magic !=2051:
            raise ValueError('Invalid magic number %d in MNIST image file: %s' %(magic, input_file.name))
        num_images = _read32(zipf)
        rows = _read32(zipf)
        cols = _read32(zipf)
        print (magic, num_images, rows, cols)
        buf = zipf.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        if is_matrix:
            data = data.reshape(num_images, rows*cols)
        else:
            data = data.reshape(num_images, rows, cols)
        if is_value_binary:
            return np.minimum(data, 1)
        else:
            return data


#抽取标签
#仿照tensorflow中mnist.py写的
def extract_labels(input_file):
    with gzip.open(input_file, 'rb') as zipf:
        magic = _read32(zipf)
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST label file: %s' % (magic, input_file.name))
        num_items = _read32(zipf)
        buf = zipf.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return labels

# 一般的knn分类，跟全部数据同时计算一般距离，然后找出最小距离的k张图，并找出这k张图片的标签，标签占比最大的为newInput的label
def kNNClassify(newInput, dataSet, labels, k, i):
    numSamples = dataSet.shape[0] # shape[0] stands for the num of row
    init_shape = newInput.shape[0]
    newInput = newInput.reshape(1, init_shape)
    #np.tile(A,B)：重复A B次，相当于重复[A]*B
    #print np.tile(newInput, (numSamples, 1)).shape
    # for i1 in range(3):
    #     temp_data = dataSet[i1].reshape(28,28)
    #     cv2.imshow('temp1', 255*temp_data)
    #     cv2.waitKey(0)
    diff = np.tile(newInput, (numSamples, 1)) - dataSet # Subtract element-wise
    # for i1 in range(3):
    #     temp_data = diff[i1].reshape(28,28)
    #     cv2.imshow('temp1', 255*temp_data)
    #     cv2.waitKey(0)

    squaredDiff = diff ** 2 # squared for the subtract
    squaredDist = np.sum(squaredDiff, axis = 1) # sum is performed by row
    distance = squaredDist ** 0.5
    sortedDistIndices = np.argsort(distance)

    classCount = {} # define a dictionary (can be append element)
    for i in range(k):
        ## step 3: choose the min k distance
        voteLabel = labels[sortedDistIndices[i]]
        # temp_data = dataSet[sortedDistIndices[i]].reshape(28, 28)
        # cv2.imshow('temp1', 255 * temp_data)
        # cv2.waitKey(0)
        # temp_data = diff[sortedDistIndices[i]].reshape(28, 28)
        # cv2.imshow('temp1', 255 * temp_data)
        # cv2.waitKey(0)
        ## step 4: count the times labels occur
        # when the key voteLabel is not in dictionary classCount, get()
        # will return 0
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

    ## step 5: the max voted class will return
    maxCount = 0
    maxIndex = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            maxIndex = key

    return maxIndex


maybe_download('train_images', 'data/mnist', SOURCE_URL+TRAIN_IMAGES)
maybe_download('train_labels', 'data/mnist', SOURCE_URL+TRAIN_LABELS)
maybe_download('test_images', 'data/mnist', SOURCE_URL+TEST_IMAGES)
maybe_download('test_labels', 'data/mnist', SOURCE_URL+TEST_LABELS)



# 主函数，先读图片，然后用于测试手写数字
def testHandWritingClass():
    ## step 1: load data
    print("step 1: load data...")

    train_x = extract_images('data/mnist/train_images', True, True)
    train_y = extract_labels('data/mnist/train_labels')
    test_x = extract_images('data/mnist/test_images', True, True)
    test_y = extract_labels('data/mnist/test_labels')

    ## step 2: training...
    print ("step 2: training...")
    pass

    ## step 3: testing
    print ("step 3: testing...")
    a = datetime.now()
    numTestSamples = test_x.shape[0]
    matchCount = 0
    test_num = numTestSamples//10
    i = random.randint(0,test_num-1)
    temp_data = test_x[i].reshape(28,28)
    # cv2.namedWindow('temp1', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    # cv2.imshow('temp1', 255*temp_data)
    # cv2.waitKey(0)
    predict = kNNClassify(test_x[i], train_x, train_y, 3, i)
    for i in range(test_num):

        predict = kNNClassify(test_x[i], train_x, train_y, 3, i)
        if predict == test_y[i]:
            matchCount += 1
        if i % 100 == 0:
            print ("完成%d张图片"%(i))
    accuracy = float(matchCount) / test_num
    b = datetime.now()
    print ("一共运行了%d秒"%((b-a).seconds))

    ## step 4: show the result
    print ("step 4: show the result...")
    print ('The classify accuracy is: %.2f%%' % (accuracy * 100))

if __name__ == '__main__':
    testHandWritingClass()