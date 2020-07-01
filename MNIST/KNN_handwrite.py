# coding:utf-8

import numpy as np
import os
import gzip
from six.moves import urllib
import operator
from datetime import datetime
from matplotlib import pyplot as plt
import random
import cv2

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
TEST_LABELS = 't10k-labels-idx1-ubyte.gz'


# 下载mnist数据集，仿照tensorflow的base.py中的写法。
def maybe_download(filename, path, source_url):
    if not os.path.exists(path):
        os.makedirs(path)
    filepath = os.path.join(path, filename)
    if not os.path.exists(filepath):
        urllib.request.urlretrieve(source_url, filepath)
    return filepath


# 按32位读取，主要为读校验码、图片数量、尺寸准备的
# 仿照tensorflow的mnist.py写的。
def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


# 抽取图片，并按照需求，可将图片中的灰度值二值化，按照需求，可将二值化后的数据存成矩阵或者张量
# 仿照tensorflow中mnist.py写的
def extract_images(input_file, is_value_binary, is_matrix):
    with gzip.open(input_file, 'rb') as zipf:
        magic = _read32(zipf)
        if magic != 2051:
            raise ValueError('Invalid magic number %d in MNIST image file: %s' % (magic, input_file.name))
        num_images = _read32(zipf)
        rows = _read32(zipf)
        cols = _read32(zipf)
        buf = zipf.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows * cols)
        return np.minimum(data,1)


# 抽取标签
# 仿照tensorflow中mnist.py写的
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
def kNNClassify(newInput, dataSet, labels, k, i1):
    numSamples = dataSet.shape[0]  # shape[0] stands for the num of row
    init_shape = newInput.shape[0]
    newInput = newInput.reshape(1, init_shape)
    # np.tile(A,B)：重复A B次，相当于重复[A]*B
    # print np.tile(newInput, (numSamples, 1)).shape
    if i1 == 0:
        im = newInput.reshape(28, 28)
        # plt.subplot(221)
        # plt.imshow(im, cmap=plt.cm.gray)
        # plt.show()

    diff = np.tile(newInput, (numSamples, 1)) - dataSet  # Subtract element-wise
    # if i==0:
    #    for j in range(784):
    #        print(diff[0][j],end='\t')
    #        if (j+1)%28==0:
    #            print('\n')
    # print('\n')
    # print('\n')
    # print('\n')
    # print('\n')
    squaredDiff = diff ** 2  # squared for the subtract
    squaredDist = np.sum(squaredDiff, axis=1)  # sum is performed by row
    distance = squaredDist ** 0.5
    sortedDistIndices = np.argsort(distance)

    classCount = {}  # define a dictionary (can be append element)
    for i in range(k):
        ## step 3: choose the min k distance
        voteLabel = labels[sortedDistIndices[i]]
        # if i1 == 0:
        #     im = dataSet[sortedDistIndices[i]].reshape(28, 28)
        #     im = im*255
        #     cv2.imshow('temp', im)
        #     cv2.waitKey(0)
        #     im = squaredDiff[sortedDistIndices[i]].reshape(28, 28)
        #     im = im*255
        #     cv2.imshow('temp', im)
        #     cv2.waitKey(0)
            # plt.subplot(221);
            # plt.imshow(im, cmap=plt.cm.gray)
            # plt.show()
            # im = squaredDiff[sortedDistIndices[i]].reshape(28, 28)
            # plt.subplot(221);
            # plt.imshow(im, cmap=plt.cm.gray)
            # plt.show()

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
pass
drawing=False
def draw(event,x,y,flags,param):
    global drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        cv2.circle(img,(x,y),1,255,-1)
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(img, (x, y), 1, 255, -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(img,(x,y),1,255,-1)

maybe_download('train_images', 'data/mnist', SOURCE_URL + TRAIN_IMAGES)
maybe_download('train_labels', 'data/mnist', SOURCE_URL + TRAIN_LABELS)
maybe_download('test_images', 'data/mnist', SOURCE_URL + TEST_IMAGES)
maybe_download('test_labels', 'data/mnist', SOURCE_URL + TEST_LABELS)


# 主函数，先读图片，然后用于测试手写数字
def testHandWritingClass():
    ## step 1: load data
    #print("step 1: load data...")

    train_x = extract_images('data/mnist/train_images', True, True)
    im = train_x[0].reshape(28, 28)
    # plt.subplot(221)
    # plt.imshow(im, cmap=plt.cm.gray)
    # plt.show()
    train_y = extract_labels('data/mnist/train_labels')
    # test_x = extract_images('data/mnist/test_images', True, True)
    # test_y = extract_labels('data/mnist/test_labels')

    ##input test data by hand

    cv2.namedWindow('temp', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    while(1):
        cv2.imshow('temp',img)
        k = cv2.waitKey(1) & 0xff
        if k==27:
            break
        cv2.setMouseCallback('temp',draw)
    im=img
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    im_b=np.zeros((28, 140), np.uint8)
    count=[]
    while im.any():
        im = im.transpose()
        im_x,im_y=np.where(im > 0)
        im = im.transpose()
        im_b[im_y[0],im_x[0]]=255
        cv2.namedWindow('temp', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        for i in range(200):
            im_b = cv2.dilate(im_b, kernel, iterations=1)
            im_b = cv2.bitwise_and(im_b, im)
        # cv2.namedWindow('temp', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        # cv2.imshow('temp', im_b)
        # cv2.waitKey(0)
        im_bx,im_by=np.where(im_b>0)
        im[im_bx,im_by]=0
        im_bx = im_bx.reshape(1, -1).T
        im_by = im_by.reshape(1, -1).T
        cnt = np.hstack((im_by, im_bx))
        rect = cv2.minAreaRect(cnt)
        box = np.int0(cv2.boxPoints(rect))
        Xs = [i[0] for i in box]
        Ys = [i[1] for i in box]
        x1 = min(Xs)
        x2 = max(Xs)
        y1 = min(Ys)
        y2 = max(Ys)
        x1 = max(x1,0)
        x2 = min(x2,140)
        y1 = max(y1,0)
        y2 = min(y2,28)
        #print(x1,x2,y1,y2)
        hight = y2 - y1
        width = x2 - x1
        # print(box)
        # box1 = np.array([[x1,y1],[x1,y2],[x2,y2],[x2,y1]])
        # print(box1)
        # cv2.namedWindow('temp', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        # cv2.drawContours(im_b, [box], 0, 255, 1)
        # cv2.drawContours(im_b, [box1], 0, 255, 1)
        # cv2.imshow('temp', im_b)
        # cv2.waitKey(0)
        cropImg = im_b[y1:y2, x1:x2]
        # cv2.namedWindow('temp1', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        # cv2.imshow('temp1',cropImg)
        # cv2.waitKey(0)
        if width >= 28:
            cropImg = cv2.resize(cropImg,(28,hight))
        #print(cropImg.shape)
        # cv2.namedWindow('temp1', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        # cv2.imshow('temp1',cropImg)
        # cv2.waitKey(0)
        left_size=0
        right_size=0
        top_size = (28-(y2-y1))//2
        bottom_size = 28-top_size-(y2-y1)
        if width < 28:
            left_size=(28-(x2-x1))//2
            right_size=28-left_size-(x2-x1)
        #print(top_size, bottom_size, left_size, right_size, cropImg.shape, y2, y1, x2, x1)
        cropImg = cv2.copyMakeBorder(cropImg, top_size, bottom_size, left_size, right_size, cv2.BORDER_CONSTANT, value=0)
        #print(top_size,bottom_size,left_size,right_size,cropImg.shape,y2,y1,x2,x1)
        data = cropImg.reshape(28*28)
        data = np.minimum(data,1)
        predict = kNNClassify(data, train_x, train_y, 3, 0)
        count.append(predict)
        #print("result:%d" % predict)
        #print(cropImg.shape)
        # print(im_b)
        # print(cropImg)
        # cv2.imshow('temp',im_b)
        # cv2.waitKey(0)
        # cv2.namedWindow('temp1', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        # cv2.imshow('temp1',cropImg)
        # cv2.waitKey(0)
    # plt.subplot(221)
    # plt.imshow(im,cmap=plt.cm.gray)
    # plt.show()
    # data = img.reshape(28*28)
    # data = np.minimum(data,1)
    # predict = kNNClassify(data, train_x, train_y, 3, 0)
    # print("result:%d" % predict)
    ## step 2: training...
    # print("step 2: training...")
    print('result:',end='')
    for i in range(len(count)):
        print(count[i],end=' ')
    pass
    # train_num = train_x.shape[0]
    # for i in range(train_num):
    #     if train_y[i] == 6:
    #         im = train_x[i].reshape(28,28)
    #         plt.subplot(221)
    #         plt.imshow(im,cmap=plt.cm.gray)
    #         plt.show()
    ## step 3: testing
    # print("step 3: testing...")
    # a = datetime.now()
    # numTestSamples = test_x.shape[0]
    # test_num = numTestSamples // 10
    # knum = np.arange(1,11)
    # accuracynum = np.arange(0,1,0.1)
    # for k in range(1,11):
    # matchCount = 0
    # for i in range(test_num):
    # i = random.randint(0, test_num)
    #  predict = kNNClassify(test_x[i], train_x, train_y, 3, 0)
    #  if predict == test_y[i]:
    #     matchCount += 1
    # if i % 100 == 0:
    #     print ("完成%d张图片"%(i))
    # accuracy = float(matchCount) / test_num
    # accuracynum[k-1]=accuracy
    # b = datetime.now()
    # plt.title("accuracy about k")
    # plt.xlabel("k")
    # plt.ylabel("accuracy")
    # plt.plot(knum,accuracynum)
    # plt.show()
    # print("一共运行了%d秒" % ((b - a).seconds))
    # print("label:%d" % test_y[i])
    # print("result:%d" % predict)
    ## step 4: show the result
    # print("step 4: show the result...")
    # print ('The classify accuracy is: %.2f%%' % (accuracy * 100))


if __name__ == '__main__':
    img = np.zeros((28, 140), np.uint8)
    testHandWritingClass()