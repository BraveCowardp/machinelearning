import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score,roc_curve,auc
from sklearn.model_selection import train_test_split
from  sklearn.linear_model import SGDClassifier
from sklearn.metrics import  classification_report
def sigmoid(x):
    result = 1.0/(1+np.exp(-x))
    return result

def loss(xMat,yMat,ws,nor):
    left = np.multiply(yMat,np.log(sigmoid(xMat*ws)+0.000000001))
    right = np.multiply(1-yMat,np.log(1-sigmoid(xMat*ws)+0.000000001))
    reg = nor * np.sum(np.square(ws)) / 2
    return (np.sum(left+right) + reg)/-(len(xMat))

def gradAscent(xArr,yArr,test_x,test_y,lr,epochs,nor):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)

    m,n=np.shape(xMat)
    ws = np.mat(np.ones((n,1)))


    lossList = []
    lossListy = []
    for i in range(epochs + 1):
        h=sigmoid(xMat*ws)
        ws_grad = xMat.T * (h-yMat) / m - (nor * ws) / m
        ws = ws - lr*ws_grad

        if i % 5 == 0:
            lossList.append(loss(xMat,yMat,ws,nor))
            lossListy.append(loss(test_x,test_y,ws,nor))
    return ws,lossList,lossListy
def predict(test_x,test_y,ws):
    xMat = np.mat(test_x)
    ws = np.mat(ws)
    result = [1 if x >= 0.5 else 0 for x in sigmoid(xMat*ws)]
    m,n=np.shape(xMat)
    num = 0
    for i in range(m):
        if result[i]==test_y[i]:
            num+=1
    return num/m

dataset = pd.read_csv('data/income.csv',header=None)
# 归一化处理
num_mean = dataset[list(range(1,58))].mean()
num_std = dataset[list(range(1,58))].std()
num_normol = (dataset[list(range(1,58))] - num_mean)/num_std
dataset.drop(columns=list(range(1,58)),inplace=True)
dataset = pd.concat([dataset,num_normol],axis=1)
# 划分x和y
df_x = dataset.drop(columns=[0,58])
df_y = dataset[[58]]
# 划分训练集和测试集并添加常数
train_x = df_x[0:3000].values
train_x = np.concatenate((np.ones((3000,1)),train_x),axis=1)
train_y = df_y[0:3000].values

test_x = df_x[3000:4000].values
test_x = np.concatenate((np.ones((1000,1)),test_x),axis=1)
test_y = df_y[3000:4000].values

lrs = np.array([0.01,0.1,1,10,20])          # 学习率
nor =  0.001                                  # 正则化常数
epochs = 1000                               # 迭代次数
i=0
lossLists = []
lossListys = []
accuracyList = []
for lr in lrs:
    ws, lossList,lossListy = gradAscent(train_x, train_y,test_x,test_y, lr, epochs, nor)
    lossLists.append([])
    lossLists[i] = lossList
    lossListys.append([])
    lossListys[i] = lossListy
    i+=1
    predictions = predict(test_x, test_y, ws)
    accuracyList.append(predictions)
# print(np.shape(costLists))


fig,ax = plt.subplots(3,1)
x = np.linspace(0,epochs,(epochs//5)+1)
ax[0].plot(x,lossLists[0],label='lr=0.01')
ax[0].plot(x,lossLists[1],label='lr=0.1')
ax[0].plot(x,lossLists[2],label='lr=1')
ax[0].plot(x,lossLists[3],label='lr=10')
ax[0].plot(x,lossLists[4],label='lr=20')
ax[0].legend()
ax[0].set_title('Train')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('loss')
ax[1].plot(x,lossListys[0],label='lr=0.01')
ax[1].plot(x,lossListys[1],label='lr=0.1')
ax[1].plot(x,lossListys[2],label='lr=1')
ax[1].plot(x,lossListys[3],label='lr=10')
ax[1].plot(x,lossListys[4],label='lr=20')
ax[1].legend()
ax[1].set_title('Test')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('loss')
ax[2].scatter(lrs,accuracyList)
ax[2].set_title('Test')
ax[2].set_xlabel('lrs')
ax[2].set_ylabel('accuracy')
plt.show()