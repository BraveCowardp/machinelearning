import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
# 加载数据集
df = pd.read_csv("data/spam.csv", encoding="latin-1")
df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
# 划分训练集和测试集
data,label = df['v2'],df['v1']
data_train,data_test,label_train,label_test = train_test_split(data,label)
# 构建词袋
vectorizer = CountVectorizer()
data_train_cnt = vectorizer.fit_transform(data_train)
data_test_cnt = vectorizer.transform(data_test)
# 贝叶斯分类
clf = MultinomialNB()
clf.fit(data_train_cnt, label_train)
predict = clf.predict(data_test_cnt)
accuracy = accuracy_score(predict,label_test)
print("accuracy=%f"%accuracy)