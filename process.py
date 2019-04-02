import re
from itertools import chain
import pandas as pd
import numpy as np
import pickle

# Read origin data
text = open('data/testdata.txt', encoding='utf-8').read()  # txt文件的保存格式应为utf-8，开头字符\ufeff不用管
# Get split sentences
sentences = re.split('[，。！？、‘’“”]/[bems]', text)  # 去掉标点符号，如？/s的形式，斜杠前面是标点，后面是标注
# Filter sentences whose length is 0
sentences = list(filter(lambda x: x.strip(), sentences))   # strip删除开头结尾空白字符，filter过滤序列，返回True的元素放入新列表python3中filter返回的并不是一个list，而是一个filter对象
# Strip sentences
sentences = list(map(lambda x: x.strip(), sentences))    # map() 会根据提供的函数对指定序列做映射

# To numpy array
words, labels = [], []
print('Start creating words and labels...')
for sentence in sentences:
    groups = re.findall('(.)/(.)', sentence)  # list 中包含若干个2个元素的tuple，返回[('人', 'b'), ('们', 'e'), ('常', 's'),....]这种形式
    arrays = np.asarray(groups)   # asarray可以将元组,列表,元组列表,列表元组转化成ndarray对象，若每个元组的size不一样，所以只是一维array，否则二维
    words.append(arrays[:, 0])
    labels.append(arrays[:, 1])
print('Words Length', len(words), 'Labels Length', len(labels))
print('Words Example', words[0])
print('Labels Example', labels[0])

# Merge all words
all_words = list(chain(*words))   # words为二维数组，通过chain和*，将words拆成一维数组
# All words to Series
all_words_sr = pd.Series(all_words)  # 序列化 类似于一维数组的对象，它由一组数据（各种NumPy数据类型）以及一组与之相关的数据标签（即索引）组成。
# Get value count, index changed to set
all_words_counts = all_words_sr.value_counts()  # 计算字频
# Get words set
all_words_set = all_words_counts.index  # index为字，values为字的频数，降序
# Get words ids
all_words_ids = range(1, len(all_words_set) + 1)  # 字典，从1开始


# Dict to transform
word2id = pd.Series(all_words_ids, index=all_words_set)  # 按字频降序建立所有字的索引，字-id
id2word = pd.Series(all_words_set, index=all_words_ids)  # id-字

# Tag set and ids
tags_set = ['x', 's', 'b', 'm', 'e']  # 为解决OOV(Out of Vocabulary)问题，对无效字符标注取零
tags_ids = range(len(tags_set))

# Dict to transform
tag2id = pd.Series(tags_ids, index=tags_set)
id2tag = pd.Series(tags_set, index=tag2id)  # 0-x,1-s,2-b,3-m,4-e

max_length = 32  # 句子最大长度

def x_transform(words):
    # print(words)
    ids = list(word2id[words])
    # print(ids)
    if len(ids) >= max_length:
        ids = ids[:max_length]
    ids.extend([0] * (max_length - len(ids)))
    return ids


def y_transform(tags):
    # print(tags)
    ids = list(tag2id[tags])
    # print(ids)
    if len(ids) >= max_length:
        ids = ids[:max_length]
    ids.extend([0] * (max_length - len(ids)))
    return ids


print('Starting transform...')
# print(words)
data_x = list(map(lambda x: x_transform(x), words))   # 字对应的id的序列，words为二维array，多个seq时map并行处理
data_y = list(map(lambda y: y_transform(y), labels))  # 字对应的标注的id的序列，二维列表

print('Data X Length', len(data_x), 'Data Y Length', len(data_y))
print('Data X Example', data_x[0])
print('Data Y Example', data_y[0])

data_x = np.asarray(data_x)
data_y = np.asarray(data_y)

from os import makedirs
from os.path import exists, join

path = 'data/'

if not exists(path):
    makedirs(path)

print('Starting pickle to file...')
with open(join(path, 'testdata.pkl'), 'wb') as f:
    pickle.dump(data_x, f)  # 序列化对象并追加
    pickle.dump(data_y, f)
    pickle.dump(word2id, f)
    pickle.dump(id2word, f)
    pickle.dump(tag2id, f)
    pickle.dump(id2tag, f)
print('Pickle finished')