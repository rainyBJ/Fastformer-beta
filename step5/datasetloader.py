'''
save amazon dataset to .npy files
'''

import  os
print(os.getcwd())

import gzip
from nltk.tokenize import wordpunct_tokenize
import numpy as np

# 10万的数据量
# construct sub_dataset 50k: 40k train; 5k val; 5k test
import random
seed = 42
random.seed(seed)
totoal_index = range(1689188)
subset_index = random.sample(totoal_index, 50000)

# 数据集
data_path = "../../dataset/reviews_Electronics_5.json.gz"
review_text = []
review_rating = []
def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield eval(l)

# using first 20 as an example
cnt = -1
for item in parse(data_path):
    cnt = cnt + 1
    if cnt not in subset_index:
        continue
    review_text.append(wordpunct_tokenize(item["reviewText"]))
    review_rating.append(item["overall"]-1)

# train_num; test num; val_num
total_num = len(review_rating)
train_num = int(total_num * 0.8) #40k
val_num = int(total_num * 0.1) #5k
test_num = int(total_num * 0.1) #5k

# construct word_dict
word_dict = {'PADDING': 0}
for sent in review_text:
    for token in sent:
        if token not in word_dict:
            word_dict[token] = len(word_dict)

#  sentence 2 word idxes
news_words = []
for sent in review_text:
    sample = []
    for token in sent: # 把一个句子转换为对应 word 的 idx 序列，句子最多 512 个词
        sample.append(word_dict[token])
    sample = sample[:512] # 超出的部分截取掉
    news_words.append(sample + [0] * (512 - len(sample))) # 不足的部分补零

data = np.array(news_words, dtype='int32')
label = np.array(review_rating, dtype='int32')

# save it to .npy files
np.save("../dataset/data.npy",data)
np.save("../dataset/label.npy",label)
np.save("../dataset/num_tokens",len(word_dict))