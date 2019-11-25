
# 6. Topic Models

## 6.1 Overview

Topic modeling is an unsupervised learning method that assumes each document consists of a mixture of topics and each topic is a probability distribution over words. The output of topic modeling is a set of word clusters. Each cluster forms a topic and is a probability distribution over words in the document collection.

#### Article

- [百度NLP | Familia：开源的中文主题模型应用工具包](http://baijiahao.baidu.com/s?id=1574779177327287)


## 6.2 LDA

#### Library

- Gensim

    - LDA Model: <https://radimrehurek.com/gensim/models/ldamodel.html>

    - LDA Seq Model: <https://radimrehurek.com/gensim/models/ldaseqmodel.html>

    - LSI Model: <https://radimrehurek.com/gensim/models/lsimodel.html>

    - Author-topic model: <https://radimrehurek.com/gensim/models/atmodel.html>

#### Practice

- <https://github.com/liuhuanyong/TopicCluster>

    基于Kmeans与LDA模型的多文档主题聚类,输入多篇文档,输出每个主题的关键词与相应文本,可用于主题发现与热点分析等应用，如历时话题建模，评论画像等。

- [NLP关键字提取技术之LDA算法原理与实践](https://mp.weixin.qq.com/s?__biz=MzI3ODgwODA2MA==&mid=2247486904&idx=2&sn=aaa7144227625b137c11a655b2aa10da)

- <https://github.com/liuhuanyong/TopicCluster>

    基于Kmeans与Lda模型的多文档主题聚类,输入多篇文档,输出每个主题的关键词与相应文本,可用于主题发现与热点分析等应用，如历时话题建模，评论画像等。

- <https://github.com/GGL12/TextMining>

    某电商手机评论的文本挖掘初体验，内含：LDA模型获取特征词

- <https://github.com/xiaoyichao/-python-gensim-LDA->

    基于python gensim 库的LDA算法 对中文进行文本分析


## 6.3 LSA

LSA = TFIDF + SVD ?


## 6.4 LSI

LDA VS LSA VS LSI ???

