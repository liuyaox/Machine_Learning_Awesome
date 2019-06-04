
# Recommendation

## Overview


## Deep Learning

### 0. Overview

a. [Classic papers and resources on recommendation](<https://github.com/wzhe06/Reco-papers>)

#### Yao

推荐系统论文、学习资料、业界分享。

### 1. [Wide & Deep Learning of Recommender System - Google2016](https://arxiv.org/abs/1606.07792)

**Keywords**: Recommender System; Wide & Deep Learning

#### Yao

Wide属于广义线性模型，用于Memorization，Deep是深度模型，用于Generalization，Memorization可修正Generalization，Wide & Deep 就是希望计算机可以像人脑一样，同时发挥Memorization和Generalization的作用。

#### Source

- **Github**: <https://github.com/tensorflow/models/tree/master/official/wide_deep>

#### Article

- [Wide & Deep Learning: Better Together with TensorFlow](https://ai.googleblog.com/2016/06/wide-deep-learning-better-together-with.html)


### 2. [Deep Neural Networks for YouTube Recommendations - Google2016](https://research.google.com/pubs/archive/45530.pdf)

**Keywords**: Recommender System; Deep Learning; Scalability

#### Yao

信息量非常大，可以学习的点特别多！

a. 问题建模：把推荐问题建模成一个**“超大规模多分类”**问题。即在时刻t，为用户U(上下文信息C)，在视频库V中精准地预测出视频i的类别(每个具体的视频就是一个类别，i即为一个类别)。

$$ P(w_t=i|U,C) = \frac{e^{v_{i}u}}{\sum e^{v_{j}u}} $$

很显然上式是一个Softmax多分类器的形式。向量u是\<user, content\>的高维Embedding，向量$v_j$是视频j的Embedding，所以DNN的目标是：在用户信息和上下文信息为输入条件下，**学习用户的Embedding向量u**。注意，非常类似**Word2Vec**的理念！

b. Example Age: 机器学习系统在训练阶段都是利用过去的行为预估未来，因此通常对过去的行为有个隐式的bias。推荐系统产生的视频集合中视频的分布，基本上反映的是**训练所取时间段的平均观看喜好**的视频，因为我们把样本的 “age” 作为一个feature加入模型中。

c. 细节太多，受启发的也太多！

#### Article

- [重读Youtube深度学习推荐系统论文，字字珠玑，惊为神文](https://zhuanlan.zhihu.com/p/52169807)

- [YouTube深度学习推荐系统的十大工程问题](https://zhuanlan.zhihu.com/p/52504407)

- [揭开YouTube深度推荐系统模型Serving之谜](https://zhuanlan.zhihu.com/p/61827629)

- [关于'Deep Neural Networks for YouTube Recommendations'的一些思考和实现(实际实现的总结)](https://www.jianshu.com/p/f9d2abc486c9)

- [Deep Neural Network for YouTube Recommendation论文精读](https://zhuanlan.zhihu.com/p/25343518)

- [Youtube基于深度学习的视频推荐](https://www.jianshu.com/p/19ef129fdde2)

#### Source

- **Github**: <https://github.com/yangxudong/deeplearning/tree/master/youtube_match_model>

- **Github**: <https://github.com/QingqingSUN/YoutubeNet>

### 3. [Latent Cross: Making Use of Context in Recurrent Recommender Systems](http://alexbeutel.com/papers/wsdm2018_latent_cross.pdf)

#### Yao

基于RNN来使用时序Context信息，比如浏览和搜索历史记录中的时序信息，应用于Youtube推荐。


### Newest

#### 1. [Generative Adversarial User Model for Reinforcement Learning Based Recommendation System - Alibaba2019](https://arxiv.org/abs/1812.10613)

**Keywords**: 

**Article**: [强化学习用于推荐系统，蚂蚁金服提出生成对抗用户模型](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650763260&idx=3&sn=ae589196211189a8aba6f56a11e2cccb&chksm=871aab82b06d22942d37c9b6efe33cd9647050599f293e59ffc8ababeedb9f8c80afbd80b509&mpshare=1&scene=1&srcid=&key=cb3ec8ead4de2b57a5408b541f10e3f0ecd245330456cb144715b99337220c527dcebbf0949b3c699fe5e195aa470f536bfe22b312054cd619966a7463b357fd5163f83998ec968509053ad0064f1925&ascene=1&uin=MjcwMjE1Nzk1&devicetype=Windows+7&version=62060739&lang=en&pass_ticket=JI1MPoW%2BxCBIF0Z4bW%2BdFwp%2F0VZpd29%2Fw5MfRsN9cbQTsYET60iGog3kaYxwyraC)