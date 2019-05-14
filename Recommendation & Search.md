# Paper Study

## Recommendation

### Overview

Classic papers and resources on recommendation (<https://github.com/wzhe06/Reco-papers>)

推荐系统论文、学习资料、业界分享

### 1. Wide & Deep Learning of Recommender System - Google2016

**Keywords**: Recommender System; Wide & Deep Learning

**url**: <>

#### Yao


#### Article


### 2. Deep Neural Networks for YouTube Recommendations - Google2016

**Keywords**: Recommender System; Deep Learning; Scalability

**url**: <https://research.google.com/pubs/archive/45530.pdf>

#### Yao

信息量非常大，可以学习的点特别多！

##### Matching模块

a. 问题建模：把推荐问题建模成一个**“超大规模多分类”**问题。即在时刻t，为用户U(上下文信息C)，在视频库V中精准地预测出视频i的类别(每个具体的视频就是一个类别，i即为一个类别)。

$$ P(w_t=i|U,C) = \frac{e^{v_{i}u}}{\sum e^{v_{j}u}} $$

很显然上式是一个Softmax多分类器的形式。向量u是\<user, content\>的高维Embedding，向量$v_j$是视频j的Embedding，所以DNN的目标是：在用户信息和上下文信息为输入条件下，**学习用户的Embedding向量u**。注意，非常类似**Word2Vec**的理念！

b. Example Age: 机器学习系统在训练阶段都是利用过去的行为预估未来，因此通常对过去的行为有个隐式的bias。推荐系统产生的视频集合中视频的分布，基本上反映的是**训练所取时间段的平均观看喜好**的视频，因为我们把样本的 “age” 作为一个feature加入模型中。

c. 

d. 

e. 


#### Article

- YouTube深度学习推荐系统的十大工程问题 (<https://zhuanlan.zhihu.com/p/52504407>)

- 关于'Deep Neural Networks for YouTube Recommendations'的一些思考和实现(<https://www.jianshu.com/p/f9d2abc486c9>)

- 论文笔记：Deep neural networks for YouTube recommendations(<https://blog.csdn.net/xiongjiezk/article/details/73445835>)

- Deep Neural Network for YouTube Recommendation论文精读(<https://zhuanlan.zhihu.com/p/25343518>)

- Youtube基于深度学习的视频推荐(<https://www.jianshu.com/p/19ef129fdde2>)


### 3. Audience Expansion for Online Social Network Advertising - LinkedIn2016

**Keywords**: Online Advertising; Audience Expansion; Lookalike Modeling

**URL**: <>

#### Yao


#### Article



## Search

#### 1. Real-time Personalization using Embeddings for Search Ranking at Airbnb

**Keywords**:

**URL**: <https://www.kdd.org/kdd2018/accepted-papers/view/real-time-personalization-using-embeddings-for-search-ranking-at-airbnb>