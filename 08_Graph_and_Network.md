# 8. Graph and Network

Attentions: This is about Graph, Network, NOT Image, Picture, Computer Vision, etc.

## 8.1 Overview

#### Paper

- [Learning Representations of Graph Data -- A Survey - UCL2019](https://arxiv.org/abs/1906.02989)

    综述：总结和探讨图数据表征学习方法的最新进展。

- <https://github.com/thunlp/GNNPapers>

    Must-read papers on GNN

- <https://github.com/deepgraphlearning/literaturedl4graph>

    Paper list about deep learning for graphs  涉及：Node Representation, KG Embedding, GNN, Application, Graph Generation.

- <https://github.com/benedekrozemberczki/awesome-graph-classification>

    A collection of important graph embedding, classification and representation learning papers with implementations.

- <https://github.com/talorwu/Graph-Neural-Network-Review>

    GNN综述

#### Library

- [「紫禁之巅」四大图神经网络架构 - 2020](https://mp.weixin.qq.com/s/5dGxP0gjoeiaBlPy85NaPw)

#### Article

- [深度学习中不得不学的 Graph Embedding 方法](https://zhuanlan.zhihu.com/p/64200072)

- [阿里凑单算法首次公开！基于Graph Embedding的打包购商品挖掘系统解析 - 2018](https://mp.weixin.qq.com/s/diIzbc0tpCW4xhbIQu8mCw)


## 8.2 Node2Vec

Node2Vec belongs to Graph Embedding. Detailed info about Graph Embedding is at <>.

[node2vec: Scalable Feature Learning for Networks - Stanford2016](https://arxiv.org/abs/1607.00653)

node2vec is an algorithmic framework for learning continuous feature representations for nodes in networks. In node2vec, we learn a mapping of nodes to a low-dimensional space of features that maximizes the likelihood of preserving network neighborhoods of nodes.

node2vec主要用于处理网络结构中的多分类和链路预测任务，具体来说是对网络中的节点和边的特征向量表示方法。简单点来说就是将原有社交网络中的图结构，表达成特征向量矩阵，每一个node(人、物或内容等)表示成一个特征向量，用向量与向量之间的矩阵运算来得到相互的关系。

#### Code

- <http://snap.stanford.edu/node2vec/>

- <https://github.com/aditya-grover/node2vec>

#### Article

- [关于Node2vec算法中Graph Embedding同质性和结构性的进一步探讨](https://zhuanlan.zhihu.com/p/64756917)

- [node2vec: Embeddings for Graph Data](https://towardsdatascience.com/node2vec-embeddings-for-graph-data-32a866340fef)



## 8.3 Graph Neural Network

#### Paper

- [Graph convolutional networks: a comprehensive review - 2019](https://link.springer.com/article/10.1186/s40649-019-0069-y)

- [Graph Convolutional Networks for Text Classification](https://arxiv.org/abs/1809.05679)

    **Article**: [论文总结】TextGCN - 2020](https://zhuanlan.zhihu.com/p/111945052)

#### Code

- <https://github.com/tkipf/gcn> (Tensorflow)

#### Article

- 从图(Graph)到图卷积(Graph Convolution): 漫谈图神经网络  [一](https://zhuanlan.zhihu.com/p/108294485), [二](https://zhuanlan.zhihu.com/p/108298787), [三](https://zhuanlan.zhihu.com/p/108299847)

- [目前看的GNN论文的一些总结 - 2020](https://zhuanlan.zhihu.com/p/129308686)

- [PinSAGE：GCN 在工业级推荐系统中的应用 - 2020](https://mp.weixin.qq.com/s/37PLr40OV_R_XzUpM2aeZQ)
