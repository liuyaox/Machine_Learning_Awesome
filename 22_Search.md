

# 22. Search

## 22.1 Overview

#### Data

- [谷歌数据集搜索正式版出炉：全面升级，覆盖2500万数据集 - 2020](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650779244&idx=2&sn=725a1adcf7fffc9ff4e041aaed14d696)



## 22.2 KNN & ANN 


## 22.2.1 Overview

KNN: K-Nearest Neighbour

ANN: Approximate Nearest Neighbour

ANN的理念是，在海量数据里，不强求寻找最相似的TopK，而是寻找相似度非常高的TopK即可，在可接受的范围内，牺牲精度以提高效率。

ANN包含了降维和聚类的思想，最常用的2种实现方式：基于哈希函数的LSH，基于向量量化的乘积量化

## 22.2.1 局部敏感哈希(LSH)

LSH: Locality-sensetive Hashing，局部敏感哈希


## 22.2.2 乘积量化(PQ)

PQ: Product Quantization，是向量量化的代表方法

本质理念：将高维向量分解为子空间的笛卡尔积，然后分别量化这些子空间。不再计算原稠密向量的距离，而是用子类中心的索引来表示向量(100维-->4维)，再计算4维向量的L2距离或向量内积。

训练阶段：计算出K^N个类中心

    训练数据：1万样本*100维
    降维：100维切分为4个25维(N=4)，共4个子空间
    聚类：每个子空间聚类出K个类，对应K个类中心向量(25维)，共有4*K个子类中心向量(25维)，聚类方法可用Kmeans
    增维：从4个子空间分别拿出子类中心向量，重新拼接为100维的类中心，则组合数为K*K*K*K=K^4，"乘积"指的就是这种笛卡尔积

预计算阶段：

    索引表示：用这4*K个子类中心的id来表示每个向量（100维向量先划分为4个25维子向量，这4个子向量分别属于子类a,b,c,d，则该向量表示=[a, b, c, d]，只有4维），记为index(.)
    距离计算：4*K个子类中心之间的距离提前计算并存储

应用阶段：主要环节是计算2个向量的距离

    距离计算：查询向量query，候选向量用索引表示为candi=[a, b, c, d]，距离计算记为d(.)
        对称方法：对query进行索引表示为[a1, b1, c1, d1]，其与候选向量的距离=d(index(query), index(candi))=|a1-a|+|b1-b|+|c1-c|+|d1-d|，各子类中心在预计算阶段提前计算好了
        非对称方法：不对query索引表示，直接切分为4个子向量为[v1, v2, v3, v4]，其与候选向量的距离=d(query, index(candi))=|v1-a|+|v2-b|+|v3-c|+|v4-d|，vi与a,b,c,d的距离只能当场计算

简单流程：降维（划分N个子空间）、聚类（每个子空间聚出K个子类）、索引表示（用子类中心的索引来表示每个向量）、距离计算（子向量距离再求和）

### 22.2.2 FAISS

Paper: [Billion-scale similarity search with GPUs - Facebook2017](https://arxiv.org/abs/1702.08734)

FAISS: A library for efficient similarity search and clustering of dense vectors.

用C++高效实现了局部敏感哈希、乘积量化、Kmeans和PCA等算法，在单个服务器的内存中，支持对数十亿的稠密向量进行高效的相似性搜索

#### Library

- <https://github.com/facebookresearch/faiss/>

- **Chinese**: <https://github.com/liqima/faiss_note>

#### Article

- [FAISS Get Started](https://github.com/facebookresearch/faiss/wiki/Getting-started)

- [海量文本求topk相似：faiss库初探](https://mp.weixin.qq.com/s/lS4sn1BFf-kvEKi4Ve74pQ)


## 22.2 Famous

### 22.2.1 PageRank


### 22.2.2 [Real-time Personalization using Embeddings for Search Ranking at Airbnb - Airbnb2018](https://astro.temple.edu/~tua95067/kdd2018.pdf)

**Keywords**: Search Ranking; User Modeling; Personalization

**Key Points**: 生成各种Embedding用于个性化搜索排序

#### Article

- [从KDD 2018 Best Paper看Airbnb实时搜索排序中的Embedding技巧](https://zhuanlan.zhihu.com/p/55149901)


## 22.3 Application

#### Article

- [知乎搜索框背后的Query理解和语义召回技术 - 2020](https://mp.weixin.qq.com/s/4Ns0qbE9d8KZRjFaSUXvRQ)

#### Practice

- <https://github.com/liuhuanyong/QueryCorrection>

    基于拼音相似度与编辑距离的查询纠错，如"手机课"-->"手机壳"

