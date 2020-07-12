# 6. Learn To Rank

## 6.1 Overview

#### Paper

- [From RankNet to LambdaRank to LambdaMART: An Overview - Microsoft2010](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf)

- [Learning to Rank Using Gradient Descent - Microsoft2005](https://icml.cc/2015/wp-content/uploads/2015/06/icml_ranking.pdf)

- [Learning to Rank Short Text Pairs with Convolutional Deep Neural Networks - Google2015](http://disi.unitn.it/moschitti/since2013/2015_SIGIR_Severyn_LearningRankShort.pdf)

    **Code**: <https://github.com/zhangzibin/PairCNN-Ranking> (Tensorflow)


#### Library

- RankLib: <https://sourceforge.net/p/lemur/wiki/RankLib/>


#### Article

- [Learning to Rank Overview - 2015](https://wellecks.wordpress.com/2015/01/15/learning-to-rank-overview/)

- [学习排序 Learning to Rank：从 pointwise 和 pairwise 到 listwise，经典模型与优缺点 - 2018](https://blog.csdn.net/lipengcn/article/details/80373744)


## 6.2 Metric

搜索词是query，不同搜索模型的结果列表有Res1, Res2, Res3, ...，或者没有query，只是单纯的一堆结果列表

- NDCG: 对Resi的一个度量，用于对比各个Resi之间的优劣
    - G: Gain，增益，当前item与query的相关性reli(介于0到1，或01二值)，或者当前item本身的得分scorei(可以大于1)，则有2种形式：
        - **Gi=reli或Gi=scorei**
        - **Gi=2^Reli-1或G=2^scorei-1**，工业界有时会这么做，当reli或scorei取01二值时，这两者结果一样
    - DCG: Discounted Cumulative Gain，对Gi加权求和，各Position的权重按Position进行**Log调和级数衰减**，**DCG=sum_{i=1}^{i=K}(Gi/log2(i+1))**
        - 若结果列表长度K固定，则DCG@K便可用于对比结果列表的优劣 **TODO**
        - 若结果列表长度不固定，则需要使用NDCG
    - IDCG: Ideal DCG, 理想情况下最大的DCG，也有2种形式：
        - 对每个结果列表resi，令其各item倒序排序为resi_sorted，**IDCG=DCG(resi_sorted)**
        - 在所有结果列表所在的整体item空间中，取最优的前K个item生成res，**IDCG=DCG(res)**
        - **TODO**可能还有1种形式：若有看齐的最理想结果列表，比如多个Resi都需要向Res看齐，则IDCG来自于Res(只不过K基于Resi长度而变化)?
    - NDCG: Normalized DCG, 归一化后的DCG，介于01之间，**NDCG=DCG/IDCG**
    - 工业界：要考虑到Position分段，以及权重方法
        - Position分段：各Position分段，同一段内权重相同，比如10个Position分为3段：[1, 2, 3], [4, 5, 6], [7, 8, 9, 10]
        - 权重方法：共有3大类方法
            - 方法1-基于对数衰减：1/logM(i+M-1)，同上文所述，为了调整衰减，可以取M=1.05, 1.5, 2, e, 3, 5等
            - 方法2-基于业务数据：各Position所承载的业务量，或所吸引的流量，或XXX
            - 方法3-基于Eye Tracking: 可行度较低，一般使用方法2来模拟，参考[Eyetracking in Online Search](https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/34378.pdf)

- MAP: 


- Precision@K:




#### Article

- [IR的评价指标-MAP，MRR和NDCG的形象理解 - 2018](https://blog.csdn.net/anshuai_aw1/article/details/83117012)

    **YAO**: 讨论到了NDCG中，业内**在G、D、N上的差异**
