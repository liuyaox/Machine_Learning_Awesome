
# 1. Machine Learning Algorithm

## 1.1 Overview

#### Code

- <https://github.com/yhangf/ML-NOTE>

    整理所学的机器学习算法，并根据自己所理解的样子叙述出来。(注重数学推导)


## 1.2 Linear Regression


## 1.3 Logistic Regression

#### Article

- [SGDClassifier和LR,SVM的区别 - 2016](https://blog.csdn.net/tianbwin2995/article/details/51853869)

    **YAO**: LR中参数更新是通过增加负样本权重，让样本重新排列，不是梯度下降法！SGDClassifier中的LR是通过梯度下降法。


## 1.4 Naive Bayesian

#### Article

- [用朴素贝叶斯进行文本分类 - 2016](http://www.sohu.com/a/57924447_308467)


## 1.8 GBDT & XGBoost & LightGBM

#### Article

- [机器学习笔记（七）Boost算法（GDBT,AdaBoost，XGBoost）原理及实践 - 2017](https://blog.csdn.net/sinat_22594309/article/details/60957594)

    **YAO**:

    因为是加性模型$J=\sum L(y_i, f_{m-1}(x) + b(x;\gamma_m))$
    
    - 自然而然可以想到，若让损失函数最小化，让$b(x;\gamma_m)$为L对当前模型$f_{m-1}$的负梯度，损失函数不就最快下降了嘛，即：

        $b(x;\gamma_m) = -\lambda \partial L(y_i, f_{m-1}) / \partial f$

    - 对于XGBoost，对L泰勒展开到二阶项为：$L = L(y_i,f_{m-1}) + g_ib(x) + h_ib^2(x)/2$，其中，$g_i = \partial L(y_i,f_{m-1})/\partial f$，$h_i = \partial^2 L(y_i, f_{m-1})/\partial f^2$

        因为基模型是决策树，所以$b(x;\gamma_m) = \sum w_jI(x \in R_j)$，代入上式L及其正则项，对$w_j$求导为0可得$w_j=-G_j/(H_j + \lambda)$，其中$G_j = \sum g_i$，$H_j = \sum h_i$

        把$w_j$代入原式子就把当前步的损失函数变成了只与上一步相关的一个新的损失函数，然后遍历数据中所有分割点，寻找新的损失函数下降最多的分割点，不断重复上述操作。

- [RF、GBDT、XGBoost面试级整理 - 2018](https://blog.csdn.net/meyh0x5vdtk48p2/article/details/79276307)

- [Understanding Machine Learning: XGBoost - 2017](https://blogs.ancestry.com/ancestry/2017/12/18/understanding-machine-learning-xgboost/)

- [为什么XGBoost在机器学习竞赛中表现如此卓越？ - 2017](https://blog.csdn.net/Uwr44UOuQcNsUQb60zk2/article/details/78495763)

- [XGBoost与深度学习到底孰优孰劣？都说XGBoost好用，为什么名气总不如深度学习？ - 2017](https://www.codercto.com/a/5669.html)

- [CatBoost vs. Light GBM vs. XGBoost - 2018](https://towardsdatascience.com/catboost-vs-light-gbm-vs-xgboost-5f93620723db)

    **Chinese**: [从结构到性能，一文概述XGBoost、Light GBM和CatBoost的同与不同](https://mp.weixin.qq.com/s?__biz=MzUxNjcxMjQxNg==&mid=2247491325&idx=4&sn=5ed726c8a3560a0eac1413a17e56b9cb)

- [一文详尽系列之CatBoost - 2019](https://mp.weixin.qq.com/s?__biz=MzIwODI2NDkxNQ==&mid=2247486708&idx=3&sn=9cf831ba8db248b4d708a375daddd122)


## 1.9 Clustering

### 1.9.1 Overview

#### Article

- [Choosing the Right Clustering Algorithm for your Dataset - 2019](https://www.kdnuggets.com/2019/10/right-clustering-algorithm.html)

    **Chinese**: [4种基本聚类算法应如何正确选择？这份攻略值得你收藏](https://mp.weixin.qq.com/s/xCIEWc2KpsjMixXHrzZ1rA)


## 1.10 Dimensionality Reduction

#### Code

- <https://github.com/eliorc/Medium/blob/master/PCA-tSNE-AE.ipynb> (Tensorflow)

    解读了最常见的三大降维技术：PCA、t-SNE 和自编码器

#### Article

- [Comprehensive Guide on t-SNE algorithm with implementation in R & Python - 2017](https://www.analyticsvidhya.com/blog/2017/01/t-sne-implementation-r-python/)

    **Chinese**: [还在用PCA降维？快学学大牛最爱的t-SNE算法吧](https://blog.csdn.net/dzjx2eotaa24adr/article/details/79132339)