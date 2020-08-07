# 11. Data Preprocessing

## 11.1 Overview

#### Article

- [What are Some Best Practices in Feature Engineering](https://www.quora.com/What-are-some-best-practices-in-Feature-Engineering)

#### Library

- feature_selector: <https://github.com/WillKoehrsen/feature-selector>

    **特征选择**: A tool for dimensionality reduction of machine learning datasets

    5 Methods to identify features to remove: Missing Values, Single Unique Values, Collinear Features, Zero Importance Features, Low Importance Features.

    **Article**: [A Feature Selection Tool for Machine Learning in Python- 2018](https://towardsdatascience.com/a-feature-selection-tool-for-machine-learning-in-python-b64dd23710f0)

- featexp: <https://github.com/abhayspawar/featexp>

    **特征EDA**: Feature exploration for supervised learning  功能有特征理解，带噪声特征，特征工程，特征重要性，特征debug，泄露检测和理解，模型监控

    **Article**: [My secret sauce to be in top 2% of a kaggle competition - 2018](https://towardsdatascience.com/my-secret-sauce-to-be-in-top-2-of-a-kaggle-competition-57cff0677d3c)

    **Chinese**: [如何达到Kaggle竞赛top 2%？这里有一篇特征探索经验帖](https://www.sohu.com/a/273552971_129720)

- featuretools: <https://github.com/featuretools/featuretools/>

    **Article**: [A Hands-On Guide to Automated Feature Engineering using Featuretools in Python - 2018](https://www.analyticsvidhya.com/blog/2018/08/guide-automated-feature-engineering-featuretools-python/)

    用Python Featuretools库实现自动化特征工程

- <https://github.com/HunterMcGushion/hyperparameter_hunter>

    Easy hyperparameter optimization and automatic result saving across machine learning algorithms and libraries

    **Chinese**: [HyperparameterHunter 3.0：一文教你学会自动化特征工程 - 2019](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650768854&idx=4&sn=f5c6ba9f90e19352ededc43187014b0d)

#### Book

- 【Great】[特征工程入门与实践 - 2019](https://www.ituring.com.cn/book/2606)

       超级赞，有时间一定要好好看看，练就扎实的基本功！

       **Github**: <https://github.com/PacktPublishing/Feature-Engineering-Made-Easy>


## 11.2 Label & Sample

### 11.2.1 Overview

#### Article

- [分类问题-样本权重（sample_weight）和类别权重（class_weight）- 2019](https://zhuanlan.zhihu.com/p/75679299)

    **YAO**: OK

    Label Weight: 
    - 含义1是不同Label误判的代价程度，某些Label误判代价大，则Weight要大；
    - 含义2是类别样本量不均衡，防止模型偏向样本量大的类别，让样本量大的类别权重小一些，比如在sklearn的LR中，若自动计算Label Weight(参数class_weight='balanced')，K个Label的样本数量为ni,(i=1,2,..,K)，总样本量N，则每个Label的权重=N/K*(1/ni)，取决于样本量的倒数

    Sample Weight: 含义是各样本来源不一，某些样本可信度低于其他样本，则应该设置较小的weight

    在sklearn的LR中，设置了label_weight和sample_weight的样本，其最终权重是label_weight * sample_weight，通过损失函数来实现


### 11.2.2 Imbalanced Label

For Label

类别加权后的一个用法是：

#### Article

- [Handling imbalanced datasets in machine learning](https://towardsdatascience.com/handling-imbalanced-datasets-in-machine-learning-7a0e84220f28)

    **Chinese**：[机器学习中如何处理不平衡数据？](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650757216&idx=4&sn=78e370b11becb62d97e35f2c42d1d390)

- [【小夕精选】如何优雅而时髦的解决不均衡分类问题](https://mp.weixin.qq.com/s?__biz=MzIwNzc2NTk0NQ==&mid=2247484993&idx=1&sn=0bd32089a638e5a1b48239656febb6e0)

- [如何处理数据中的「类别不平衡」？](https://zhuanlan.zhihu.com/p/32940093)

- [不均衡学习的抽样方法 - 2016](https://blog.csdn.net/u011414200/article/details/50664266)

- [不平衡数据集的处理 - 2018](https://www.cnblogs.com/kamekin/p/9824294.html)

- [分类中解决类别不平衡问题 - 2020](https://mp.weixin.qq.com/s/NE4JLEGjyCYh7nqYK8W1IQ)

- [分类机器学习中，某一标签占比太大（标签稀疏），如何学习？ - 2020](https://www.zhihu.com/question/372186043)


### 11.2.3 Sample Weights

TODO: adboost, XGBoost里都有，有没有一些通用的做法、library？

样本加权后的一个用法是：对loss加权，即模型训练时，10个样本计算出来的loss=[loss1, loss2, ..., loss10]，weights=[w1, w2, ..., w10]，则最终LOSS=loss * weights

#### Article

- [样本生而不等——聊聊那些对训练数据加权的方法 - 2018](https://zhuanlan.zhihu.com/p/53545036)


### 11.2.4 Label Smoothing

TF和Keras中的Loss函数已经支持，PyTorch暂无官方支持

#### code

- [[PyTorch][Feature Request] Label Smoothing for CrossEntropyLoss - 2018](https://github.com/pytorch/pytorch/issues/7455)



## 11.3 Feature Preprocessing

### 11.3.1 Overview

#### Book & Code

- 【Great】[Feature Engineering for Machine Learning](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/)

    特征工程领域的宝典！

    **Code**: <https://github.com/fengdu78/Data-Science-Notes/tree/master/9.feature-engineering>

#### Article

- [Discover Feature Engineering, How to Engineer Features and How to Get Good at It - 2014](https://machinelearningmastery.com/discover-feature-engineering-how-to-engineer-features-and-how-to-get-good-at-it/)

- [How Feature Engineering can help you do well in a Kaggle competition - Part I - 2017](https://medium.com/unstructured/how-feature-engineering-can-help-you-do-well-in-a-kaggle-competition-part-i-9cc9a883514d)

- [How Feature Engineering can help you do well in a Kaggle competition — Part II - 2017](https://medium.com/unstructured/how-feature-engineering-can-help-you-do-well-in-a-kaggle-competition-part-ii-3645d92282b8)

- [How Feature Engineering can help you do well in a Kaggle competition - 2017](https://medium.com/unstructured/how-feature-engineering-can-help-you-do-well-in-a-kaggle-competition-part-iii-f67567aaf57c)


### 11.3.2 Standardization & Normalization

#### Article

- 【Great】[About Feature Scaling and Normalization – and the effect of standardization for machine learning algorithms - 2014](http://sebastianraschka.com/Articles/2014_about_feature_scaling.html)

- [关于标准化和归一化的一切 - 2019](https://mp.weixin.qq.com/s?__biz=MzIwODI2NDkxNQ==&mid=2247484804&idx=3&sn=74e2179759b1a3a5f9b8c3434cdbfb01)

- [标准化和归一化什么区别 - 2018](https://www.zhihu.com/question/20467170)


### 11.3.3 Onehot Encoding

#### Article

- [数据预处理：独热编码（One-Hot Encoding）- 2017](https://blog.csdn.net/pipisorry/article/details/61193868)


## 11.4 Feature Selection

#### Article

- [文本挖掘之特征选择(python 实现) - 2013](https://www.cnblogs.com/wangbogong/p/3251132.html)

    DF, MI, IG, CHI，WLLR,WFO

- [An end to end guide on how to reduce the number of Features in a Dataset with practical examples in Python - 2019](https://towardsdatascience.com/feature-selection-techniques-1bfab5fe0784)

    **Chinese**: [特征选择怎么做？这篇文章告诉你](https://mp.weixin.qq.com/s?__biz=MzU3NjE4NjQ4MA==&mid=2247486261&idx=4&sn=04a6e216468b8429e38f580089dd9f82)


## 11.5 Dimensionality Reduction

#### Code

- <https://github.com/heucoder/dimensionality_reduction_alo_codes>

    PCA、LDA、MDS、LLE、TSNE等降维算法的python实现

#### Article

- [十个技巧，让你成为“降维”专家 - 2019](https://mp.weixin.qq.com/s?__biz=MzIwOTc2MTUyMg==&mid=2247490939&idx=3&sn=d4e3a04c6b94a93c35bc0264bb056213)

