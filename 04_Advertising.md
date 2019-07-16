
# 4. Advertising

## 4.1 Overview

### 4.1.1 计算广告CTR预估系列 1-9 

from公众号：机器学习荐货情报局

- [计算广告CTR预估系列(一)--DeepFM理论](https://mp.weixin.qq.com/s?__biz=MzU0NDgwNzIwMQ==&mid=2247483673&idx=1&sn=256e57219c8d577c61f25221c346053c)

- [计算广告CTR预估系列(二)--DeepFM实践](https://mp.weixin.qq.com/s?__biz=MzU0NDgwNzIwMQ==&mid=2247483677&idx=1&sn=5bf0ac27124f57553cc8c17aa48664c7)

- [计算广告CTR预估系列(三)--FFM理论与实践](https://mp.weixin.qq.com/s?__biz=MzU0NDgwNzIwMQ==&mid=2247483685&idx=1&sn=36de5b8814c7a1ca5d5a19315b3f1ed1)

- [计算广告CTR预估系列(四)--Wide&Deep理论与实践](https://mp.weixin.qq.com/s?__biz=MzU0NDgwNzIwMQ==&mid=2247483689&idx=1&sn=c6e55677fe4ee1983e8f51fb61dffab5)

- [计算广告CTR预估系列(五)--阿里Deep Interest Network理论](https://mp.weixin.qq.com/s?__biz=MzU0NDgwNzIwMQ==&mid=2247483704&idx=1&sn=2b80e3def93056e4afb39cc1e744d18a)

- [计算广告CTR预估系列(六)--阿里Mixed Logistic Regression](https://mp.weixin.qq.com/s?__biz=MzU0NDgwNzIwMQ==&mid=2247483707&idx=1&sn=5810c525e2880edb795543d5b8bd4aa2)

- [计算广告CTR预估系列(七)--Facebook经典模型LR+GBDT理论与实践](https://mp.weixin.qq.com/s?__biz=MzU0NDgwNzIwMQ==&mid=2247483711&idx=1&sn=14e8d906d84de78b249510b33d423b89)

- [计算广告CTR预估系列(八)--PNN模型理论与实践](https://mp.weixin.qq.com/s?__biz=MzU0NDgwNzIwMQ==&mid=2247483719&idx=1&sn=ab9b912145c94ef299bc8484372794e9)

- [计算广告CTR预估系列(九)--NFM模型理论与实践](https://mp.weixin.qq.com/s?__biz=MzU0NDgwNzIwMQ==&mid=2247483738&idx=1&sn=61334a86c12f027cf6964196b62b3e7e)


## 4.2 Classic

### 4.2.1 [Practical Lessons from Predicting Clicks on Ads at Facebook - Facebook2014](http://quinonero.net/Publications/predicting-clicks-facebook.pdf)

**Keywords**: LR + GBDT

**Key Points**: 利用树模型(GBDT)组合特征的能力自动做特征组合，作为新的特征叠加到LR模型里再训练一个LR模型。

#### Code

- <https://github.com/neal668/LightGBM-GBDT-LR>

#### Article

- [Practical Lessons from Predicting Clicks on Ads at Facebook](http://www.bubuko.com/infodetail-1902390.html)


### 4.2.2 Mixed LR - [Learning Piece-wise Linear Models from Large Scale Data for Ad Click Prediction - Ali2017](https://arxiv.org/abs/1704.05194)

**Keywords**: Mixed LR

**Key Points**: 基于Mixed LR的CTR模型，核心思想是聚类LR

#### Article

- [计算广告CTR预估系列(六)–阿里Mixed Logistic Regression](https://blog.csdn.net/u010352603/article/details/80681239)


### 4.2.3 FTRL - [Ad Click Prediction - A View From the Trenches - Google2013](http://static.googleusercontent.com/media/research.google.com/en/us/pubs/archive/41159.pdf)

**Keywords**: Online Advertising; Data Mining; Large-scale Learning

**Key Points**: FTRL，一种大规模在线学习算法

#### Article

- [各大公司广泛使用的在线学习算法FTRL详解](http://www.cnblogs.com/EE-NovRain/p/3810737.html)


## 4.3 Deep Learning

### 4.3.1 Overview


### 4.3.2 [Audience Expansion for Online Social Network Advertising - LinkedIn2016](https://www.kdd.org/kdd2016/papers/files/adf0483-liuA.pdf)

**Keywords**: Online Advertising; Audience Expansion; Lookalike Modeling

**Key Points**: 


### 4.3.3 DIN - [Deep Interest Network for Click-Through Rate Prediction - Ali2018](https://arxiv.org/abs/1706.06978)

**Keywords**: CTR Prediction; Display Advertising; E-commerce; DIN; Attention Mechanism

**Key Points**: 引入了Attention Mechanism

#### Article

- [推荐系统中的注意力机制——阿里深度兴趣网络(DIN)](https://zhuanlan.zhihu.com/p/51623339)

- [计算广告CTR预估系列(五)--阿里Deep Interest Network理论](https://blog.csdn.net/u010352603/article/details/80590152)


## 4.4 Competition

- <https://github.com/wangle1218/Advertising-algorithm-competition>

    2018 腾讯广告算法大赛/IJCAI 阿里妈妈搜索广告转化预测竞赛/讯飞广告营销算法/OGeek，模型有：LightGBM, LR, DeepFFM

- <https://github.com/DiligentPanda/Tencent_Ads_Algo_2018> (PyTorch)

    Tencent advertisement algorithm competition 2018. Ranked the 3rd place in the final round.

- <https://github.com/BladeCoda/Tencent2017_Final_Coda_Allegro>

    腾讯2017社交广告源码（决赛排名第23位），模型有：LightGBM, XGBoost

- <https://github.com/hengchao0248/ccf2016_sougou> (Keras)

    2016CCF 大数据精准营销中搜狗用户画像挖掘 final winner solution

- <https://github.com/classtag/ijcai18-mama-ads-competition> 

    IJCAI-19 阿里妈妈搜索广告转化预测初赛方案，模型有：LightGBM, CatBoost