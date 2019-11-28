
# 2. Data Mining and Algorithm

## 2.1 Overview


## 2.3 Similarity

欧式距离、曼哈顿距离、余弦距离、相关系数这些easy的就不说了

#### Article

- [概率分布之间的距离度量以及python实现 - 2017](https://www.cnblogs.com/wt869054461/p/7156397.html)


### 2.3.1 Histogram & Distribution Similarity

直方图Similarity，有时也适用于计算概率分布的Similarity

**Bin-by-bin**

相同索引的bin要一一对应，要求2个直方图的bin索引和个数完全一样

#### Metric

Correlation, Chi-Square, Alternative Chi-Squre, Intersection, Bhattacharyya Distance, Kullback-Leibler Divergence (KL散度，亦即相对熵)

#### Library

OpenCV: [compareHist](https://docs.opencv.org/3.0-beta/modules/imgproc/doc/histograms.html#comparehist)

#### Article

- [OpenCV: Histogram Comparison](https://docs.opencv.org/3.0-beta/doc/tutorials/imgproc/histograms/histogram_comparison/histogram_comparison.html)

#### Code

```python
import cv2
import numpy as np
from sklearn.preprocessing import minmax_scale

# Histogram Array
h1 = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)   # 需要指定为float32类型，否则报错
h2 = np.array([2, 3, 0, 5, 6, 7], dtype=np.float32)

# MinMax归一化 参考Article中的normalize
h1_n = minmax_scale(h1)
h2_n = minmax_scale(h2)

# 遍历各种Metricss
methods = [(cv2.HISTCMP_CORREL, 0, '相关系数'), (cv2.HISTCMP_CHISQR, 1, '卡方'), 
           (cv2.HISTCMP_INTERSECT, 2, '十字'), (cv2.HISTCMP_BHATTACHARYYA, 3, '巴氏系数'), 
           (cv2.HISTCMP_HELLINGER, 3, '同巴氏系数'), (cv2.HISTCMP_CHISQR_ALT, 4, '调整的卡方'), 
           (cv2.HISTCMP_KL_DIV, 5, 'KL散度or相对熵')]
for method, method_id, method_name in methods:
    print('Method-' + str(method) + ': ' + str(round(cv2.compareHist(h1_n, h2_n, method), 4)), method_name)
```
输出
```
Method-0: 0.7898 相关系数
Method-1: 0.6871 卡方
Method-2: 2.6    十字
Method-3: 0.3405 巴氏系数
Method-3: 0.3405 同巴氏系数
Method-4: 1.5615 调整的卡方
Method-5: 8.5316 KL散度or相对熵
```

**Cross-bin**

2个直方图的bin索引和个数都可以不一样，当直方图有偏移时，也能识别出其相似性

#### Metric

[Earth Mover's Distance (EMD)](https://en.wikipedia.org/wiki/Earth_mover%27s_distance)

#### Library

OpenCV: [EMD](https://docs.opencv.org/3.0-beta/modules/imgproc/doc/histograms.html#emd)

#### Article

- [向量相似度度量（一）：EMD - 2014](https://blog.csdn.net/wangdonggg/article/details/32329879)
  
- [向量相似度度量（二）：EMD的MATLAB对照实现 - 2014](https://blog.csdn.net/wangdonggg/article/details/32691445)
  
- [向量相似度度量（三）：科普-为什么度量方式很重要 - 2014](https://blog.csdn.net/wangdonggg/article/details/35280735)


### 2.3.2 Simhash

[Detecting Near-Duplicates for Web Crawling - Google2007](http://www.wwwconference.org/www2007/papers/paper215.pdf)

#### Code

- <https://github.com/leonsim/simhash> (Python)

#### Article

- [海量数据相似度计算之simhash短文本查找 - 2013](http://www.lanceyan.com/tag/simhash)

- [文档去重算法：SimHash和MinHash - 2015](https://blog.csdn.net/heiyeshuwu/article/details/44117473)


### 2.3.3 Locality Sensitive Hashing

LSH：即局部敏感哈希，用于海量高维数据的近似最近邻快速查找

#### Article

- [局部敏感哈希(Locality-Sensitive Hashing, LSH)方法介绍 - 2017](https://www.cnblogs.com/wt869054461/p/8148940.html)


## 2.4 Information Theory

### Overview

#### Article

- 【Great】[Visual Information Theory - 2015](http://colah.github.io/posts/2015-09-Visual-Information/)

    可视化信息理论，如交叉熵等！

- [机器学习笔记十：各种熵总结 - 2017](https://blog.csdn.net/xierhacker/article/details/53463567)

#### KL Divergence

- [Kullback-Leibler Divergence Explained - 2017](https://www.countbayesie.com/blog/2017/5/9/kullback-leibler-divergence-explained)


## 2.5 SVD

#### Article

- [We Recommend a SVD - 2009](http://www.ams.org/samplings/feature-column/fcarc-svd)

    A geometric explanation of singular value decompositions and some applications of them.


