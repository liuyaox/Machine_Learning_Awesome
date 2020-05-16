# 15. Model Selection and Evaluation

## 15.1 Overview

#### Article

- [万字长文总结机器学习的模型评估与调参 - 2019](https://mp.weixin.qq.com/s?__biz=MzIwOTc2MTUyMg==&mid=2247492923&idx=2&sn=15fd5960ca20f1bd81916e625f929448)

## 15.2 Distributed Training

#### Library

- [DISTRIBUTED COMMUNICATION PACKAGE - TORCH.DISTRIBUTED](https://pytorch.org/docs/stable/distributed.html)

#### Article

- [Distributed data parallel training in Pytorch - 2019](https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html)

    **Chineses**: [PyTorch 分布式训练简明教程](https://mp.weixin.qq.com/s/zv56wfusfPkBNYPiA2a51g)

- [分布式训练的方案和效率对比 - 2020](https://mp.weixin.qq.com/s/nTBuYuW7h9wZYuo3w1xGmQ)


## 15.4 Model Ensembling

### 15.4.1 Overview

#### Paper

- [Snapshot Ensembles: Train 1, get M for free - 2017](https://arxiv.org/abs/1704.00109)

#### Article

- [KAGGLE ENSEMBLING GUIDE - 2015](https://mlwave.com/kaggle-ensembling-guide/)

- [Introduction to Python Ensembles - 2018](https://www.dataquest.io/blog/introduction-to-ensembles/)

#### Library

- <https://github.com/yzhao062/combo>

    A Python Toolbox for Machine Learning Model Combination

    **Doc**: <https://pycombo.readthedocs.io/en/latest/>

    **Article**: [大部分机器学习算法具有随机性，只需多次实验求平均值即可吗？](https://www.zhihu.com/question/328157418/answer/746533382)


### 15.4.2 Boosting

#### Code

- <https://github.com/brightmart/text_classification/blob/master/a00_boosting/a08_boosting.py> (Tensorflow)


### 15.4.3 Bagging


### 15.4.4 Stacking 

对训练好的基学习器的应用结果进行**非线性融合**(输入并训练一个新的学习器)

偷懒的话，可直接使用 **Out-of-fold(OOF)** 做 Stacking

**解读方式1：先循环KFold，然后再循环各基模型**

![](https://raw.githubusercontent.com/liuyaox/ImageHosting/master/for_markdown/Stacking.png)

定义训练集P={(x1, y1), (x2, y2), ..., (xm, ym)}，测试集Q，基模型E1, E2, ..., ET，主模型E，则伪代码如下：

```
for P1, P2 in KFold(P) do       # 训练集拆分：训练集分成P1(k-1 folds)和P2(1 fold)，分别用于训练和验证，要遍历所有KFold
    for t = 1, 2, ..., T do
        Et.train(P1)            # 基模型训练：使用P1训练每个基模型
        prt = Et.apply(P2)      # 基模型应用：基模型应用于P2生成prt
        qrt = Et.apply(Q)       # 基模型应用：基模型应用于Q 生成qrt
    
    pr = [pr1, pr2, ..., prT]   # 收集每个基模型应用结果prt，用于主模型训练，只是1/K份的训练数据
    qr = [qr1, qr2, ..., qrT]   # 收集每个基模型应用结果qrt，用于主模型测试，是全份的测试数据

PR = concate(pr)                # 训练数据纵向堆叠，共同组成总训练数据，共T列，表示T个新特征
QR = average(qr)                # 测试数据纵向求均值，当作最终测试数据，列同上

E.train(PR)                     # 主模型训练
score = E.evaluate(QR)          # 主模型测试

Y(x) = E.apply(E1.apply(x), E2.apply(x), ..., ET.apply(x))  # 全流程应用
```

注意：各Fold的结果是**纵向堆叠或纵向均值**

**解读方式2：先循环各基模型，然后再循环KFold**

每个基模型内先做交叉验证，交叉验证的valid结果纵向拼接为1列，test结果求均值，共同组成该基模型的结果，**表示1列特征**，最后各基模型的结果**横向拼接成T列**

#### Article

- [模型融合之stacking&blending](https://zhuanlan.zhihu.com/p/42229791)

    **Code**: <https://github.com/InsaneLife/MyPicture/blob/master/ensemble_stacking.py>

- [图解Blending&Stacking](https://blog.csdn.net/sinat_35821976/article/details/83622594)

- [为什么做stacking ensemble的时候需要固定k-fold？](https://www.zhihu.com/question/61467937/answer/188191424)

- [A Kaggler's Guide to Model Stacking in Practice](http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/)


### 15.4.5 Blending

理念与Stacking比较类似，模型分为两层，不过比Stacking简单一些，不需要通过KFold这种CV策略来获得主模型的特征，而是建立一个Holdout集，直接使用不相交的数据集用于两层的训练。以两层Blending为例，详细来说为：

Step1: 训练集划分为两部分P1和P2，测试集为Q

Step2: 用P1训练每个基模型

Step3: 基模型应用于P2的结果，训练主模型

Step4: 基模型应用于Q的结果，测试主模型，模型应用时与测试流程一样。

#### Article

- [Blending 和 Stacking - 2018](https://blog.csdn.net/u010412858/article/details/80785429)
