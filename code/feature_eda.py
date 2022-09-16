
import json
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.max_rows', 100)     # 显示所有行

from scipy import stats
from sklearn.feature_selection import SelectKBest, chi2
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('xxx.csv')


# 1. label处理
df.label.fillna(-999, inplace=True)
df.label.value_counts(dropna=False)


# 2. 无用列处理
df.columns
cols2del = ['comment_label', 'comment_god', 'comment_label_1', 'comment_label_2', 'comment_label_3', 'comment_label_4',
            'comment_id2', 'user_id', 'author_user_id', 'author_device_id', 'date', 'cert_name', 'item_create_day', 
            'group_comment_counter', 'group_comment_feature', 'user_comment_feature_cmt', 'user_comment_feature_verify',
           'group_comment_counter0', 'group_comment_feature0', 'user_comment_feature_30d_cmt', 'user_comment_feature_verify0']
for col in cols2del:
    del df[col]


# 3. 缺失值处理
pd.set_option('display.max_rows', None)
df.isnull().sum().sort_values()
cols2del = ['xx1', 'xx2', 'xx3']
df = df.drop(cols2del, axis=1)



# 4. 特征相关性分析
df.columns
cols_cate = ['xx1', 'xx2']
cols_num = ['xx3', 'xx4']

# 4.1 数值变量 VS 二分类变量
# 4.1.1 点二列相关系数
# 点二列相关系数是Pearson相关系数的一种特殊形式
corr = {}
for col in cols_num:
    x, y = df['label'], df[col]
    x = x[~y.isnull() & x.isin([0, 1])]
    y = y[~y.isnull() & x.isin([0, 1])]
    corr[col] = stats.pointbiserialr(x, y).correlation
corr = pd.Series(corr).sort_values()
corr.plot(kind='barh', figsize=(20, 10))

thresh = 0.1
corr = abs(corr)
cols2drop = corr[corr < thresh].index.to_list()
print(f'删除的列:\n{cols2drop}')
remaining = corr[corr >= thresh].index.to_list()
print(f'保留的列:\n{remaining}')


df = df.drop(cols2drop, axis=1)
cols_num = [x for x in cols_num if x not in cols2drop]


# 4.1.2 箱线图
sns.boxplot(y='xx3', x='label', data=df)


# 4.1.3 T检验-省略


# 4.2 分类变量 VS 二分类变量
# 4.2.1 卡方检验
select_k_best = SelectKBest(chi2, k=6)
select_k_best.fit_transform(abs(df[cols_cate].fillna(999)), df['label'])

p_scores = dict(zip(cols_cate, zip(select_k_best.scores_, select_k_best.pvalues_)))
corr = pd.Series(p_scores).sort_values(key=lambda x: x.str.get(0), ascending=False)
corr

cols_k_best = [cols_cate[idx] for idx in select_k_best.get_support(True)]
cols_k_best


# 4.2.2 卡方检验2
corr = {}
for col in cols_cate:
    crosstab = pd.crosstab(df['label'], df[col])
    corr[col] = stats.chi2_contingency(crosstab)[:2]            # 前2个是卡方值、p值
corr = pd.Series(corr).sort_values(key=lambda x: x.str.get(0), ascending=False)
corr

cols2drop = [x for x in cols_cate if x not in cols_k_best and x not in corr.index.to_list()[:6]]

df = df.drop(cols2drop, axis=1)
cols_cate = [x for x in cols_cate if x not in cols2drop]


# 4.3 数值变量 VS 数值变量
# 4.3.1 相关系数
corr = df[cols_num + ['label']].corr().loc['label'][cols_num].sort_values()
corr.plot(kind='barh', figsize=(20, 15))

thresh = 0.1
corr = abs(corr)
cols2drop5 = corr[corr < thresh].index.to_list()
print(f'删除的列:\n{cols2drop5}')
remaining5 = corr[corr >= thresh].index.to_list()
print(f'保留的列:\n{remaining5}')


# 5. 低方差特征
df[cols_num].var().astype('str').sort_values()

cols2del = ['report_cnt']
df = df.drop(cols2del, axis=1)
cols_num = [x for x in cols_num if x not in cols2del]


# 6. 多重共线性
# 6.1 数值变量
# 6.1.1 基于相关系数
sns.set(rc={'figure.figsize':(16,10)})
corr = df[cols_num + ['label']].corr()
sns.heatmap(corr, annot=True, linewidths=.5, center=0, cbar=False, cmap="PiYG")
plt.show()

cols2drop = ['xx5', 'xx6']
df = df.drop(cols2drop, axis=1)
cols_num = [x for x in cols_num if x not in cols2drop]


# 6.1.2 基于方差膨胀因子（VIF）
# VIF: 1表示无相关性，1-5表示中等相关性，>5表示高相关性
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = {}
for i, col in enumerate(cols_num):
    vif[col] = variance_inflation_factor(df[cols_num].fillna(0).values, i)

vif = pd.Series(vif).sort_values(ascending=False)
vif
vif[vif < 10].index.to_list()
