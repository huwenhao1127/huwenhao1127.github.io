---
  
layout:     post
title:      使用sklearn学习K-means
subtitle:   聚类
data:       2019-01-14
author:     HuWenhao(Hexo Hu)
header-img:                            #标题背景图
catalog:    true                       # 是否归档
tags:
    - 数据挖掘
    - 聚类
    - 机器学习

---


# 聚类分析：k-means 
<p style="text-indent:2em;line-height:1.5">
    聚类分析（cluster analysis）简称聚类（clustering），是一个把数据对象划分为子集的过程。每个子集是一个簇（cluster），我们的目标是使簇中的对象彼此相似，但与其它簇中的对象不相似。基本聚类方法可以分为如下几类：划分方法（partitioning method）、层次方法（hierarchical method）、基于密度的方法（density-based method）、基于网格的方法（grid-based method）。
</p>
<p style="text-indent:2em;line-height:1.5">
    聚类分析最基本、最简单的方法是划分。本文将介绍最著名并且常用的划分方法：k-means，以及如何通过sk-learn使用k-means聚类。
</p>


### 一、 K-means简介
<p style="text-indent:2em;line-height:1.5">
    K-Means将簇内所有点的均值作为簇的质心。算法流程如下：初始化k个质心，依次计算每个样本点到质心的欧式距离，将样本点分配到距离最小的簇；计算簇内所有点的均值作为新的质心；再依次计算每个样本点到新的质心的欧式距离，重新将样本点分配到距离最小的簇；迭代，不断更新簇的质心与簇中元素，直到收敛。
</p>

### 二、使用sklearn学习K-Means聚类
#### 1. K-Means类简介
<p style="text-indent:2em;line-height:1.5">
    scikit-learn中有两种K均值算法：传统K-Means和Mini batch K-Means。分别对应KMeans类和MiniBatchKMeans类。
</p>
<p style="text-indent:2em;line-height:1.5">
    KMeans类常用参数如下：
</p>
<p style="text-indent:2em;line-height:1.5">
    (1) n_clusters： cluster的数目，即聚类的类数。默认为8。
</p>
<p style="text-indent:2em;line-height:1.5">
    (2) max_iter：最大迭代次数。默认为300。
</p>
<p style="text-indent:2em;line-height:1.5">
    (3) n_init：使用n种不同的质心初始化。KMeans的结果会受初始化的影响，因此可以选择不同的质心运行多次，选择最好的结果。默认值为10。
</p>
<p style="text-indent:2em;line-height:1.5">
    (4) init：初始化质心。‘random’为随机初始化，‘k-means++’为一种优化过的质心选择方法，也可以自己选择质心。默认值为‘k-means++’。自己选择质心的方式是传入一个array，array的每一行代表一个质心，行数为簇数，列数为数据点的维数。
</p>
<p style="text-indent:2em;line-height:1.5">
    (5) algorithm：“auto”, “full” or “elkan”。默认值为”auto”，为密集（dense）数据选择elkan算法，为稀疏（sparse）数据选择full算法。full算法为传统的KMeans算法。elkan算法为elkan KMeans算法，与传统方法相比elkan KMeans在数据密集时，不必计算每个样本点到质心的距离，能够有效提高迭代速度。
</p>

#### 2.  K-Means类常用methods
<p style="text-indent:2em;line-height:1.5">
    (1)fit：给定数据计算K-Means聚类。
</p>
<p style="text-indent:2em;line-height:1.5">
    (2)predict：预测给定数据的类别。
</p>
<p style="text-indent:2em;line-height:1.5">
    (3)transform：计算给定数据到每个质心的距离。
</p>
<p style="text-indent:2em;line-height:1.5">
    
</p>
#### 3. 实战（数据挖掘概念与技术第三版 习题10.2）


```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 给定数据
X = np.array([[2,10],[2,5],[8,4],[5,8],
              [7,5],[6,4],[1,2],[4,9]])
# 指定初始化质心
initClusterCenter = np.array([[2, 10], [5, 8], [1, 2]])
plt.figure()
plt.scatter(X[:,0], X[:,1])
plt.show()
```


![png](output_3_0.png)


训练数据（迭代一轮）：


```python
kmeans = KMeans(n_clusters=3, n_init=1, init=initClusterCenter, max_iter=1).fit(X)
```

查看每个样本点所属的簇（共有0、1、2三个簇）：


```python
kmeans.labels_
```




    array([0, 2, 1, 1, 1, 1, 2, 0])



查看簇中心：


```python
kmeans.cluster_centers_
```




    array([[ 2. , 10. ],
           [ 6. ,  6. ],
           [ 1.5,  3.5]])



训练数据至收敛（注意给定质心的情况下，n_init只能为1）：


```python
kmeans = KMeans(n_clusters=3, n_init=1, init=initClusterCenter).fit(X)
```

查看每个点到质心的距离：


```python
kmeans.transform(X)
```




    array([[1.94365063, 7.55718937, 6.51920241],
           [4.33333333, 5.04424865, 1.58113883],
           [6.61647775, 1.05409255, 6.51920241],
           [1.66666667, 4.1766547 , 5.70087713],
           [5.20683312, 0.66666667, 5.70087713],
           [5.51764845, 1.05409255, 4.52769257],
           [7.49073502, 6.43773597, 1.58113883],
           [0.33333333, 5.54777233, 6.04152299]])



绘图：


```python
plt.figure()
plt.scatter(X[:,0], X[:,1], c=kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],marker='+')
plt.show()
```


![png](output_15_0.png)


预测某一点属于哪个簇，并计算到三个簇中心的距离：


```python
y = np.array([[2,2]])
kmeans.predict(y), kmeans.transform(y)
```




    (array([2]), array([[7.19567771, 5.51764845, 1.58113883]]))




```python

```

