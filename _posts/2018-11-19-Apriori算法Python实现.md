---

layout:     post
title:      频繁项挖掘（一）：Apriori算法
subtitle:   频繁项挖掘
data:       2018-11-19
author:     HuWenhao(Hex)
header-img: 		               #标题背景图
catalog:    true                       # 是否归档
tags:
    - 数据挖掘
    - 频繁项挖掘
    - 算法
    
---

## 相关概念与公式
- 支持度计数  
suppor_count(A)：为数据集中有项集A的事务的个数，不是A出现的次数。
- 支持度  
关联规则A=>B的支持度为：support(A=>B) = P(A U B) = support_count(A U B)/|D|  
|D|为数据集中的事务总数
- 置信度  
关联规则A=>B的置信度为：confidence(A=>B) = P(B|A) = support\_count(A U B)/support\_count(A)

## 算法核心思想
**先验性质**：频繁项集的所有非空子集一定是频繁的。即如果一个k-1项集不是频繁的，那么它的超集一定不是频繁的，这也是反单调性的一种。  
**反单调性（antimonotone）**：如果一个集合不能通过测试，那么它的所有超级都不能通过测试，称它为反单调的。
## 算法步骤
1. 产生候选1项集C1，并给每一个项集计数。
2. 通过支持度阈值筛选，从C1中找到频繁1项集L1。  
3. 迭代，由L(k-1)产生候选k项集Ck，再由CK产生Lk，直到无法产生候选项集。 
4. 由频繁项集产生关联规则。若两个频繁项集是强关联的，它们的交集也应该是频繁项集。

**问题：** 如何由L(k-1)项集产生Ck？  
1. 连接步。由L(k-1)项集与自身连接产生候选k项集Ck。首先将L(k-1)中每个频繁项集中的项都按字典序排序，若两个频繁项的前k-2个项相同（除了最后一项不同，其余项都相同），则这两项可以连接。例如：三个4项集 abcd abce abfg，abcd abce可连接生成abcde，abfg不能与另外两项连接。
2. 剪枝步。依次判断Ck中k项集的所有子集是否在L(k-1)中，若不在，则可从Ck中删除。由先验性质可知，频繁k项集所有子集必然是频繁的，而L(k-1)中都是频繁k-1项集。因此，若CK中一个k项集有一个子集不在L(k-1)中，该k项集一定不是频繁的。

## 算法实现
- 计数函数
为候选k项集中每个k项集计数，每一次计数需要扫描一次数据集，时间复杂度高。

```
def CkCount(Ck, dataSet):
    for kItems in Ck.keys():
        for thing in dataSet:
            if set(thing) | set(kItems) == set(thing):
                Ck[kItems] += 1
```

- 连接函数  
输入为频繁k-1项集的集合L(k-1)，先将L(k-1)中每一个频繁项集sort，再做连接，产生候选k项集。

```
def selfJoining(Lk):
    Ck = {}
    for i in range(len(Lk)):
        Lk[i] = sorted(Lk[i])
    for i in range(len(Lk)):
        for j in range(i+1, len(Lk)):
            if Lk[i][:-1] == Lk[j][:-1]:
                set1 = set(Lk[i])
                set2 = set(Lk[j])
                Fk_1 = frozenset(set1 | set2)
                Ck[Fk_1] = 0
    return Ck
```

- 剪枝函数
输入候选k项集，输出为剪枝后的候选k项集。剪枝原则：若k-1项子集不在L(k-1)中，则删除。

```
def pruning(Ck, Lk):
    for kItems in Ck.keys():
        for i in range(len(kItems)):
            temp = list(kItems)
            temp.pop(i)
            if temp not in Lk:
                del Ck[kItems]
```

- 发现强关联规则函数  
思路是对所有频繁项集逐个求并集，若并集是频繁项集，则继续求confidence，conf符合confMin则为强关联规则。

```
def calBigRules(frequentItems, confMin):
    bigRules = []
    confidences = []
    keys = list(frequentItems.keys())
    for i in range(len(frequentItems)):
        for j in range(len(frequentItems)):
            if i is not j:
                temp = frozenset(set(keys[i]) | set(keys[j]))
                if (temp in frequentItems) and (not (set(keys[i]) & set(keys[j]))):
                    conf = frequentItems[temp]/frequentItems[keys[i]]
                    if conf >= confMin:
                        bigRules.append([keys[i], keys[j]])
                        confidences.append(conf)
    return bigRules, confidences
```

- 主函数  
输入为数据集、最小支持度和最小置信度，输出为存放所有频繁项及其支持度计数的字典、存放强关联规则的列表和存放强关联规则置信度的列表。首先根据数据集产生候选1项集并计数，再由C1生成L1。得到L1后开始迭代，迭代过程为:L(k-1) => C(k); C(k) => L(k)。迭代结束条件为：产生的C(k)为空。

```
def apriori(dataSet, supMin, confMin):
    frequentItem = {}
    C1 = {}
    supCountMin = supMin * len(dataSet)
    for thing in dataSet:
        temp = set(thing)
        for item in temp:
            if item not in C1:
                C1[item] = 1
            else:
                C1[item] += 1
    L1 = []
    for item in C1.keys():
        if C1[item] >= supCountMin:
            L1.append([item])
            frequentItem[frozenset({item})] = C1[item]
    Lk_1 = L1
    while True:
        Ck = selfJoining(Lk_1)
        if len(Ck) == 0:
            break
        CkCount(Ck, dataSet)
        Lk_1.clear()
        for items in Ck.keys():
            if Ck[items] >= supCountMin:
                Lk_1.append(list(items))
                frequentItem.update({items: Ck[items]})
    bigRule, confidence = calBigRules(frequentItem, confMin)
    return frequentItem, bigRule, confidence
```

## 总结
1. 当数据集很大时仍然需要产生大量的候选项集。
2. 由于每次对C(k)计数都需要扫描一次数据集，因此算法的时间复杂度高。

## 参考
- 数据挖掘概念与技术 Jiawei Han, Micheline Kamber， Jian Pei

## 程序源码
[Apriori算法 Python简单实现](https://github.com/huwenhao1127/DataMining/blob/master/Frequent%20Itemsets%20Mining/Apriori.py)

