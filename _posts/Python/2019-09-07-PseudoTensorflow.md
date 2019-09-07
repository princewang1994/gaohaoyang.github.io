---
layout: post
title: "实现一个简易的计算图功能"
date: 2019-09-07
categories: Python
tags: Python DeepLearning
mathjax: true
author: Prince
---

* content
{:toc}

众所周知，Tensorflow之类的静态图DL框架是基于计算图的，那么他们是如何实现先生成图然后再喂数据的呢，本篇博客将实现一个简单的计算图，可以实现类似TensorFlow中图的定义以及生成Session并把实际的数据喂进去的功能。

![](http://princepicbed.oss-cn-beijing.aliyuncs.com/blog_20190907174512.png)




### 要实现一个什么样的功能

为了让读者能够更好理解最终要做什么，我们先把测试代码写好，然后再去看怎么实现这个功能，测试代码如下：

```python
# 主函数调用
if __name__ == '__main__':

    Graph.clear()

    # 初始化placeholder
    a = PlaceHolder()
    b = PlaceHolder()

    # 计算图构建
    c = a + b
    d = b + a
    e = c * d + a + b
    p = e ** 2

    # 计算图喂数据
    sess = Session()
    print(sess.run([c, d, e], feed_dict={a: 8, b: 2}))
    print(sess.run([p], feed_dict={a: 10, b: 20}))
```

哈哈，是不是很类似Tensorflow的操作，接下来我们一步一步讲如何实现一个"盗版"Tensorflow的功能：

### 1. Tensor定义

TF中所有的计算节点我们都把他们认为是一个Tensor，所谓Tensor是一个虚拟的东西，在构建计算图的时候并不知道他的具体值是多少，只有在喂数据的时候才知道他的具体值是多少，所以我们先定义计算图节点Tensor：

```python
# Tensor抽象类，无法实例化
class Tensor(object):
    
    def __add__(self, rhs):
        return AddTensor(self, rhs) 
    
    def __mul__(self, rhs):
        return MulTensor(self, rhs)

    def __pow__(self, pow):
        return PowTensor(self, pow)
    
    def forward(self, feed_dict):
        raise NotImplementedError()
```

为了实现测试代码中的`c = a + b`这个功能，我们在Tensor中重载了几个运算符，分别是加号，乘号和乘方，读者可以看到，里面只是返回了另外一个叫做AddTensor的东西，另外Tensor还有一个公共的虚函数叫做forward，熟悉深度学习的同学应该很清楚这是用来计算前向的操作。不同的操作会用不同的方法来返回，比如加法就把a和b加起来，接下来我们来定义这几个Tensor的子类：

```python
# PlaceHolder类，forward直接返回其值
class PlaceHolder(Tensor):
    
    def __init__(self):
        Graph.graph[self] = None
        
    def forward(self, feed_dict):
        return feed_dict[self]

# 加法Tensor，初始化时保存他的结果是从哪两个值算出来的
class AddTensor(Tensor):
    
    def __init__(self, a, b):
        Graph.graph[self] = (a, b)
    
    def forward(self, feed_dict):
        a, b = Graph.graph[self]
        return a.forward(feed_dict) + b.forward(feed_dict)
    
# 乘法Tensor，初始化时保存他的结果是从哪两个值算出来的
class MulTensor(Tensor):
    
    def __init__(self, a, b):
        Graph.graph[self] = (a, b)
    
    def forward(self, feed_dict):
        a, b = Graph.graph[self]
        return a.forward(feed_dict) * b.forward(feed_dict)

# 一元运算符乘方
class PowTensor(Tensor):

    def __init__(self, a, pow=2):
        self.pow = pow
        Graph.graph[self] = a

    def forward(self, feed_dict):
        a = Graph.graph[self]
        return a.forward(feed_dict) ** self.pow
```

首先，我们把PlaceHolder也认为是一种Tensor，毕竟人家也是在计算图里面的基本节点，在forward的时候可以简单的返回其feed_dict中的值就好了，注意到这几个函数中都出现了Graph这个类别，他是什么东西呢，其实就是一个全局的计算图保存，举个例子，在测试用例中，`c = a + b`，会生成一个新的tensor——c，最终计算的时候我们会把a和b的具体值都算出来，然后再加起来返回，因此我们需要一个全局的计算图来保存节点与节点之间的联系，即c是由那两个tensor计算而来的，注意，这里我们使用了全局字典的方式，不一定是最好的方法，也可以直接将前继作为成员变量在初始化中保存。

接下来我们定义计算图Graph，其实很简单，就一个字典，该字典的格式如下：

```python
{
    tensor_c: (tensor_a, tensor_b),
    tensor_d: (tensor_b, tensor_a)
}
```

记录每个tensor的前继是谁，如果这个操作没有前继(比如PlaceHolder)，那么就是None

```python
class Graph(object):
    
    graph = {}
    
    def __init__(self):
        pass
    
    @staticmethod
    def clear():
        Graph.graph = {}
```

最后定义会话Session，用feed_dict来喂实际的数据，另外提供几个额外的函数

```python
class Session(object):
    
    def __init__(self):
        pass
    
    def run(self, var_list, feed_dict):
        if isinstance(var_list, Tensor):
            return var_list.forward(feed_dict)
        return [var.forward(feed_dict) for var in var_list]

# tensorflow中也提供这样的函数，不使用操作符
def tensor_sum(a, b):
    return a + b

def tensor_mul(a, b):
    return a * b
```

### 还有什么

通过以上的代码，我们已经可以简单的实现一个Tensorflow的前向传播的功能啦，但是这只是一个非常非常简易的例子，能够时间最基本的计算图功能，如果读者有兴趣，可以继续往下想有哪些地方是欠缺的，我这里给出几点：

1. 该例子中所有数据都是单元素变量，而实际框架中一般是矩阵，比如AddTensor其实是两个矩阵相加，在Tensor相加的时候就能知道两个矩阵的形状是不是正确，这也是静态图和动态图相比的好处。
2. 没有反向传播功能（实际上是比较懒）
3. 大量重复计算，细心的读者会发现，在计算`sess.run([c, d, e], feed_dict={a: 8, b: 2})`的时候，计算e的时候需要用到c的值，而实际上我们通过递归的方法是重复计算了一遍c的值的，性能会有很大的损失，比较好的方法是用记忆化搜索的方式，把所有Tensor的值都存下来，如果再次用到了，不需要重复计算
