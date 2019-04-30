---
layout: post
title: "从Python迭代器到PyTorch的DataLoader"
date: 2019-04-30
categories: Python
tags: Python
mathjax: true
author: Prince
---

* content
{:toc}

迭代器（Iterator）是设计模式中一个很重要的模式，其主要作用是通过Iterator类对容器进行迭代，不断返回容器中的元素，以达到遍历容器或其他功能。

大多数语言如C++的STL，Java等都内置了迭代器模式，Python也不例外，本篇博客总结一下Python中的迭代器与生成器的相关知识点，并以PyTorch的DataLoader为例，使读者对Pytorch的数据加载有更深的理解。




## 一个简单的例子：摆动列表迭代器

我们以一个简单的例子作为开头，要求实现这样一个容器`SwinList`，按照以下顺序访问列表中的元素：

- 第一次返回第一个元素
- 第二次返回倒数第一个元素
- 第三次返回第二个元素
- 第四次返回倒数第二个元素
- ...
- 直到列表全部访问完成

比如输入`[1, 2, 3, 4, 5, 6, 7, 8, 9]`，迭代顺序为：`[1, 9, 2, 8, 3, 7, 4, 6, 5]`

根据上面的需求我们首先创建一个容器类`SwinList`来存储摆动列表的数据：

```python
class SwinList(list):
    """
    使用list初始化摆动列表
    """
    def __init__(self, data=[1, 2, 3, 4, 5]):
        self.data = data
    
    """
    获取列表中的某个元素
    """
    def __getitem__(self, item):
        return self.data[item]
    
    """
    实现__iter__()函数，这样才能使用(for ... in ...)访问该列表
    """
    def __iter__(self):
        return SwinListIter(self)
    
    """
    返回list的长度
    """
    def __len__(self):
        return len(self.data)
```

需要注意的是，在上面的定义中`__len__`和`__getitem__`两个函数不是必须实现，只是为了保证`SwinList`功能的完整性，方便迭代器调用。我们注意到在`__iter__`函数中返回了一个`SwinListIter`对象，并把自己作为初始化参数。实际上，当我们执行`for elem in container`的时候，就调用的是他的`__iter__`函数，得到一个迭代器对象，使用这个对象对容器进行访问。

接下来重点来了，对于一个迭代器我们需要实现的最重要的函数是`__next__`(在Python2中是`next`)，迭代器的主要职责：

- 保存当前访问到容器的状态，即某种形式的index，这里我们记录当前访问到的数组的左界`self.left`和右界`self.right`
- 每次调用`__next__`的时候，返回容器的下一个元素并更新迭代器的状态，以便下次执行
- 如果本次迭代已经结束，需抛出`StopIteration`异常使迭代停止

```python
class SwinListIter(object):
    
    """
    使用容器初始化迭代器
    """
    def __init__(self, swin_list):
        self.container = swin_list
        self.return_left = True  # 判断当前返回左值还是右值的flag
        self.left = 0  # 当前访问到的左值index
        self.right = len(swin_list) - 1  # 当前访问到的右值的index
        
    def __iter__(self):
        return self
		
    """
    实现__next__函数，每次调用返回容器的下一个元素
    """
    def __next__(self):
        
        # 如果已经迭代结束，则抛出StopIteration异常
        if self.left > self.right:
            raise StopIteration()
				
        # 当前访问左值
        if self.return_left:
            self.return_left = not self.return_left
            ret = self.container[self.left]
            self.left += 1
            return ret
        # 当前访问右值
        else:
            self.return_left = not self.return_left
            ret = self.container[self.right]
            self.right -= 1
            return ret

    def __len__(self):
        return len(self.container)
```

有了以上的定义我们就可以使用Python自带的for语句访问`SwinList`了

```python
swinlist = SwinList([1, 2, 3, 4, 5, 6, 7, 8, 9])
for elem in swinlist:
    print(elem)
```

其实for以上的for语句等同于下面的while语句，首先获得需要迭代的容器的迭代器（`iter(_list)`），然后不断调用迭代器的`__next__`函数(`next(_it)`)，直到`__next__`函数抛出异常：

```python
it = iter(swinlist)
while True:
    try:
        print(next(it))
    except StopIteration:
        break
```

当然，由于迭代过程和容器存储是分开的，我们的`SwinListIter`还可以用于其他容器的迭代，只要这个容器与list的访问方式相当，并实现了`__len__`函数，比如直接迭代Python自带的list，这就是解耦带来的好处：

```python
ori_list = [1, 2, 3, 4, 5, 6]
it = SwinListIter(ori_list)

for elem in it:
    print(elem)
```

### 几个问题

**1. 容器本身实现`__next__`是否可以？**

理论上是可以的，但这样容器本身除了保存数据外，还需要保存迭代的状态，解耦不充分，并且无法实现多个迭代器同时迭代同一个容器，灵活性会差很多。另外，迭代器无法复用到其他容器上，代码可扩展性差很多

**2. 迭代器`SwinListIter`本身为什么也要实现`__iter__`函数**

一些时候，会把迭代器单独赋值然后使用for语句，比如

```python
it = iter(swinlist)
for elem in it:
    ...
```

这个时候需要`SwinListIter`本身也是可以返回迭代器的，当然只需要简单的返回self就可以了

**3. 可不可以实现一个永远迭代不完的迭代器**

可以，只需要在`__next__`中永远不抛出`StopIteration`，容器的迭代就不会结束，比如随机数生成迭代器等，但是要注意控制循环语句的循环次数，也可以通过直接调用`next(it)`的形式防止无限循环。

## 浅析生成器

说完了迭代器我们在说一下生成器(Generator)，Python中生成器是一种特殊的迭代器，但是不需要显示实现`__iter__`和`__next__`函数，也不需要显示抛出`StopIteration`异常，生成器的定义非常类似函数定义，唯一区别是：生成器中含有`yield`关键字，用于不断生成新的元素，我们还是以摆动列表为例简单讲一下生成器的使用：

```python
def swin_generator(swinlist):
    left = 0
    right = len(swinlist) - 1
    return_left = True
    
    while(left < right):
        if return_left:
            return_left = not return_left
            ret = swinlist[left]
            left += 1
            yield ret
        else:
            return_left = not return_left
            ret = swinlist[right]
            right -= 1
            yield ret

gen = swin_generator(swinlist)
for elem in gen:
    print(elem)
```

生成器的定义过程中使用`yield`产生下一个数据，每次调用`next(gen)`的时候程序会执行到第一个`yield`的地方然后"卡"在那里，直到下一次调用next的时候，恢复断点并执行到下一个`yield`所在的语句，当函数全部走完的时候，自动停止迭代。

## Pytorch中的DataLoader与\_DataLoaderIter

说完了迭代器和生成器，接下来可以看一下PyTorch中是如何使用迭代器实现数据加载的。使用过PyTorch的同学应该都知道PyTorch的DataLoader机制，一个典型的DataLoader使用如下：

```python
# 定义dataset
dataset = MyDataset(...)
dataloader = data.DataLoader(dataset, batch_size=4, ...)

# 使用迭代器迭代
for image, label in dataloader:
    image = image.to(device)
    label = label.to(device)
    ...
```

那么PyTorch是如何实现的呢，和上面一样，我们可以认为dataloader就是一个容器，（dataset的作用只是实现`__getitem__`函数方便使用`[]`调用，因此dataloader才是真正需要迭代的容器），而在dataloader的`__iter__`函数中，会返回一个`_DataLoaderIter`对象用于迭代dataloader，我们把Dataloader的源码中有关迭代的代码抽出来如下：

```python
class DataLoader(object):
    """
    使用dataset, batch_size等参数构造DataLoader
    """
    def __init__(self, dataset, batch_size, ...):
        self.dataset = dataset
        self.batch_size = batch_size
        pass
    def __iter__(self):
        return _DataLoaderIter(self)
        
class _DataLoaderIter(object):
    """
    存储容器loader的引用，也获取了dataloader的各种参数，如collate_fn，batch_sampler等，用于不同策略的迭代
    """
    def __init__(self, loader):
    	  self.dataset = loader.dataset
        self.collate_fn = loader.collate_fn
        self.sample_iter = iter(self.batch_sampler) # 使用一个iterator来记录当前迭代的状态
        pass
    
    """
    返回self，和上面的例子一样
    """
    def __iter__(self):
        return self
    
    """
    使用batch_size等(代码经过简化)
    """  
    def __next__(self):
        
        # 获取下一个batch包含dataset中哪些index，next可能会抛出StopIteration异常
        indices = next(self.sample_iter)  
        
        # 这里调用了dataset的__getitem__函数真正获取了数据并返回
        batch = self.collate_fn([self.dataset[i] for i in indices]) 
        
        return batch
```

### 值得注意的几个问题

**1. 迭代器对于容器是否一定是只读（readonly）的？**

如果没有特殊需求，一般迭代器对于容器是只读的，个人认为这样对于多个迭代器同时访问同一个容器是有好处的，如果在迭代过程中修改了容器的数据，其他迭代器可能就会访问到错误的数据，比如在做augmentation的时候，``__getitem__``中一般不对dataset进行修改

**2. PyTorch为什么要dataset和dataloader分开**

更加方便的解耦，dataset的存在旨在给所有的数据集一个统一的结构(使用中括号取数据)，对于每一个数据集，读取方法几乎都是不同的。而dataloader的行为大都趋于一致，按照batch取数据，最多是在如何对数据采样等方法上有一些定制。因此将二者分开，用户只需要重写经常变化的部分(dataset)，对于dataloader保持稳定性，毕竟不常扩展。

## 总结

Python中迭代器与生成器是非常重要的概念，其灵活性使开发者能够定制出自己需要的迭代方式，在实现上，私以为比较重要的问题是**容器与迭代器的解耦**与**对容器的只读**。
