---
layout: post
title: "Python装饰器详解"
date: 2019-04-16
categories: Python
tags: Python
mathjax: false
author: Prince Wang
---

* content
{:toc}

在Python中，装饰器是一种极为优雅的给现有函数增加功能的方式，本篇博客总结一些装饰器的常见使用方法与复杂装饰器使用方法，文末给出一些例子帮助读者理解。

* 参考博客：https://foofish.net/python-decorator.html



## 基本装饰器

Python中，装饰器是用于给现有函数增加功能的方法，首先我们假设装饰器需要装饰的函数是这样：

```python
def func(a, b):
    return a + b
```

现在已有函数func，希望通过decorator增加func的功能，比如在func执行前与执行后都print一段文字，我们可以将装饰器的运行过程理解成以下代码：

```python
# 原来函数这么执行
result = func(a, b)
```

```python
# 有了装饰器以后这么执行
func = decorator(func)
result = func(a, b)
```

那么如何写一个装饰器呢，一个最简单的装饰器可以定义如下：

```python
def decorator(func):
    def wrapper(a, b):
        print('before')
        result = func(a, b)
        print('after')
        return result
    return wrapper
```

这样我们就可以通过func = decorator(func)来调用装饰器来装饰func函数了
**装饰器的理解有以下几个要点(很重要！)**：

- 装饰器是一个本质上是一个函数（或者带有`__call__`成员函数的类也可以）
- 装饰器的输入是一个**函数**
- 装饰器的返回值也是一个**函数**（wrapper）
- 装饰器的输入函数与他返回的wrapper函数的形参**必须一致**，这样才能保证装饰以后可以没有错误的调用，比如上面的wrapper的输入参数也是a，b


## 更加复杂的装饰器

有了以上的要点，我们不难写出装饰器来，以下是几个注意的点：

- 装饰器既然独立装饰各种函数，那么之前的decoroator里面wrapper的函数的形参是固定的（a,b），因此只能装饰入参为两个的函数，其他函数就无能为力了，为了解决这个问题，我们可以利用`*args`和`**kwargs`来代替a,b，使装饰器可以装饰任意入参的函数：

```python
def decorator(func):
    def wrapper(*args, **kwargs):
        print('before')
        result = func(*args, **kwargs)
        print('after')
        return result
    return wrapper
```

- 装饰器的语法糖，python中使用@符号来表示装饰器以代替`func = decorator(func)`这代码，如刚才这个装饰器我们可以写成：

```python
@decorator
def func(a, b):
    return a + b
```

- 装饰器比较灵活，其实我们可以这么理解，`func = XXX(func)`，这就是一个装饰器，这里的XXX可以是任何“**输入参数是一个函数，返回是和func入参一样的函数**”的东西，因此装饰器的实现相当灵活，我们在读带有装饰器的代码的时候，可以把`@`后面的所有东西拿出来(不管他多复杂)，然后带入`func = XXX(func)`中，这样有助于我们理解复杂装饰器的逻辑。
- 装饰器可以带参数，这就需要将上面的装饰器再包裹一层，让wrapper在调用func的时候用到装饰器的参数。
- 装饰器也可以是一个类，和带参数的装饰器很像，不过初始化变成了一个类，而调用变成实现`__call__`函数。
- 装饰器还可以是一个类的成员函数，总之是那句话，只要是任何**"输入是一个函数对象，输出是一个与输入函数入参一样的函数”**的东西，都可以成为一个装饰器。


## 一些简单的例子

- 将所有的被装饰函数的输入和输出打印log出来的装饰器：

```python
def decorator(fn):
    def wrapper_func(*args, **kwargs):
        print('input params:', args, kwargs)
        res = fn(*args, **kwargs)
        print('output:', res)
        return res
    return wrapper_func

@decorator
def sum(a, b):
    return a + b

@decorator
def div(a, b):
    return a / b

sum(1, 2)
div(1, 2)

>>> input params: (1, 2) {}
>>> output: 3
>>> input params: (1, 2) {}
>>> output: 0.5
```

- 带参数的装饰器，用于输出不同的log，warning，error等等（本质是把上面的装饰器用一个函数再包裹一下，加一个参数）：

```python
def logging(mode='warning'):
    def decorator(fn):
        def wrapper_func(*args, **kwargs):
            print('{}: input params:'.format(mode), args, kwargs)
            res = fn(*args, **kwargs)
            print('{}: output:'.format(mode), res)
            return res
        return wrapper_func
    return decorator

@logging(mode='warning')
def sum(a, b):
    return a + b

@logging(mode='warning')
def div(a, b):
    return a / b
 
sum(1, 2)
div(1, 2)

>>> warning: input params: (1, 2) {}
>>> warning: output: 3
>>> error: input params: (1, 2) {}
>>> error: output: 0.5
```
- 用类做装饰器

```python
class Logger(object):
    
    def __init__(self, mode='warning'):
        self.mode = mode
        
    def __call__(self, fn):
        def wrapper_func(*args, **kwargs):
            print('{}: input params:'.format(self.mode), args, kwargs)
            res = fn(*args, **kwargs)
            print('{}: output:'.format(self.mode), res)
            return res
        return wrapper_func

@Logger('error')
def sum(a, b):
    return a + b

@Logger('warning')
def div(a, b):
    return a / b
  
sum(1, 2)
div(1, 2)
>>> warning: input params: (1, 2) {}
>>> warning: output: 3
>>> error: input params: (1, 2) {}
>>> error: output: 0.5
```

- 多重装饰器

装饰器可以不止一个，我们可以通过多个`@`堆叠的方式来多个嵌套多个装饰器，功能类似`fucn = d2(d1(func))`，具体语法如下：

```python
# 离函数越近的装饰器越先调用
@Logger('decorator2')
@Logger('decorator1')
def sum(a, b):
    return a + b

>>> decorator2: input params: (1, 2) {}
>>> decorator1: input params: (1, 2) {}
>>> decorator1: output: 3
>>> decorator2: output: 3
```

