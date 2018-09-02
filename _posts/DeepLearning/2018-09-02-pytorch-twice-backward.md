---
layout: post
title: "PyTorch RuntimeError: Trying to backward through the graph a second time"
date: 2018-09-02
categories: DeepLearning
tags: DeepLearning PyTorch
mathjax: true
author: Prince
comment: true
---

* content
{:toc}

最近在使用Pytorch的时候，在backward函数中报backwrd两次的错误：
```python
RuntimeError: Trying to backward through the graph a second time, but the buffers have already been freed. Specify retain_graph=True when calling backward the first time.
```
但实际上我只使用了一次backward，在经过一些debug以后总结一下错误导致的原因。




这种错误并不一定是由于调用了两次backward函数(当然如果真的调用了两次backward肯定也会报这个错，这里说的是不那么显而易见的情况)，还有另外一种情况可能导致了这个错误——在计算的中间变量被临时保存下来，在下一个batch的forward中使用了，也会造成这个错误。

**原因分析**：为了节省内存，在执行backward过程中，pytorch会把不用的中间计算结果（用来执行反向传播的）缓存删掉，所以其实已经反传不回去了，这个时候如果使用到已经被删除缓存的变量的时候，将会报上面这个错误

接下来举一个小例子来说明这个问题，首先写一个非常小的例子：输入`[1, 2, 3]`，结果总是预测数1:

```python
import torch
import torch.nn as nn
from torch.autograd import Variable

# 定义输入输出
a = torch.Tensor([1, 2, 3])
t = torch.Tensor([1])

# 定义网络结构
fc1 = nn.Linear(3, 5)
fc2 = nn.Linear(5, 1)
loss_func = nn.MSELoss()

# 定义优化器
sgd = torch.optim.SGD(params=list(fc1.parameters()) + list(fc2.parameters()), lr=0.01)

# 执行10次循环
for i in range(10):
    input = Variable(a)
    target = Variable(t)
    h = fc1(input)
    out = fc2(h) 
    loss = loss_func(out, target)
    sgd.zero_grad()
    loss.backward()
    sgd.step()
    print(loss.data[0])
```
这个例子跑下来是没问题的，我们看到输出loss逐渐减小：

```
2.724853277206421
0.9550517797470093
0.3340338468551636
0.11237471550703049
0.03643478825688362
0.011504091322422028
0.003571596462279558
0.0010977740166708827
0.00033545514452271163
0.00010217395174549893
```

接下来我们做一件事情，把第一个batch的中间结果h保存下来，用于下一次的forward计算，主循环改成这样：

```python
# 记录一个临时值
temp = []

# 执行10次循环
for i in range(10):
    input = Variable(a)
    target = Variable(t)
    h = fc1(input)
    out = fc2(h) 
    temp.append(h) # 把中间结果存一下用于下次forward
    if i != 0:
        out = out + temp[0].mean() # 第二次开始就使用temp中的变量计算
    loss = loss_func(out, target)
    sgd.zero_grad()
    loss.backward()
    sgd.step()
    print(loss.data[0])
```

这个时候出现错误：

```
0.01125330664217472
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
<ipython-input-5-943bd347bdab> in <module>()
     11     loss = loss_func(out, target)
     12     sgd.zero_grad()
---> 13     loss.backward()
     14     sgd.step()
     15     print(loss.data[0])

/usr/share/Anaconda3/lib/python3.6/site-packages/torch/autograd/variable.py in backward(self, gradient, retain_graph, create_graph, retain_variables)
    165                 Variable.
    166         """
--> 167         torch.autograd.backward(self, gradient, retain_graph, create_graph, retain_variables)
    168 
    169     def register_hook(self, hook):

/usr/share/Anaconda3/lib/python3.6/site-packages/torch/autograd/__init__.py in backward(variables, grad_variables, retain_graph, create_graph, retain_variables)
     97 
     98     Variable._execution_engine.run_backward(
---> 99         variables, grad_variables, retain_graph)
    100 
    101 

RuntimeError: Trying to backward through the graph a second time, but the buffers have already been freed. Specify retain_graph=True when calling backward the first time.

```
其实大部分时候，上一次的循环都不需要回传，解决办法是在保存下来的时候增加一个detach()，比如刚才这个代码，修改为`temp.append(h.detach())`，这样就不会回传梯度了，另外一个办法是把variable的data拿出来，再重新包装为Variable，也可以避免这种错误（注：PyTorch 0.4.0以后取消了Variable和Tensor的区别，故重新包装为Variable的方法在0.4.0以后的版本不可用，尽量使用`detach()`）

如果真的需要对上一轮迭代的梯度回传的话，也可以使用`backward(retain_graph=True)`来保存中间计算结果，但是显存的使用量将会累计，故慎用这个flag。