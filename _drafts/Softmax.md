## Softmax求导

[TOC]

### 前言
* Softmax函数是Sigmoid函数的推广形式，Sigmoid常常被用于二分类问题的顶层，作为类标为1的概率。当问题被推广为多分类问题时，Sigmoid函数就不能使用了，此时往往使用Sigmoid函数的推广形式，即Softmax函数。
对于C个类别的分类问题，可以把输出结果定义为一个C维的向量$z^C = [z_0, z_1 ... z_C]​$，其中，如$C=3​$可以认为输出结果是一个3维向量$[2, 3, 4]​$，但是我们需要的是概率,所以要使用Softmax把向量转换为概率，Softmax函数的定义如下:

$$
    a_k = \frac{e^{z_k}}{\sum_{k}{e^{z_k}}}
$$

如上例，把$z$转换为$[\frac{e^{z_0}}{e^{z_0}+e^{z_1}+e^{z_2}}, \frac{e^{z_1}}{e^{z_0}+e^{z_1}+e^{z_2}}, \frac{e^{z_2}}{e^{z_2}+e^{z_1}+e^{z_2}}]$,也就是$[\frac{e^{2}}{e^{2}+e^{3}+e^{5}}, \frac{e^{3}}{e^{2}+e^{3}+e^{5}}, \frac{e^{5}}{e^{2}+e^{3}+e^{5}}]$，然后就可以当做概率与真正的$y$求出loss，反向传播。

对于softmax的求导其实比较简单，根据除法求导法则
$$
F(x) = \frac{g(x)}{h(x)}
$$
有导数：
$$
    \frac{dF}{dx} = \frac{g'(x)h(x)-h'(x)g(x)}{[h(x)]^2}
$$
所以对$z_k$求偏导可得：
$$
     \frac {\partial a_k}{\partial z_k} = \frac {(e^{z_k})'(\sum_{k}{e^{z_k}})-(\sum_{k}{e^{z_k}})'e^{z_k}}{(\sum_{k}{e^{z_k}})^2}
$$

由于求偏导数，
$$
    (\sum_{k}{e^{z_k}})'=\frac {\partial}{\partial z_k}(\sum_{k}{e^{z_k}}) = e^{z_k}
$$
所以：
$$
    \frac {\partial a_k}{\partial z_k} = \frac {(e^{z_k})\sum_{k}{e^{z_k}}-e^{z_k}e^{z_k}}{(\sum_{k}{e^{z_k}})^2} = \frac {e^{z_k}}{\sum_{k}{e^{z_k}}} - (\frac{e^{z_k}}{\sum_{k}{e^{z_k}}})^2 = a_k(1-a_k)
$$

但是，由于不光是$a_k$对$z_k$有导数，其实$a_0$对$z_1$也是有导数的，为什么呢，因为在$a_0$的计算式子中也有$z_1$嘛，所以当$i \neq j$时有
$$
    \frac {\partial a_i}{\partial z_j} = \frac {e^{z_i}e^{z_j}}{-(\sum_{k}{e^{z_k}})^2}  = -a_ia_j
$$

### Cross Entropy配合Softmax
softmax的一个典型配合的loss函数是Cross Entropy,这个函数是对多分类的一个loss函数，用来计算，输出的概率和目标0/1标签的差距，其公式是：
$$
L(y, a) = \sum_k {y \log a_k}
$$
其实理解起来很简单，因为除了label为1的那个第k位置导数为$-\frac{1}{a_k}$，其他的loss计算公式全都是0，所以也就没有梯度，此时，只有label为1的位置向前传梯度，我们假设label为1的位置为$a_y$，用上面那个例子，softmax的结果是$a = [0.2, 0.3, 0.5]$，而目标的$y = [0, 1, 0]$，那么$L = 0 \cdot log(0.2) + 1 \cdot log(0.3) + 0 \cdot log(0.5)$, 其导数$dL = [0, -\frac{1}{0.3} ,0]$，传下去。

### Cross Entropy - Softmax求导

刚才讲到，Cross Entropy的函数微分很简单，用数学等式表示其实就是
$$
    if \ y_k = 1 \ \ dL = - \frac {1}{a_k} \ \ Else \ \ dL = 0
$$

与Softmax的导数连起来就是，对于$k=y$的位置的梯度，利用上面的第一个梯度：

$$
\frac {\partial L}{\partial z_y} = \frac {\partial L}{\partial a_y}\frac {\partial a_y}{\partial z_y} = a_y(1-a_y)(-\frac{1}{a_y}) = a_y - 1
$$

而对于其他$k \ne y$的位置来说：

$$
\frac {\partial L}{\partial z_k} = \frac {\partial L}{\partial a_k}\frac {\partial a_k}{\partial z_k} = -a_ka_y(-\frac{1}{a_y}) = a_k
$$

我们简单整理一下上面的推导：label为1的位置概率减1，其他位置概率不变，也就是对z的梯度了。

还是用上面那个例子是，softmax的结果是$a = [0.2, 0.3, 0.5]$，我们计算出cross-entropy为$L = 0 \cdot log(0.2) + 1 \cdot log(0.3) + 0 \cdot log(0.5)$，那么$L$对$z$的导数为:$\frac {\partial L}{\partial z_k} = [0.2, -0.7, 0.5]$,由于一般Cross Entropy前面还会除以一个n(batch size)，所以，最后的$da_k$一般也会除以一个n

本篇笔记主要参考[知乎 自然语言处理和机器学习](https://zhuanlan.zhihu.com/p/25723112)

### 具体求导实现

参考[CS231n](http://cs231n.github.io/neural-networks-case-study/)中softmax的方法我们可以写一个简单的softmax回归代码：

```python
#Train a Linear Classifier

# initialize parameters randomly
W = 0.01 * np.random.randn(D, K)
b = np.zeros((1, K))

# some hyperparameters
step_size = 1e-0
reg = 1e-3 # regularization strength

# gradient descent loop
num_examples = X.shape[0]
for i in xrange(200):
  
  # evaluate class scores, [N x K]
  scores = np.dot(X, W) + b 
  
  # compute the class probabilities
  exp_scores = np.exp(scores)
  probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
  
  # compute the loss: average cross-entropy loss and regularization
  corect_logprobs = -np.log(probs[range(num_examples), y])
  data_loss = np.sum(corect_logprobs) / num_examples
  reg_loss = 0.5 * reg * np.sum(W * W) # L2 regularization
  loss = data_loss + reg_loss

  if i % 10 == 0:
    print "iteration %d: loss %f" % (i, loss)
  
  # compute the gradient on scores!!!这里就是求softmax+cross entropy的具体实现
  dscores = probs
  dscores[range(num_examples), y] -= 1
  dscores /= num_examples
  
  # backpropate the gradient to the parameters (W,b)
  dW = np.dot(X.T, dscores)
  db = np.sum(dscores, axis=0, keepdims=True)
  
  dW += reg * W # regularization gradient
  
  # perform a parameter update
  W += -step_size * dW
  b += -step_size * db
```

