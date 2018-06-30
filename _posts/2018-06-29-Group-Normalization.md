---
layout: post
title:  Group Normalization
date:   2018-06-29
categories: Deeplearning
tags: MachineLearning DeepLearning ComputerVision
mathjax: true
author: Prince
---

* content
{:toc}

文章地址[ arXiv:1803.08494](https://arxiv.org/abs/1803.08494)

Group Normalization是今年何凯明大神的新作，抽时间拜读了一下，收获良多，也给了我在normalization和group conv上的一些新的理解，由于篇幅原因，在这篇博客中没有添加分组卷积的内容，如果有需要以后会继续补充。




## 为什么需要规范化

规范化（Normalization）在批标准化（Batch Normalization）被提出来以后被广泛运用于各种模型的学习中，其中最著名的就是ResNet了，利用BN-Conv-ReLU作为最基本的模块构成残差块，再加上残差连接，就构成了ResNet的基本网络结构。批标准化可以使网络的传输过程中，梯度更加稳定，如何解释呢：

假设一个Linear层，经过激活函数(如Sigmoid)函数以后，如果值太大或者太小，往往导致反传回来的梯度非常小(S型函数两遍梯度很小)，经过几层以后，梯度越来越小，就会造成所谓的“梯度消失”，反之如果太大，就会造成梯度爆炸。使用BN过后在用激活函数能够把值控制在一个有效的范围内，这个值一般是-1到1范围，也就是一个标准正态分布的分布区间。后来被实验证明，在不是S型函数的激活函数中，BN也能起到不错的效果。

在BN出现之前，也有许多结构用来解决梯度回传不稳定的问题，其中比较著名的是Local Response Normalization(LRN)，使用相邻通道来做Norm，公式可以表示成这样：

$$
b^{i}_{x,y} = a^{i}_{x,y} /(k + \alpha \sum_{j=\max (1-n/2)}^{\max (N-1, 1+n/2)} (a^{j}_{x,y})^2 )
$$

看这个公式这么复杂，其实就做了一个事情：把当前通道的上面n/2和下面n/2个通道求个二范数，然后除以这个二范数，起到一个normalization的目的。

## BN存在的问题

BN大法虽然好，但是仍存在着一些问题：

1. BN只在**训练过程**中起作用
2. BN在Batch**小**的时候，梯度计算不稳定

第一个问题，以图像处理为例，BN的作用是在training-time，在训练的时候通过计算同一个Batch中所有图片同一个通道的均值和标准差来规范化，换句话来说，BN在训练时候单个input的输出，取决于当前batch的所有输入，而不仅仅取决于输入。这个特性在**训练**的时候可以看做是一种正**则化**，因为每一个epoch训练时batch都会被打乱，不同的图片被重新组合到一个batch中，增加了BN在计算时候的多样性。

然而在**测试时**这个特性会使预测不稳定，原因是在测试的时候我们希望测试的输出仅仅取决于输入，和同一个batch中的其他输入（图像）没有关系。试想如果这个模型被运用于重要场合，比如诊断，在测试的时候如果也是批处理，一个患者的特征和A同时输入是一个结果，和B同时输入时又换了一个结果，那么这个系统的鲁棒性就大打折扣了。因此BN的原始论文提出：

> The normalization of activations that depends on the mini-batch allows efﬁcient training, but is neither necessary nor desirable during inference; **we want the output to depend only on the input, deterministically**. For this, once the network has been trained, we use the normalization

好了，第一种的解决办法有了，就是使用BN中提出来的Moving Average，简单来说，就是BN层计算它见过的所有通道值的均值，在预测的时候仅仅只用这个均值代替训练时的batch均值。

这个问题虽然解决了，但是并不能保证，计算过程中计算出的均值是稳定的，这也就是第二个问题：当batch小的时候，moving averge的计算不如大Batch的时候准确。计算的average不准，自然小batch的时候效果和大batch之间差距比较大。然而在一些需要高分辨率的任务（如检测，分割等）上，我们碍于图像尺寸，又不得不选择小Batch训练，在这种情况下，BN往往不能发挥出很好的作用。

## 与其他规范化方法的关系

为了解决BN必须要借助同一个Batch中其他数据的问题，提出了许多方法，其中本文中提到了Layer Norm和InstanceNorm。下面我们会介绍这两种规范化方法：

### 插播

这里我们和GN文中一样使用图像作为例子说明GN和BN，LN，IN的关系。由于图像和普通数据维度不完全相同，这里有必要讨论一下BN在图像中使用与在普通特征数据的区别：

我们知道在传统机器学习中，一个样本被描述为多个特征比如（身高，体重，年龄）等等，在经过量化，bin，enumerate之类的处理以后，每个样本将会被表示为一个N维向量(x0, x1, ... x_n)，这个向量中，每个数据的量纲都是不一样的，比如升高是200以内的小数，年龄是100以内的整数等等，自然需要把每个特征都进行归一化，我们把一批N个样本描述为一个(Nxn)的矩阵，对这个矩阵的第二个维度进行标准化，即把所有的身高都归一化，这样，所有的数据都会是一个接近[-1, 1]的小数，因此在普通的特征数据中，BN需要为每个特征计算一个mean和std，然后做归一化。

那么情况到了图像中是怎么样呢，图像数据的维度应该比普通特征数据高两个维度，一批N个数据会被描述为一个（NxHxWxC）的矩阵，最直观的方法会认为，直接把后面几个维度拉平，变成一个Nx(HxWxC)的矩阵，然后按照传统的方法去做，这样需要计算的mean就是HxWxC个，相当于对每个通道的每个点都要计算一个均值。其实BN在实现的时候都没有这么做，而是把一个通道作为一个norm的维度，值计算C个均值，这是为什么呢，我的理解是CNN网络中一个特征图的同一个通道是由同一个卷积核生成的，因此其分布与这个卷积核的分布高度相关，所以某种意义上看，其实同一个通道可以认为是同一个特征。**因此每个通道求一个mean和std就好啦**。

接下来说正事：BN，LN，IN，GN的关系，先上图：

![](http://oodo7tmt3.bkt.clouddn.com/blog_20180629220007.png)

我们假设所有的normalization方法都是为了求某一个集合$\mathcal{S_i}$的均值很标准差，只是$\mathcal{S_i}$略有所差别，上图很好的描述了各种Norm方法的差异，先看坐标轴，这里为了降低维数，把一张特征图的H,W维度给展开了，其中蓝色的方块表示属于$S_i$的元素。

- **Batch Normalization(BN)**

对于我们熟悉的BN，就和我们刚才说的一样，一张图中每个通道一个mean，这个mean是通过N张图片中同一个通道的所有像素值求出来的
$$
\mathcal{S}_i = \{k|k_C = i_C\}
$$

- **Layer Norm(LN)**

一张图一个mean，使用这张图中所有的像素计算，不管通道
$$
\mathcal{S}_i = \{k|k_N = i_N\}
$$

- **Instance Norm(IN)**

和BN比较像，一张图片一个mean，不过mean只由本张图片的mean构成，不涉及同batch中其他图像
$$
\mathcal{S}_ i= \{k|k_N = i_N, k_C = i_C\}
$$

- **Group Norm(GN)**

把一张图片分为G个组，每个组包含C/G个通道，这些通道使用同一个mean来norm，不包括其他图像
$$
\mathcal{S}_ i= \{k|k_N = i_N, \lfloor \frac{k_C}{C/G} \rfloor = \lfloor \frac{i_C}{C/G} \rfloor\}
$$

注意到这几种方法中，只有BN使用到了同一个batch里面的多张图像，其他的均只使用了当前图片的信息，因此除了BN，其他的都不需要记录runing mean,GN可以看成是LN和IN的中间版本，当G=C就是IN，当G=1就是LN，他们的共同特点是，训练过程中不需要记录均值，训练和验证行为完全一致。

GN这种思想借鉴了目前比较流行的Group Convolution的思想，认为相邻的通道表示的特征可以有一致性。文章后面通过实验证明了，GN在大Batch训练师在性能上能够接近BN的效果，而在小batch的时候能够和大Batch拥有几乎一样的性能。

## 实现
### Tensorflow实现

```python
def GroupNorm(x, gamma, beta, G, eps=1e−5):
    # x: input features with shape [N,C,H,W]
    # gamma, beta: scale and offset, with shape [1,C,1,1]
    # G: number of groups for GN
    N, C, H, W = x.shape
    x = tf.reshape(x, [N, G, C // G, H, W])
    mean, var = tf.nn.moments(x, [2, 3, 4], keep dims=True)
    x = (x − mean) / tf.sqrt(var + eps)
	x = tf.reshape(x, [N, C, H, W])
    return x ∗ gamma + beta
```

### PyTorch官方实现

```python
 class GroupNorm(Module):
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super(GroupNorm, self).__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_channels))
            self.bias = Parameter(torch.Tensor(num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            self.weight.data.fill_(1)
            self.bias.data.zero_()

    def forward(self, input):
        return F.group_norm(
        	input, self.num_groups, self.weight, self.bias, self.eps)
```

PyTorch中F.group_norm暂时还没找到在哪实现的，不过应该和tf的实现差不多，其实就是Reshape+norm的一个操作，然后就是一个BN经典的重分布操作（$\gamma x + \beta$）使分布可学习。

## 实验

实验代码[在这里](https://github.com/facebookresearch/Detectron/ blob/master/projects/GN.)可以找到，这里我们重点看下面两张图：

![](http://oodo7tmt3.bkt.clouddn.com/blog_20180629215920.png)

上图描述了BN等一系列方法在大batch（size=32）时训练和验证上的误差表现，可以看到，只对比BN和GN的话，GN的过拟合比BN严重一些，这可能是因为GN没有我们在第一小节中讲到的BN的“正则化”效果有关系，尽管如此，GN还是比LN和IN要强一些。文章也提到，在GN中增加合适的正则化约束有可能改进GN的效果。

![](http://oodo7tmt3.bkt.clouddn.com/blog_20180629215940.png)

接下来重点看小batch上的情况，我们看到，随着batchsize的缩小，BN的性能急剧下降，在2 -> 4的过程中下降尤为显著，而GN在batch size下降的过程中性能几乎不受影响。这个实验证明了GN在小batch情况下的性能突出。

## 总结

本篇博客从BN和GN理解出发，主要讲了一下自己对规范化，BN，在图像中使用BN和GN的一些见解，最新的网络中，ResNext，Xception，ShuffleNet等结构，都开始在通道维数下功夫，我们一直习惯的“卷积核通道数必须与特征图通道数相同”的观念也逐渐被打破，GN，Group Conv等操作在接下来，可能还会发掘出其他的功能。

就本篇文章来说，我认为结合了LN，IN，BN以及目前流行的分组方法，在LN和IN中间找到一个合适的位置，并超过了IN和LN，Normalization这个点非常基础，玩的好就能获得很不错的效果。

欢迎讨论！