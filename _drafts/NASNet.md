## Learning Transferable Architectures for Scalable Image Recognition(NASNet)

## motivation

目前的分类网络等设计基本都是人工设计结构，比如ResNet，ResNext等等，这些人工设计的结构不一定能达到最优，本文希望能让网络自己学习一种网络结构，本质上是一种网络空间上的搜索，找到一个在小数据集上最优的结构，迁移到大数据集上。

本文的贡献主要有以下几点：

- 设计了一种基于RNN的和增强学习的网络搜索算法，利用小数据集搜索网络最优结构
- 提出了在Cell上而不是在整个网络上搜索结构，减小搜索空间，并增加泛化能力
- ImageNet分类上比state of art提升3.1%
- Faster-RCNN上作为骨干网络提升4.0%

## 限定搜索空间

很容易想到的是，在数据集较大的时候，每一组参数（网络层数，与那个层连接等等）都需要花很长的实现在大数据集上训练一遍，然后才能得到这组参数所对应的性能指标，因此非常耗时，几乎是不可能完成的。所以NASNet提供了以下几个方法加速搜索的过程：

- 在小数据集上训练获得指标，然后在大数据集上应用，由于网络的性能具有相对性，如果一组参数在小数据集上的性能比另一组参数好，那么有很大的可能在大数据集上也好于另一组参数。这样可以快速重复试验。本文选择了CIFAR100作为搜索的数据集
- 即使是使用小数据集，网络的搜索空间也是巨大的，几乎无法穷尽，所以本文将问题简化，先设计好网络的主结构（block之间的顺序连接），然后只搜索每个Block内部的连接，每个block内部连接完全一样，这样大大缩小了搜索空间，每个block叫做一个Cell

### 使用Cell搜索的优点

使用Cell搜索的有以下几个优点：

- 加速搜索： 如上面所说的，只需要搜索有限的结构，使搜索网络变的可行
- 增加了网络的扩展性： 加入在整个空间搜索网络，可能容易搜索出一个只适用于CIFAR的网络，迁移到大数据集上以后的效果很差。如果使用Cell搜索，确保每个Cell在CIFAR上的性能是最优的，然后迁移到大数据集上的时候可以根据数据集的大小适当增加Cell的数量和结构，这样可以有效的迁移模型
- 获得灵活的性能：比如目标是移动设备或者其他计算能力有限的设备的时候，可以使用已经搜索好的Cell作为基础结构灵活的减小模型，适应任意计算能力。
- 其他任务迁移：具有泛化能力的Cell方便的作为其他任务(检测，分割等)的迁移，实验证明效果也不错

## Method

### 两种不同的Cell

就像ResNet的Residual block一样，要实现分类，必须使特征图分辨率逐层降低，而且也得存在中间分辨率不变的特征提取层，因此本文设计了两种Cell：

- Normal Cell: 输出和输入特征图大小一致的特征图的cell（类似于Residual Block）
- Reduce Cell: 输出的特征图是输入特征图长和宽都是输入1/2的Cell，对于这种Cell，把第一个卷积层（或者pooling层）的stride设为2，然后接下来所有的操作都基于新的分辨率（1/2H和W）

![](http://oodo7tmt3.bkt.clouddn.com/blog_20180826203701.png)

如图是为CIFAR10和ImageNet设计的两种结构，分为Normal Cell和Reduce Cell，用不同的N来调整适应不同的数据集，对于更大的输入图像，需要加入更多的Reduction Cell，对于更大数量的数据集也需要使用更深的网络来增加参数。

注：上面的所有Cell操作都没有通道层面的搜索，只包含卷积的kernel大小，是否是depthwise搜索，是卷积还是池化等。也就是说，搜索不会包含每个卷积层的输出通道数，对于通道是怎么设计的呢，这里作者使用了一个比较启发式的规则，也就是：如果特征图分辨率缩小了两倍，那么通道数就变成前一层的两倍，在每一个Reduction Cell使用，这里也进一步限定了搜索空间。

### 搜索策略

搜索是通过一个T=t的RNN完成的，主要使用了LSTM，每次产生一个长度为t的序列，接下来说明搜索的过程：

![](http://oodo7tmt3.bkt.clouddn.com/blog_20180826205007.png)

上图搜索的示意图，t=5，每次输出5个选择：初始化有两个hidden state，前一个cell的输出$h_i$和前两个cell的输出特征图$h_{i-1}$
1. 选择第一个hidden state 可以在$h_i$和$h_{i-1}$里面去选，也可以在前一个block的输出里面选
2. 选择第二个hidden state （同上）
3. 选择第一个hidden state的操作，对1.的操作，比如pooling, conv
4. 选择第二个hidden state的操作，对2.的操作，比如pooling, conv
5，选择一个方法来结合3和4输出的特征图（elementwise add，mul，concat）

重复1-5步骤B次，将所有生成的特征图concat到一起，生成本次的输出$h_{i+1}$，实验中B=5，其中第3，4步的操作限定在以下的几种操作中：

- identity
- 1x7 then 7x1 convolution
- 3x3 average pooling
- 5x5 max pooling
- 1x1 convolution
- 3x3 depthwise-separable conv
- 7x7 depthwise-separable conv
- 1x3 then 3x1 convolution
- 3x3 dilated convolution
- 3x3 max pooling
- 7x7 max pooling
- 3x3 convolution
- 5x5 depthwise-seperable conv

总结起来就是选择kernel大小，核的长宽比，conv/pooling和是否depthwise操作，这里的depth-wise convolution其实就是Xception里面说的，使用G=N的分组卷积，把每个通道分为一组，减少参数量的正则化方法。

这样每一次RNN都会生成5xB的序列，然后组成一个网络，将这个网络在CIFAR10上训练到收敛，得到一个指标，这里我们可以认为，CIFAR10相当于是增强学习中的environment，通过增强学习的方法，可以优化RNN的参数，最后可以输出一组最优的参数，也就是我们的网络结构了。



## 实验

实验阶段给出了在CIFAR10上搜索得到的最好的两种Cell，下面两个图就是，实验证明，使用这种搜索得到的Cell，在ImageNet上获得不错的提升（实验使在500个GPU上搜索的，有钱真好……）

![](http://oodo7tmt3.bkt.clouddn.com/blog_20180826212119.png)

![](http://oodo7tmt3.bkt.clouddn.com/blog_20180826212144.png)



可以看到，NASNet在同等参数量上，对比DenseNet还是有提升的。

![](http://oodo7tmt3.bkt.clouddn.com/blog_20180826212837.png)

## 总结

NASNet是一种网络结构上的新探索，使网络结构本身能够学习，个人认为其中的几个亮点，如降低搜索空间的方法，以及小数据集优化大数据集迁移的思想，都是值得学习的，为了使整个搜索方案可行，大量降低了搜索空间，也可能使网络“错过”一些优秀结构，比如不同的通道数量，以及Block之间的连接（典型的Unet这样的Encoder，Decoder结构）使整体结构达不到最优，但是碍于目前的硬件水平，能做到这么大规模的搜索实属不易，并且作者也已将代码开源，对于已经训练出来的结构，也很有价值，目前已经看到了一些其他文章的实现开始使用了NASNet搜索出来的结构，很期待网络结构搜索上的新气象。






