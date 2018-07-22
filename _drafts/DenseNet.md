# DenseNet

### 基本思想:

在每一个block内，每层卷积的输入是所有前驱层的concat，同时本层是输出也是所有后继层输入之一

![](media/15042697070054.jpg)

**Trick:**

- 两个block中间使用transition layer连接
- `DenseNet-B`: 使用bottleneck层作为两层间传递的结构，减少参数的数量，不同于ResNet的bottleneck，这种bottleneck被定义为：

    **BN-ReLU-Conv(1×1)-BN-ReLU-Conv(3×3)**
    
    bn -> relu -> **conv(1x1, 4k)** -> bn -> relu -> **conv(3x3, k)**
    
> 这里减少的数量还是挺可观的，假设在一个含有48层的DenseBlock中，那么最后一层的输入就是[X, X, 48k]直接3x3卷积需要[3, 3, 48k, k]，使用了bottleneck以后需要[1, 1, 48, 4k]+[3, 3, 4k, k]大概减少了一半的参数，控制计算量
    
- `DenseNet-C` 在transition层使用参数$\theta$控制特征的增长量，假设某层的特征输出通道数为m，那么经过transition层后数量降低为$\theta m, 0 < \theta \le 1$, DenseNet-C使用了$\theta = 0.5$
- 如果B和C两个trick都使用到，那么模型叫做`DenseNet-BC`

### 实现细节

以下是Paper中提到的，ImageNet的训练结构：

![ImageNet](media/15042701885526.jpg)

在进入第一个Dense Block之前，将图片使用16通道的卷积层处理一遍这个16有可能是24或32，是一个超参数，在init conv阶段，把特征缩小为原来的4倍，对于所有3x3的卷积层，都是用了1的padding来保持形状不变。在经过4个Dense Block以后，有缩小8倍，一共是32倍（与VGG的缩小倍数相同，最后的特征数是7x7）,表中所有的conv都是**BN-ReLU-Conv**

在每个Dense Block中，参数有：

- l：每个block有多少层，即第一列的6, 12, 24, 16
- k: 增长率，即每层把所有的输入卷积到的目标通道数

具体的实现如下图： 

![](media/densenet.png)

**可以看到，逻辑上每一个layer都是所有后续layer的输入，在实现上，首先把上一层的所有输入卷积为k层，然后只要把上一层的输出concat到输出即可，比如把上面虚线框中所有通道卷积为k(黄色的块)然后把上层虚线框中的块原封不动的concat到下层**

在每两个连续的block之间，transition层使用conv(1x1)-avgpool(2x2)连接。

在最后一个block后，使用global average pooling添加正则化，而不使用L1L2正则化。

这里有一个[PyTorch实现](https://github.com/gpleiss/efficient_densenet_pytorch/blob/master/models/densenet.py)和一个[Tensorflow的实现](https://github.com/ikhlestov/vision_networks/blob/master/models/dense_net.py)

### 网络结构

***

**DenseBlock**：

```flow
input=>inputoutput: Input[X, X, m]
bottleneck=>operation: BatchNorm
                    ReLU
                    Conv 1x1, 4k
     
convolution=>operation: BatchNorm
                    ReLU
                    Conv 3x3, k
dropout=>operation: Dropout

e=>inputoutput: Output[X,X,k]
input->bottleneck->convolution->dropout->e

```
作用是把任何进来的m通道的特征图转换为k通道，特征图大小不变，上面的结构是Bottleneck层DenseNet-B中使用了这种结构

***

**TransitionBlock**：

```flow
input=>inputoutput: Input[X, X, m]
convolution=>operation: BatchNorm
                    ReLU
                    Conv 1x1, θm
avgpool=>operation: AveragePool 2x2
e=>inputoutput: Output[X/2, X/2, θm]
input->convolution->avgpool->e

```
把m通道的特征图转换为$\theta m$通道

***

**整体结构**：

```flow
input=>inputoutput: Input[X, X, 3]
init=>operation: init Conv 3x3, 16
block1=>operation: DenseBlock1
transition1=>operation: Transition1
block2=>operation: DenseBlock2
transition2=>operation: Transition2
block3=>operation: DenseBlock3
avgpool=>operation: AveragePool 8x8
bn_final=>operation: BatchNorm
e=>inputoutput: Output[1, 1, θm]
input->init
init->block1->transition1
transition1->block2->transition2->block3
block3->bn_final->e

```

===

**2017.9.3更新：**

### DenseNet For Semantic Segmentation

使用DenseNet结构完成语义分割任务，整体结构类似与U-Net，在upsample分支添加downsample的skip connection最后输出与原图一样的分辨率，注意这里还有一个TU(Transition Up)，包含了一个2x反卷积，为了扩大左侧avg_pooling造成的分辨率降低。

**整体结构如下：**

![](media/15044163099329.jpg)

**TU(Transition Up):**
TU的做法是先把上一层的顶端layer做一个2x的反卷积，然后与左侧对应的skip层concat起来，注意这里的反卷积通道数$m=last\_block\_layer*growth\_rate$

|TU+Concat|
|:---:|
|deconv 3x3 stride=2|
| channel=last_block_layer*growth_rate|
|concat [skip_conn, deconv_out] = channel+skip_conn|

![](media/15044162375258.jpg)

以上是文章中几种结构的配置详情，需要注意的就是TU的卷积通道数文章没有提到，阅读源码[FC-DenseNet](https://github.com/SimJeg/FC-DenseNet/blob/master/FC-DenseNet.py)以后知道获取通道数的方法是**上一个block的卷积层数 x 增长率**

文章最后说到，总体的层数是：
![](media/15044219361461.jpg)
但是对源码分析以后发现应该是：

| layers | channels |
| --- | --- |
|4 layers|m=112|
|5 layers|m=192|
|7 layers|m=304|
|10 layers|m=464|
|12 layers|m=656|
|15 layers|m=896|
|12 layers|m=1088|
|10 layers|m=816|
|7 layers|m=576|
|5 layers|m=384|
|4 layers|m=256|
|1x1 conv|m=c|
|softmax|

===

### Effective-Memory DenseNet

2017.9.7更新

最近使用FC-DenseNet(Tiramisu)进行语义分割，发现一个比较大的问题，就是显存的占用量非常大，其中最明显的是增加`growth rate`带来的显存增幅，限于GPU的数量和Memory的大小，512的结构如果需要在单张显卡上完成是非常困难的。因此找到了一种占用显存较少的实现[efficient_densenet_pytorch](https://github.com/gpleiss/efficient_densenet_pytorch/blob/master/models/densenet_efficient.py)，主要理论支持可以参考这篇[技术报告](https://arxiv.org/pdf/1707.06990.pdf)，这个Repo也是DenseNet的作者小组在GitHub上发布的，因此代码写的也比较好。

#### DenseNet显存占用量大的原因

我们知道，DenseNet中最重要的操作是**Concat**，在每一个Block中，把当前层之前所有层的输出全部都concat到当前层的输出上，这也就是DenseNet的基本结构了，另外，DenseNet中大量存在这**BatchNorm**层，对显存的影响也十分巨大，因为在每一个BatchNorm中，Forward的时候，需要计算每一个位置的均值(除了通道维度，假设一个[5, 224, 224, 3]的Tensor需要计算5x224x224个中间结果)，和方差，因此非常消耗内存，下设每一层都加BN和Concat，那么对显存的影响是巨大的。

**BatchNorm 和 Concat层的共同点**

技术报告中提到，BatchNorm和Concat均为“计算简单，但是占用显存较大的层”，BN的本质就是一个O(n)的求均值的算法，而Concat更是如此，这两个计算比起动辄$O(mn^3)$的FC层和Convolution层，计算复杂度是非常低的，这么说的话，同样频繁出现的**ReLU**层也具有相同的特点，而这篇技术报告并没有将ReLU考虑在其中，**原因是目前存在的框架大部分都将ReLU实现为inplace的，因此并没有占用额外的显存**

#### 如何解决

简单来说解决方法就是：**对BN和Concat使用ShareMenory机制，不预先分配给显存给BN和Concat(因为他们都占用显存量大)，当计算向前传播，开始计算新的BN和Concat的时候，覆盖ShareMenory中旧的结果，而不缓存他们的计算结果。**

那么反向传播的时候需要计算他们的缓存值该怎么办呢，**这个时候只需要即使演算出来即可，原因是他们的计算时间通常很短**，这是一个典型的用时间换空间的方法，文章中提到，这样会是网络的训练时间增加5%~15%但是内存用量大量减小了。

以下是示意图，其中右下角的SharedMemory Storage1负责Concat，Storage2负责BN。新的内存高效的DenseNet使得200层以上的DenseNet得以训练，也给网络进一步加深带来了不错的前景

![](media/15047704776503.jpg)

代码实现比较复杂，这里就先不深究了，但是这也给以后实现或者设计网络结构一些启发，时间换空间也不失为一种巧妙方法。

===


### 总结

DenseNet在结构上非常简单，而效果却比那些复杂结构还要好的原因是：**他的skip connection，大多数具有skip connection的结构（如ResNet）都可以看做是DenseNet的特例**，只需要把某些连接的权值设为0即可，因此DenseNet的能力更强，在全局只有4个block的情况可以达到很好的训练效果，而其参数远小于其他skip connection的数量，CVPR中出现如此优雅而简洁的结构，best paper当之无愧。





