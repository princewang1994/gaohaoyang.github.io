## Region Proposal by Guided Anchoring

[TOC]

* 文章链接：[Region Proposal by Guided Anchoring](https://arxiv.org/pdf/1901.03278)

Region Proposal by Guided Anchoring是港中文2019年提出的anchor自动生成方法，其主要贡献是提出了一种代替滑动窗口式生成anchor覆盖图像的所有像素的方法，大量减少了anchor的数量，并且根据图像内容生成对应的anchor，有效的提升了2stage检测网络的训练速度，是目标检测在“anchor-free”，或“fewer-anchor”上的一种探索。


### Anchor机制带来的问题

目前大多数的one-stage/two-stage网络都基于anchor机制，所谓anchor机制就是在模型开始训练之前就定义好一些固定大小，长宽比例的锚点框，使用滑动窗口的方法在特征图上的每一个位置都生成相同数量的anchor，那么Anchor机制有什么弊端呢？

- 需要针对不同的问题设计不同的Anchor大小，ratio，一个错误的设计可能导致网络训练无法收敛，性能不佳等问题
- 为了保证anchor的recall，anchor的数量需要足够多，造成了大量的假阳性，阴阳性比例不平衡问题十分严重
- 大量的anchor给计算带来了较大的负担，也降低了网络训练的速度 

### GA-RPN

#### GA-RPN的引入

为了解决传统Anchor带来的问题，GA-RPN提出，不需要在特征图上的每一个点都生成anchor，可以先用网络预测出那些地方需要anchor，这样做既能大量降低anchor的使用数量，而且使anchor的生成取决于图像本身，更加灵活而且与图像内容相关，总结一下，GA-RPN的优点主要有以下几方面：

- 训练/预测速度快
- RPN的Recall高
- 由于anchor是可学习的，不是预先定义的，所以对于那些形状特殊的物体：特别高的物体或者特别宽的物体效果比较好
- 可以加入到任意基于anchor的网络中

为了介绍GA-RPN，我们首先提出两个对于RPN非常有用的要点：**对齐**和**一致性**
- **对齐**：anchor的中心必须和object的中心对齐
- **一致性**：对于每个anchor来说，其大小需要与生成他的feature的感受野尽量一致

传统的detector没有考虑特征和anchor之间的对齐，比如一个物体最终用于预测他的anchor所在的grid不一定是在他的中心，而anchor在总体上学习到的是以他的感受野为中心的特征，因此这种不一致性可能导致预测的偏移，因此需要使用bbox regression来把这部分误差偏移回来，GA-RPN在设计的时候充分考虑到了“对齐”和“一致性”，主要方法分为两个大步骤：

1. 找到可能包含object的区域
2. 确定anchor的scale和ratio，这两个也是可学习的，在原来的anchor中是固定的

总结起来就是这个公式：

$$
p(x, y, w, h|I) = p(x,y|I) p(w, h | x, y, I)
$$

- $p(x, y|I)$,表示输入输入图像I，预测那些地方存在物体
- $p(w,h |x,y,I)$表示已知图像，anchor中心位置，预测的Anchor的长和宽

![](http://princepicbed.oss-cn-beijing.aliyuncs.com/blog_20190421222947.png)

以上是GA-RPN的主框架，在每一个feature上都预测一些anchor，这些每个位置预测3个数：

- 是否有object，用NL子网络来预测
- object的高宽，wh，用NS子网络来预测

#### 中心点预测

NL子网络用来预测当前位置是是某个object中心的概率，只包含了一个1x1的卷积用于缩减通道数，出来的概率经过一个sigmoid归一。

那么用什么标签来训练这个网络呢？根据下面这张图，文章把一个object分成CR，IR和OR三个部分：

- $CR = R(x_g' , y'_g , σ_1w' , σ_1h' )​$，以gt的中心为中心，把h和w都乘上缩小因子σ1
- $IR= R(x_g' , y'_g , σ_2w' , σ_2h' )$，以gt的中心为中心，把h和w都乘上缩小因子σ2
- CR和IR满足（$\sigma2>\sigma1$）
- 图中绿色的部分为CR，黄色的部分为IR
- 除了CR和IR的部分，其他为OR部分，作为负例训练，CR作为正例训练，IR不进行反向传播

![](http://princepicbed.oss-cn-beijing.aliyuncs.com/blog_20190421223023.png)



#### Anchor形状预测

对于形状预测，首先，w和h的范围很大，可能是1000以上，这样直接回归是很难学习的，因此需要归一化，作者这里使用了log进行归一化：

$$
w = \sigma \cdot s \cdot e^{dw} \ \ \ \ \ \ h = \sigma \cdot s \cdot e^{dh}
$$

σ是一个经验预设值，文章实验中设为8，s即当前feature map的stride，比如在一个下采样32倍以后的特征图上，w = 8 * 32 * exp(dw)，实际上网络预测的是dw，根据这个预设我们知道，在32x这个特征图上，预测出dw∈[-1, 1]就相当于预测出w∈[94, 696]这么大范围的w，对于大部分目标检测物体已经足够了，下采样次数越小（浅层特征图），预测的框就越小，这也比较符合直觉。

同样的，用什么样的标签来训练NS呢，因为在Faster RCNN中，很容易计算出某个anchor和gt的iou，然而，在我们不知道w和h的情况下无法计算，由于要满足对齐的目的，输出的anchor框是不能移动的，只能改变w和h让他和对应gt的IoU最大，于是不管怎么做都无法满足anchor和gt完美重合，因此文章定义了vIoU：

$$
vIoU(a_{wh}, gt) = \max_{w>0,h>0}IoU_{normal}(a_{wh}, gt)
$$

意思是调整w和h的值，使某一组w和h的pair能够使IoU(a, gt)最大，可想而知，wh的范围非常大，根本无法遍历所有的组合，因此实验中只选择了预设好的9组wh对，选取其中最大的一组，作为NS网络在这个位置回归的目标值。

**用什么loss来回归dw和dh？**

这里采用了iou loss，不过做了一个小小的变化，就是只求取w和h的loss，忽略x和w，原因很显然。

#### Anchor-Guided Feature Adaptation

之前的两阶段模型因为anchor是预设好的，因此每个特征图上特征的scale都比较统一，比如全是很小的或者全是很大的，因此是满足上面说的一致性的。使

用GA-RPN的方法会在同一个特征图上出现较大的不一致性，同一个每一个location的特征有大有小，这样很不利于学习，试想一下，后面使用某个proposal在特征图上做ROIPooling的时候，会使用到这些特征，不一致会导致后面的分类和回归效果变差。

因此GA-RPN又做了一个想让不同的位置学习到不同感受野的操作，我们希望每个位置的感受野是不一样的，如果某个点预测的wh更大，那么他的感受野也应该更大，如果预测的wh小，他的感受野也应该小，满足以下公式：

$$
\mathbf{f}_{i}' =\mathcal{N}_{T}(\mathbf{f}_{i}, w_{i}, h_{i})
$$
这就需要可形变卷积（deformable convolution）来帮忙了，看以下部分：

![](http://princepicbed.oss-cn-beijing.aliyuncs.com/blog_20190421223049.png)

先把NS的输出（w,h）作为输入预测过1x1的卷积提取特征，然后这个作为deformable convolution的感受野输入，强行改变特征的感受野，最后得到一个新的特征图，作为后续ROIPooling，proposal分类和回归的特征

#### 如何使用GA-RPN

​      GA-RPN可以直接接入Faster-RCNN做做end2end训练，但是实验发现提升点数有限（不到1个点）但是发现，如果提高positive的阈值，效果就会提升，原因是GA-RPN的proposal质量比普通的RPN高，因此提高阈值可以使用到更高质量的，不使用那些低质量的proposal

​      另外文章也提到，可以使用fine-tune的方法来提升现有的两阶段模型，如何做呢：

- 去掉原始两阶段模型的RPN部分
- 换成GA-RPN训练3个epoch
- 使用GA-RPN来做inference

这样做精度提高很多（large margin）

### 实验效果

下图是普通RPN和GA-RPN的比较，可以看出GA-RPN生成的anchor框数远远少于普通的RPN，而且质量也更高：

![](http://princepicbed.oss-cn-beijing.aliyuncs.com/blog_20190421223118.png) 

### 总结

GA-RPN提出了一种anchor的预测方法，能够大量减少基于sliding window方法的RPN的anchor数量。在方法的设计上，主要围绕“对齐”和“一致性”两个理念。对齐，意思是anchor的位置相对于产生他的feature不会有偏移，每个位置只学习w和h，一致性，至anchor预测出来某位置i，这个位置产生的feature的感受野需要和他预测的w和h大小尽量一致。文章使用了deformable conv来解决GA-RPN的feature和anchor size不一致的问题。

- 参考博客：https://zhuanlan.zhihu.com/p/55314193