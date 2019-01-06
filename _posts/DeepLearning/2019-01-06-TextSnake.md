---
layout: post
title: "TextSnake: A Flexible Representation for Detecting Text of Arbitrary Shapes"
date: 2019-01-06
categories: DeepLearning 
tags: DeepLearning SceneTextDetection ObjectDetection
mathjax: true
author: Prince
---

* content
{:toc}

![image-20190106142351222](http://princepicbed.oss-cn-beijing.aliyuncs.com/blog_201901061654440574.png)

* 文章链接： [TextSnake: A Flexible Representation for Detecting Text of Arbitrary Shapes](https://arxiv.org/abs/1807.01544)
* PyTorch实现: [princewang1994/TextSnake.pytorch](https://github.com/princewang1994/TextSnake.pytorch)

最近由于项目上做到与文本检测相关的应用，因此研究了几篇最新的文本检测文章，本文（TextSnake）是旷视在ECCV2018上的文章，主要提出了一种能够灵活表示任意弯曲形状文字的数据结构——TextSnake，主要思想是使用多个不同大小，带有方向的圆盘(disk)对标注文字进行覆盖，并使用FCN来预测圆盘的中心坐标，大小和方向进而预测出场景中的文字，本文的主要贡献有以下几点：

- 提出一种基于多圆盘覆盖来表示文字的方法——**TextSnake**
- 基于TextSnake表示，提出一种检测场景文本的方法
- 在两个新提出的数据集（**Total-Text** and **SCUT-CTW1500**）上，获得了state-of-the-art的效果，这两个新的数据集都包含曲线文本，波浪形文本等非常规标注




### 文本检测的难点

场景文本检测（Scene Text Detection）的主要任务是给定场景图，要求识别出该场景中的文本，比如街拍，标语，广告牌等等。场景文本检测任务与目标检测任务（Object Detection）有着密不可分的联系，两者的目标基本一致。目前深度学习在Object Detection任务上已经大展身手，Faster-RCNN，SSD等经典检测网络已经能够将场景中大部分的object检测出来，然而直接将Object Dection任务的框架应用在场景文本检测上效果并不理想，其中几个原因包括：

- 文本的形状多为ratio较大的矩形，长条形，传统的检测网络大多数是针对比例适中的矩形框进行检测，因此，在感受野上和Anchor的设计上有别于传统检测网络
- 场景文本不同于传统OCR任务，其场景多位于室外，文本形状复杂多变，而大多数检测网络大多使用水平矩形框就能覆盖（人，交通工具，动物）。当文本倾斜或弯曲或透视（仿射变换）的时候，水平预测框很难将文本大部分覆盖

### 解决方法

目前针对于场景文本分类，目前已经有一些文章解决，具体可以参考[这篇博客](https://zhuanlan.zhihu.com/p/38655369?utm_source=qq&utm_medium=social)，其中比较出名的方法有：

- CPTN：只允许**水平检测框**
- RRPN：允许**带角度的矩形框**覆盖
- EAST：允许**带角度矩形框**或任意**四边形覆盖**
- TextBox：**水平矩形框**
- TextBox++：**旋转矩形框**

然而，无论是使用水平检测框还是使用带角度的矩形检测框，或是任意四边形，在更复杂的情况下都无法对文本进行较好的覆盖，我们看下面这张图，（a）使用水平矩形框（b）使用旋转矩形框（c）使用任意四边形，很容易看得出，在文字弯曲程度较大是，abc三者只能在一定程度上覆盖了目标文字，我我们更希望有一种方法能像（d）一样，能够用一条类似面条一样的结构把文字精准覆盖。

![image-20190106142351222](http://princepicbed.oss-cn-beijing.aliyuncs.com/blog_201901061654440574.png)

因此本文提出了一种基于圆盘覆盖的方法——TextSnake，来解决以上的问题。


### TextSnake方法

首先我们定义一下TextSnake的基本单元，圆盘（Disk），我们将任意弯曲形状（假设文字与文字之间，同一个文字的不同部分之间不会重叠）的文本描述为若干不同大小的圆盘序列，每个圆盘拥有几个属性：

- 中心的(x, y)坐标（center）
- 圆盘半径（radius）
- 圆盘方向（orientation）

所有圆盘序列的中心点连接可以构成一条中心线（text center line, TCL），具体如下图，绿色为中心线，黄色区域为文字区域（text region, TR），蓝色为每个圆盘区域，这里要注意的是，圆盘的方向$\theta$表示的是该圆盘的中心点与下一个圆盘中心点连接线和水平方向的角度。

![image-20190106142422753](http://princepicbed.oss-cn-beijing.aliyuncs.com/blog_201901061654450787.png)

有了这个数据结构，我们就可以通过构建深度学习模型来学习圆盘的各种集合属性，最后用于推测文字的位置和形状了，下图是TextSnake整个模型的pipeline，接下来会逐一解释：

![image-20190106152137586](http://princepicbed.oss-cn-beijing.aliyuncs.com/blog_201901061654450715.png)

#### FCN部分

下面我们定义TextSnake的FCN网络结构，如下图所示，主干网络（蓝色）结构为一个VGG-16/19网络，VGG分为5次下采样，最后将原图缩小为1/32大小，并开始上采样，融合下采样时的特征图：

![image-20190106142507757](http://princepicbed.oss-cn-beijing.aliyuncs.com/blog_201901061654460832.png)

其中中括号为concatenate操作，用这种方式将特征图再恢复到原图大小（熟悉FCN的读者应该很容易理解此操作），最后用一个1x1的卷积操作把特征图映射为[H, W, 7]的7通道特征图，其中H和W是原图的高和宽，具体的卷积结构可以用下面的公式表示：

$$
h_1 = f_5
$$

$$
h_i = conv_{3 \times 3}(conv_{1 \times 1}[f_{6 - i}; UpSampleing_{\times 2}]), for\  i =2, 3, 4, 5
$$

$$
h_{final} = UpSampling_{\times2}(h_5)
$$

$$
P = conv_{1 \times 1}(conv_{3 \times 3}(h_{final}))
$$


这7个通道的作用我们在下面阐述。

#### 特征图预测

前面说到每个点预测出一个7通道的特征图，我们先来解释前4个通道，其中2个通道预测TR的部分，2个通道用于预测TCL。文章不仅预测TCL的区域，还预测了TR的区域（GT中TCL的区域被完全包含在了TR中），最后将预测得到的TCL的mask和TR的mask做一个element-wise product得到了最终的TCL区域，作为圆盘的中心点，这么做的原因有几个：

- 基于我们前面的假设，标注中的每段文字的TR部分是不重叠的（即使有重叠面积也较小），因此可以使用TR部分的heatmap来区别不同文字
- 我们生成的GT中，TCL部分被完全包括在TR中，两者相乘能够很好的将非TCL部分的噪声点去除，如主流程图中TCL中的那些噪声部分

接下来说后面3个通道：分别是圆盘半径（radii）和圆盘的方向（$\theta$），其中$\theta$只使用cos或者sin是无法唯一确定的，因此这里文章使用两个通道分别预测cos和sin。

#### Inference

假设我们已经有一个能够很好预测TCL，TR，radii，sin和cos的FCN网络了，接下来就是如何将这这几个特征转换为最终的文字预测。首先和上面说的一样，TCL和TR的mask相乘得到去噪的TCL mask。对于TCL，其实在预测的热力图中不是一条线而是一个狭长的区域，我们需要使用角度信息找到TCL的主干点，首先把TCL的mask拆分为不相连的区域，每个独立区域都是一个文字，对于每个TCL区域具体步骤如下：

![image-20190106142541932](http://princepicbed.oss-cn-beijing.aliyuncs.com/blog_201901061654460871.png)

以TCL区域的任意一个点作为起始点，做Centralizing操作（Act(a)操作），具体步骤如下

- 初始化：在起始点利用sin和cos画出改点位置的切线（虚线）和法线（实线）

- `Act(a) Centralizing`: 法线部分与TCL区域第一次相交的两个点取中点作为Centraling点

- `Act(b) Striding`: 选定Centraling点，想切线方向迈一步，步长为$\pm \frac12r \times \cos{\theta}$
- `Act(c) Sliding`: 从初始点的左右分别都生成到末端，每次在中心点画一个圆，最后所有被圆覆盖的地方我们就作为Text预测出来

此外文章还预设了启发式的过滤规则：

- 如果TCL的mask面积不到半径的0.2倍，剔除
- 如果最终预测出来的Text面积与TR的交集不到TR的一半，剔除

有了以上步骤，就可以将sin，cos，TCL，TR，radii转换成最终的文本预测

#### 训练数据生成

上面的步骤并不困难，如何生成正确的TR区域和TCL区域在实现起来其实才是非常让人头疼的地方，这里我们介绍用于训练的TCL是如何生成的，如下图：

![image-20190106142607399](http://princepicbed.oss-cn-beijing.aliyuncs.com/blog_201901061654460147.png)

1. **确定底边**

我们假设每个文本标注都是蛇形的（Snake-shaped），什么叫蛇形呢？文章中的解释是，有两个底边（bottom）和两条长边（sideline），并且两条底边距离较远，两条长边接近平行，在这种假设下，我们下面的方法才能够适用。我们寻找底边的特点：如(a)图所示，HA这条边是底边，他的特点是，**他前一条边（GH）与后一条边（AB）的夹角是180度**，有了这个信息，用公式表示就是$\cos \langle \vec{GH}, \vec{AB} \rangle \approx -1 $，我们就可以定义一条边是底边的度量指标：

$$
M(e_{i, i+1}) = \cos \langle e_{i+1,i+2} , e_{i−1,i} \rangle
$$

如果该指标接近-1，那么他就是底边。

2. **分割长边**

确定底边以后，他们收尾连接就得到了对于的长边，将两个长边分为相等的小段，把每个等分点相连，对应点连接后取中点，这个中点就是TCL的所在点，而连接线的长度就是圆盘的直径。两点之间连线的角度就是方向

3. **生成圆盘和TCL**

为了让TCL能够完全包括在TR中，作者还将TCL的左右两端分别向内缩了1/2r的长度，然后将TCL向外扩展了1/5r大小，就是(c)图中的浅红色区域了

#### 损失函数

损失函数包括两个部分，分类loss和回归loss:

$$
L = L_{cls} + L_{reg}
$$

$$
L_{cls} = \lambda_1 L_{tr} + \lambda_2 L_{tcl}
$$

$$
L_{reg} = \lambda_3 L_{r} + \lambda_4 L_{sin} + \lambda_5 L_{cos}
$$

分类loss用于TR和TCL的分类，使用的是普通的Cross Entropy，而回归loss都是用了SmoothL1Loss：

![image-20190106164256149](http://princepicbed.oss-cn-beijing.aliyuncs.com/blog_201901061654460624.png)

### 实验

到此为止TextSnake的方法部分已经全部结束，接下来就是实验部分：作者主要选择了**TotalText**和**CTW1500**这两个含有较多曲线形状文字的数据集，所有的实验均在人工合成数据集SynthText上训练一个epoch，然后在其他数据集上finetune得到：

![image-20190106164437269](http://princepicbed.oss-cn-beijing.aliyuncs.com/blog_201901061654470322.png)

在这两个数据集上，TextSnake的效果都比其他方法好了非常多，最主要的原因是TextSnake的数据描述更加的灵活。

在传统的旋转矩形/任意四边形上的效果也不属于state-of-the-art方法：

![image-20190106164750982](http://princepicbed.oss-cn-beijing.aliyuncs.com/blog_201901061654470688.png)

### 代码复现(非官方)

私货：本人基于PyTorch实现[princewang1994/TextSnake.pytorch](https://github.com/princewang1994/TextSnake.pytorch)，目前还不算完善，只支持Total-Text数据集，还有许多待跟进，欢迎star~

在实现中比较深的感触就是，几何操作非常繁琐，TotalText数据集并不像想象的那么干净，一些特例的情况在生成训练标签的时候遇到了很多困难。例如bottom边由多个点组成，而不是两个点，bottom边不止有两个等等等等。。。处理这些异常还不是很鲁棒，将就着看吧😂

### 总结

TextSnake是一种基于圆盘覆盖的文字结构表示，其特点是相对于现有的水平预测框和四边形预测，更加灵活，适用于较为复杂的场景文本检测。作者基于TextSnake的数据结构设计了一种深度学习结构用于预测TextSnake表示，个人感觉这篇文章读起来非常有意思，对我目前的工作也有一定启发作用。