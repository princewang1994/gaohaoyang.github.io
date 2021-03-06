---
layout: post
title:  "Object Detection via Region-based Fully Convolutional Networks"
date:   2018-07-13
categories: DeepLearning
tags: DeepLearning ComputerVision
mathjax: true
author: Prince
---

* content
{:toc}




在Detection任务中现在主要分为1-stage和2-stage两种方法，其中典型的方法分别是Faster-RCNN和SSD，两种方法各有优缺点，对于Faster-RCNN这一族的主要缺点就是比较慢，原因在前几篇文章中都提到了，Faster-RCNN使用了RPN来加速Selective Search的过程，并且使网络实现了End to End。R-FCN重点解决了FasterRCNN头（header）比较中的问题，利用本文中提出的PS-ROIPooling，使RPN提出的每一个ROI，都在Backbone网络的顶层特征图上提取一次，而不再次经过特征提取，从而加速的Faster RCNN的训练和测试速度。

## Faster-RCNN存在的问题 

Faster RCNN中，即使加入了RPN，Faster-RCNN还是比one stage方法慢原因是什么呢？分析原因如下：

- 重复计算：

  Faster RCNN中，RPN将会生成多个ROI区域，通过ROI Pooling将这些ROI区域逐一输入Fast-RCNN中进行分类，分类使用的是全连接层，由于RPN提出的ROI数量通常比较多（300以上），会有大量重叠的roi，这些roi每个都得经过ROIPooling之后传入特征提取网络（Fast RCNN）导致了大量冗余计算。

- 特征提取的Header比较重：

  - 使用VGG作为backbone时，Faster-RCNN使用VGG的全连接层(4096 -> 4096 -> 21)来进行分类
  - 使用ResNet作为backbone时，Faster-RCNN使用ResNet的C4层特征图传入RPN，而C5层则作为分类器放在ROIPooling后面（1024->2048）

那么有没有办法在ROI分类的时候使计算能够共享，这样只需要将原图通过一系列卷积成一个大特征图，然后直接在这个生成的大特征图上提取ROI区域而不使用全连接层呢？（这种想法是不是很类似于RCNN演变为Fast-RCNN的方法？）这就促使研究者们开始研究“全卷积化”的Faster-RCNN，也就是RFCN。

## R-FCN 与 Faster-RCNN的关系与区别

R-FCN最重要的特点是，结构中不存在全连接层，而使用卷积层让最后分类的计算也能够共享计算，提高分类的速度。结构上来看，基本是与Faster RCNN一致的，所以我们先回顾一下Faster-RCNN的结构：

![](media/15063381254862.jpg)

首先是RPN网络用来提取ROI区域，目的是对找出包含Object可能性较高的候选框，然后在共享计算的特征图上把这些ROI抠出来，使用SPP的方法（也就是Faster RCNN所说的ROI Pooling）映射为固定长度的特征向量再通过FC层最后分类。

R-FCN取消了ROI Pooling，取而代之的使用**Position-Sensitive ROI Pooling(ps-roi-pooing)**来使所有的ROI能够共享计算，而不对每个ROI都使用一次完全的Fully Connected Layer。

## R-FCN的整体结构

R-FCN的整体结构如下：

![](media/15063403082127.jpg)

由于Faster RCNN的特征提取网络中大量的卷积和下采样，所以平移不变性的问题在R-FCN中尤为凸显，最直观的感受就是特征图大小逐渐缩小，而给出图像中足够的位置信息。相比于Faster-RCNN利用ROI Pooling来截断平移不变性，全卷积网络不停的卷积会出现很大程度上的平移不变性，从而导致Localization的准确率降低。

### 解决方法

1. 增大特征图

一种方法是使用Atrous Convolution增大特征图，在本文中，使用ResNet作为Backbone的网络，在conv5中把stride=2改成了stride=1，然后把conv5的每个卷积层的dilation都设成2，这样有效增大了ResNet的特征图大小，并且保持了conv4阶段的特征图分辨率，并拥有了增大了感受野。


2. Position Sensitive ROI Pooling 

简单来说，PS ROI Pooling就是通过向Pooling中强制添加位置信息，比如下面这个例子：

![](media/15063399348960.jpg)

假设这是一个婴儿的类别，我们把RPN得出的ROI分成3x3的bins，对左上角的bin判断“这是婴儿的左上部分吗”，或“这是婴儿的右上部分吗”，从而得到9个分数的特征图，通过分数的投票来判断这个框是否是一个婴儿，可以预见的是，如果这个框是比较准确的，那么9个格子中的值应该都是比较大的，文中使用了Average Pooling来进行投票。

### PS-ROIPooling的实现方法

接下来描述一下ps roi pooling的具体运行方式：

首先由于共享计算，RPN得到的features会被复用到后面的分类中，这个特征提取器文中是使用的resnet101，假设这个特征图的大小是[1, H, W, 1024]，本来ROI Pooling直接就在这上面截取ROI来分类了，而现在我们再后面再接上ResNet的C5（1024->2048, dilation=2），继续在上面接一个1x1的卷积(2048->1024)，不改变特征图大小，把通道缩减为k x k x (C+1)，这里的k指的是bin的数量，也就是上面说的3，类别假设是20。

![](media/15063386443591.jpg)

现在特征图大小是[1, H, W, 3x3x21]了，如下图，相同颜色的特征图对应着结果中对应颜色的位置，比如黄色的特征图对应结果中的左上角。特征图被分为9x21个，每21个特征图代表这个位置（比如左上角）的21个分类的分数，**比如[0~21]的特征图中每个点(x, y)表示原图上这个点是对应物体“左上部分“的概率，[22~41]的特征图中每个点(x, y)表示原图上这个点是对应物体“上部分“的概率**，按下图的方法，把对应的特征图的部分做Average Pooling变为9个[1x1x21]的特征图后把这些特征图拼成[9x9x21]，这个新的特征图的每个位置表示了该ROI候选框的9个子块区域是21个物体某一部分的概率。可以形象理解为下图：

![PSROIPooing](media/PSROIPooing.png)
用文中的公式表示这个过程就是下面这样：

$$
    r_c(i, j | \Theta) = \sum_{(x,y) \in bin(i,j)} z_{i,j,c} (x+x_0,y+y_0 | \Theta) / n.
$$

这里的$r_c$是输出的最后结果，$z_{i, j, c}$表示某一个特征图比如$z_{1, 2, 1}$表示的是编号为1的物体的“中右部分”的概率，也就是第(1\*3 + 2) * 21 + 1 = 106层特征图，把这层特征图对应ROI的中右部分做一个Average Pooling得到的一个值就是$r_c(i, j)$的值了，最后把得到的[300x3x3x21]（假设有300个ROI）的特征图。

然后进入Voting环节，把上面得到的特征图做Average Pooling得到[300x21]，判断每个ROI是哪个类别。

别忘了还有老朋友bounding box回归，这里是在与ps roi pooling平行的卷积出

到此为止R-FCN介绍结束，这里有一个[PyTorch版的实现](https://github.com/PureDiors/pytorch_RFCN)可以参考。

## PSROIPooling的cuda实现

该cuda代码来自R-FCN的[caffe版RFCN](https://github.com/pytorch/pytorch/blob/master/modules/detectron/ps_roi_pool_op.cu)，以下我们假设在做一个3x3的PSROIPooling，即`pooled_height == pool_width == 3 `，class数量为21，图中的变量已经在下图中标出：

![](http://oodo7tmt3.bkt.clouddn.com/blog_20180713200808.png)



```c++
template <typename T> //这里先定义一个T目的是为了更好的泛化，如float32，float64

// 前向函数参数定义
__global__ void PSRoIPoolForward(
    const int nthreads, // 多少个线程来执行
    const T* bottom_data, // 输入特征图，shape为(batch_size, pool_w*pool_h*n_class, Height, Width)
    const T spatial_scale, // 输入特征图与原图的缩放比例，比如output_stride=16，那么spatial_scale=1/16.0
    const int channels, // 输入特征图通道数量，pool_w*pool_h*n_class
    const int height, // 输入特征图的高
    const int width, // 输入特征图的宽
    const int pooled_height, // pool size，文章中的3x3
    const int pooled_width, // pool size，文章中的3x3
    const T* bottom_rois, // rpn提出来的框，shape为n x 5, [roi_batch_ind, w_start, h_start, w_end, h_end]
    const int output_dim, // 输出通道数，即n_class
    const int group_size, // 分组数
    T* top_data, // 输出矩阵指针，shape为[n, ctop, ph, pw]
    int* mapping_channel // 为了反向传播记录通道映射
) {
    
  //这里是开始一个CUDA一重循环  
    
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
      
    // 定义输出大小为(n, ctop, ph, pw)，使用index变量开始一重循环，可以看做有n*ctop*ph*pw个线程同时在完成这个任务（其实没那么多）
    // 先把index解析到(n, ctop, ph, pw)
    // n: 当前box编号
    // ctop: 当前输出的class标签编号
    // pw: 当前在计算的窗口的在3x3中的横坐标
    // ph: 当前在计算的窗口在3x3中的纵坐标
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int ctop = (index / pooled_width / pooled_height) % output_dim;
    int n = index / pooled_width / pooled_height / output_dim;

    // [start, end) interval for spatial sampling
    // 先把roi指针移到第n个box的开头
    const T* offset_bottom_rois = bottom_rois + n * 5;
        
    // 解析box的5个位置，分别是[roi_batch_ind, w_start, h_start, w_end, h_end]，roi_batch_ind表示这个roi属于batch中的第几张图片
    int roi_batch_ind = offset_bottom_rois[0];
    T roi_start_w = static_cast<T>(
      roundf(offset_bottom_rois[1])) * spatial_scale;
    T roi_start_h = static_cast<T>(
      roundf(offset_bottom_rois[2])) * spatial_scale;
    T roi_end_w = static_cast<T>(
      roundf(offset_bottom_rois[3]) + 1.) * spatial_scale;
    T roi_end_h = static_cast<T>(
      roundf(offset_bottom_rois[4]) + 1.) * spatial_scale;
	
    // 为了防止roi太小，这里做了一些约束，使最小也要为0.1
    T roi_width = max(roi_end_w - roi_start_w, 0.1);  // avoid 0
    T roi_height = max(roi_end_h - roi_start_h, 0.1);

    // Compute w and h at bottom
    // 计算每个网格bin的长宽，即roi的大小除以pooling size
    T bin_size_h = roi_height / static_cast<T>(pooled_height);
    T bin_size_w = roi_width / static_cast<T>(pooled_width);
	
    // 利用bin_size和ph, pw来计算当前的当前网格在特征图上的的起止位置hstart, wstart, hend, wend
    int hstart = floor(
      static_cast<T>(ph) * bin_size_h + roi_start_h);
    int wstart = floor(
      static_cast<T>(pw)* bin_size_w + roi_start_w);
    int hend = ceil(
      static_cast<T>(ph + 1) * bin_size_h + roi_start_h);
    int wend = ceil(
      static_cast<T>(pw + 1) * bin_size_w + roi_start_w);
        
    // 为了防止ROI超过特征图的大小，需要把hstart, wstart, hend, wend做一下clip
    hstart = min(max(hstart, 0), height);
    hend = min(max(hend, 0), height);
    wstart = min(max(wstart, 0),width);
    wend = min(max(wend, 0), width);
    bool is_empty = (hend <= hstart) || (wend <= wstart);
	
    ///////////////////////////////////////////////////////////////////////////////
    ///////到此为止计算目标的空间坐标已经确定，下面开始计算我需要在哪个通道上来计算avg/////////
    ///////////////////////////////////////////////////////////////////////////////
    
   	// c = (ctop * group_size * group_size) + (ph * group_size + pw)
    // ctop * group_size * group_size: 跳过前ctop组3x3通道
    // ph * group_size + pw: 具体坐标偏移
        
    int gw = pw;
    int gh = ph;
    int c = (ctop * group_size + gh) * group_size + gw;
	
    // 移动特征图的指针
    // (roi_batch_ind * channels * height * width) + (c * height * width)
    // (roi_batch_ind * channels * height * width): 跳过前roi_batch_ind张特征图
    // (c * height * width): 当前特征图上的偏移
    const T* offset_bottom_data = bottom_data + (roi_batch_ind * channels + c) * height * width;
        
    // 所有的计算目标都已经使用hstart, hend, wstart, wend, offset_bottom_data定位好了
    // 现在开始计算输入特征图上矩形[hstart, hend, wstart, wend]，通道编号为c的所有值的平均值，两重循环
    T out_sum = 0;
    for (int h = hstart; h < hend; ++h){
     for (int w = wstart; w < wend; ++w){
       int bottom_index = h*width + w;
       out_sum += offset_bottom_data[bottom_index];
     }
    }
	
    // 统计一共计算了多少个值的平均值，如果没有值，直接返回0(为了防止除零错误)
    T bin_area = (hend - hstart) * (wend - wstart);
    top_data[index] = is_empty ? 0. : out_sum / bin_area;
        
    // 这里是为了反向传播的时候使用
    mapping_channel[index] = c;
  }
}
```

**需要注意的一点是**，在这份cuda实现里面，通道的排列顺序和图中不同，原图为了更好理解，排列顺序为：（21x3x3）而cuda中为了实现简单，保证每一个类别的所有位置都是连续的，所以使用了3x3x21的形式。

![PS-roipooling](/Users/prince/Downloads/PS-roipooling.png)

因此，在寻找当前的ctop对应的是原图中的哪个通道的时候，先要跳过ctop个3x3通道，然后在跳过ph*pool_width个通道，再偏移pw个通道，所以就有了这样的代码：这里的group size一般是等于pool size的，在这里也就是3.

```c++
c = (ctop * group_size + gh) * group_size + gw;
```

## 总结

RFCN是在Faster-RCNN的基础上，把ROI Pooling使用共享计算的FCN来代替，产生每个像素是对应物体某一部分的概率，从而加速计算的方法。在准确率方面会比Faster-RCNN稍微差一点，但是速度提升了非常多，这也是归功于全卷积网络的使用和PSROIPooling的提出。




