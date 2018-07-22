## Dilated Residual Network

### Motivation

- 以往的卷积网络往往把特征图的分辨率降低到很小(1/32)
- 分类网络经常被使用在迁移学习中，作为检测或分割网络的base model

### Solution

以ResNet为例，设计了一种基于ResNet的改进方法，来增加特征图的resolution。

通过Dilated Convolution的方法，可以去掉ResNet**最后两个**Block的stride=2的选项，改成膨胀卷积。从而增加了特征图的大小，并且这种方法使得整个网络的感受野不变。

![](media/15060861951909.jpg)


由于Dilated Convolution在增加分辨率的同时，在特征图中引入了很多类grid的特征，文章通过两种方法来取消格子效应

- `DRN-B-26`: 去掉网络头部的max pooling并使用stride=2的residual block来代替max pooling，然后在网络的尾部加上两个stride=1的residual block来去掉grid效应
- `DRN-C-26`: 在上一条的基础上，由于residual block会把之前带grid的特征图以skip connection的形式加入到当前层，所以去掉了residual connection来去掉grid效应


实验证明，对于检测任务和分割任务，这种结构能够有效的增加提高定位任务的准确率。

### 膨胀卷积的缺点——Griding

Dilated Convolution在DeepLab笔记中已经着重记录，因此在这里不做重点记录。

但是要说一下该方法的问题，我们知道膨胀卷积是带洞的卷积，意味着两个有效的卷积权值之间会有一些空洞来增加感受野，这种结构导致了grid，如下图可以看出来，Input只有一个点，(b)是一个膨胀卷积核，在滑动的过程中，如果是普通的卷积核产生的将会是连续的输出，而膨胀卷积在周围将会产生一些散点，在特征图中可以认为是噪音。

![](media/15060865051214.jpg)


### 基本结构

代码实现: [Dilated Residual Network](https://github.com/fyu/drn)

下图是DRN的结构，可以看到，绿线代表stride=2的普通convolution，最下面的Dilation代表的是膨胀卷积的rate：

![](media/15060863783909.jpg)

仔细分析可以注意到，DRN-A-18相对于ResNet18而言，值改变了两个地方

- 去掉了第5和第6level，也就是通常我们说的block4和block5中的stride=2，把他改为stride为1，提高了4倍的特征图resolution（每去掉一个stride=2会让特征图提高两倍）
- 在第5和第6个level，加上了2和4的dilation来增加感受野，普通的卷积层可以看做是dilation为1的特殊情况，每把dilation增加为两倍，感受野将会变为原来的4倍(当然内存的使用量也将变为原来的4倍)，这个在DeepLab文章中也有提到。

结构变化完后，将会把特征图的大小变为28x28(输入时224x224)，以下是不同深度配置的DRN。

![](media/15060873136878.jpg)

对应的结构中我们看到，DRN有三种结构：

- `DRN-A`: 只是按照上面的方法把模型改造为DRN的形式，所以层数与对应的ResNet完全一致
- `DRN-B`: 在DRN的基础上增加前后增加了residual block而去掉了max pooling，因此层数上比对应的ResNet增加了8层
- `DRN-C`: 在B的基础上去掉了末尾两个block的skip connection

B,C两个模型都是为了**degriding**

### 分类网络

在分类网络中，与ResNet类似，在末尾添加一个Global Average Pooling然后跟上一个fc层，softmax。

### 分割网络

对于以往的网络来说，最后一层的特征图非常小(7x7)，在把网络运用在分割或检测任务的时候需要去掉最后一些层然后提高网络的分辨率，如SSD中就是用了Downsample的第4个Block(C4)作为最后的特征图，然后对其进行反卷积和softmax。网络深度的降低，将会影响网络特征提取的能力。

在把DRN运用在分割任务上的时候，由于resolution已经足够大，我们不需要去掉后面的层，只需要把末尾的fc层改为一个或多个反卷积，在最后跟上一个**conv1x1**即可。

### 检测网络

对于检测网络，文中没有使用比较流行的2stage或者1stage的方法，而是使用了一种简化版：

$$
    g(w, h) =  \{ c | ∀1 \le c' \le C. f(c, w, h) \ge f(c′, w, h)\} 
$$

g(w, h)实际上是位置为(w, h)的特征图上，所有通道的最大值

$$
    B_i = \{ ((w1,h1),(w2,h2)) \ | \ ∀g(w, h) = c_i \ and \  f(w, h, c_i) > t.w1 \le w \le w2 \ and \ h1 \le h \le h2 \}
$$

也就是说，$B_i$其实是满足一个矩形((w1, h1), (w2, h2))中每个点(w, h)，都满足$g(w, h) > t$的最小外接矩形，直观上看其实是先对通道维度做一个max变成一个heat map，然后把heat map上集中值较大的区域框起来，尽量使用最小的矩形把g(w, h)大于阈值的点框起来。

检测的方法在作者的代码中并没有实现，但是可以预见到，使用Faster RCNN的方法，把以ResNet底座的结构改为DRN，应该会有所提升，在这种情况下，直接对feature map使用ROI Pooling，然后接FC即可








