## 非极大值抑制算法与源码分析

非极大值抑制是物体检测众多方法（包括但不限于深度学习方法）中的一个很重要的步骤，通常在模型训练结束以后，使用模型对输入图片进行预测，会预测出大量的框，这些框中很大一部分是重复的，但在置信度上有所不同，非极大值抑制的主要任务就是将重复的方框去掉，同一个物体只留下置信度最高的框。

### 非极大值抑制算法流程简介

本站另外一篇博客http://blog.prince2015.club/2018/07/22/FasterRCNN/ 中给出了非极大值抑制的详细过程，这里简单回顾一下NMS的流程：

假设所有预测框的集合为S，算法返回结果集合S’初始化为空集，具体算法如下：

1. 将S中所有框的置信度按照降序排列，选中最高分框B，并把该框从S中删除
2. 遍历S中剩余的框，如果和B的IoU大于一定阈值t，将其从S中删除
3. 从S未处理的框中继续选一个得分最高的加入到结果S‘中，重复上述过程1, 2。直至S集合空，此时S’即为所求结果。
算法整体看是比较简单而直接的，然而在实际执行过程中却不是这样，由于检测模型预测出来的框数量非常多(通常在几百个到一千个)，那么这个算法最坏情况下（当所有框都互相不相交的时候）的复杂度是$O(n^2)$，这个时候使用Python作为主循环的速度可想而知。因此，在以Python实现的Faster RCNN或SSD中，如果使用了NMS，一般采用Cython加速版或是CUDA版，接下来我们就分别分析这两个版本的NMS：

### Cython加速版NMS

本节需要一些使用C语言加速Python，并编译生成动态库的知识，如果对此方面知识不了解的读者，可以阅读这个[Github Repo](https://github.com/princewang1994/python_c_extension)

接下来言归正传，我们主要分析了一个比较高star的Pytorch版Faster-RCNN中的Cython NMS，源码地址在[这里](https://github.com/longcw/faster_rcnn_pytorch/blob/master/faster_rcnn/nms/cpu_nms.pyx)

代码写的很短小，所以直接上核心函数：

```python
def cpu_nms(np.ndarray[np.float32_t, ndim=2] dets, np.float thresh):
    """
    Args:
    	dets: numpy_array, 待处理的box集合，即S，shape为[N, 5]，每一行代表[x1, y1, x2, y2, confidence]
    	thresh: float, 取值范围[0, 1]，ms最低阈值t
    """
    cdef np.ndarray[np.float32_t, ndim=1] x1 = dets[:, 0]
    cdef np.ndarray[np.float32_t, ndim=1] y1 = dets[:, 1]
    cdef np.ndarray[np.float32_t, ndim=1] x2 = dets[:, 2]
    cdef np.ndarray[np.float32_t, ndim=1] y2 = dets[:, 3]
    cdef np.ndarray[np.float32_t, ndim=1] scores = dets[:, 4]
	
    # compute area of all boxes
    cdef np.ndarray[np.float32_t, ndim=1] areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    cdef np.ndarray[np.int_t, ndim=1] order = scores.argsort()[::-1]
	
    # N
    cdef int ndets = dets.shape[0]
    cdef np.ndarray[np.int_t, ndim=1] suppressed = \
            np.zeros((ndets), dtype=np.int)

    # nominal indices
    cdef int _i, _j
    # sorted indices
    cdef int i, j
    # temp variables for box i's (the box currently under consideration)
    cdef np.float32_t ix1, iy1, ix2, iy2, iarea
    # variables for computing overlap with box j (lower scoring box)
    cdef np.float32_t xx1, yy1, xx2, yy2
    cdef np.float32_t w, h
    cdef np.float32_t inter, ovr

    keep = []
    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]
        iarea = areas[i]
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            xx1 = max(ix1, x1[j])
            yy1 = max(iy1, y1[j])
            xx2 = min(ix2, x2[j])
            yy2 = min(iy2, y2[j])
            w = max(0.0, xx2 - xx1 + 1)
            h = max(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (iarea + areas[j] - inter)
            if ovr >= thresh:
                suppressed[j] = 1

    return keep
```

我们分析一下这个代码，主要分为两部分，申明部分和核心循环部分，申明部分包括了输入框的数量以及阈值，并构造了suppressed数组作为从S集合移出的标记，长度和输入框的大小一致。keep变量作为返回变量，记录了最终S'内的框编号。

主循环部分，\_i和\_j用于for数量循环，实际上是的index变量是使用i和j。首先使用一个i循环按照置信度从大到小遍历每一个框

```python
for _i in range(ndets):
    i = order[_i]
```

如果如果第i个框已经被移出(if suppressed[i] == 1)，直接跳过循环。否则把该框加入结果集keep

```python
if suppressed[i] == 1:
    continue
keep.append(i)
```

接下来把和框i的IOU大于阈值t的所有框都移出：

```python
for _j in range(_i + 1, ndets): # i前面的框要么被移入S'了，要么被移出S了，所以不用判断
    j = order[_j]
    if suppressed[j] == 1: # 只判断还在S里面的框
        continue
    # 下面开始求IOU
    xx1 = max(ix1, x1[j])
    yy1 = max(iy1, y1[j])
    xx2 = min(ix2, x2[j])
    yy2 = min(iy2, y2[j])
    w = max(0.0, xx2 - xx1 + 1)
    h = max(0.0, yy2 - yy1 + 1)
    inter = w * h
    ovr = inter / (iarea + areas[j] - inter)
    # 如果IOU大于thresh，移出
    if ovr >= thresh:
        suppressed[j] = 1
```

至此NMS的Cython版分析完毕，其实整体流程就和上面描述的一样，没有非常多值得关注的亮点，但是其Cython的加速还是比较可以看的，接下来我们使用`cython -a`来看一下这个文件中被加速的部分：

![](http://oodo7tmt3.bkt.clouddn.com/blog_20180722220658.png)

可以看到，前面的申明部分大部分是没有加速，而后面的主循环部分和Python的加速少一些，57，58行的比较亮，说明还是有一些地方可以优化的，比如把keep完全变成01数组，有兴趣的读者可以尝试做一些优化。

## CUDA加速版NMS

按照上面的算法，似乎是一个单线程算法，因为需要主循环来做一些判断，每一个过程都是强依赖的，那么如何使用CUDA加速呢，其实有一个很重要的地方是可以加速的，就是计算每两个框之间的IOU和并判断IOU是否大于阈值（0/1）。假设两两之间的框的判断结果事先已经存在一个表中，可以大大加速算法。

### CUDA加速NMS原理

CUDA的加速的原理是使用多个线程计算出一个逻辑上NxN的矩阵，而矩阵的每一个位置都是一个0/1的bool，但是我们都知道GPU读取global memory的速度是很慢的，所以常常使用share memory进行加速读取，使同一个block的线程可以共享存储，这就需要对任务进行分块，在CUDA版加速算法中，就是这么做的，我们首先假设输入框的数量为N，最后需要得到一个NxN的矩阵，我们就把NxN的矩阵分为若干个长度为k的block，每个block中使用k个线程计算k x k个0/1值，每个线程计算k个bool，而在每一个block中，存储是共享的。具体如下图：

![nms_cuda](/Users/prince/Downloads/nms_cuda.png)

接下来结合代码来说明NMS的过程，下面的例子我们假设N=105，每个k=25，计算一下，$\lceil 100/5 \rceil = 5$，那么将整个矩阵分为5x5个block，每个block求一个25x25的小矩阵，最后全部拼在一起就成了最后的矩阵。不过注意到这里为了节省内存和使用移位操作的加速，代码中实际上并没有真正算出NxN的矩阵，而是对每个block都求k个整数，每个数都是一个25位二进制掩码，所以说最后**实际求出的是一个$N \times \lceil N/k \rceil$的矩阵**，知道了这个，下面的代码就比较好理解了：

**注意这里的输入boxes_host已经按照最后一维(预测框置信度)从大到小排列好了**，接下来我们使用注释来解释这段源码([地址](https://github.com/jwyang/faster-rcnn.pytorch/blob/master/lib/model/nms/nms_kernel.cu))：


```c
// ------------------------------------------------------------------
// Faster R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Shaoqing Ren
// https://github.com/jwyang/faster-rcnn.pytorch/blob/master/lib/model/nms/nms_kernel.cu
// ------------------------------------------------------------------

#include <stdbool.h>
#include <stdio.h>
#include <vector>
#include <iostream>
#include "nms_cuda_kernel.h"

#define CUDA_WARN(XXX) \
    do { if (XXX != cudaSuccess) std::cout << "CUDA Error: " << \
        cudaGetErrorString(XXX) << ", at line " << __LINE__ \
<< std::endl; cudaDeviceSynchronize(); } while (0)

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      std::cout << cudaGetErrorString(error) << std::endl; \
    } \
  } while (0)

#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0)) // 相除后向上取整
int const threadsPerBlock = sizeof(unsigned long long) * 8; // 分块数量，例子中为25

// 定义计算IOU函数
__device__ inline float devIoU(float const * const a, float const * const b) {
  float left = max(a[0], b[0]), right = min(a[2], b[2]);
  float top = max(a[1], b[1]), bottom = min(a[3], b[3]);
  float width = max(right - left + 1, 0.f), height = max(bottom - top + 1, 0.f);
  float interS = width * height;
  float Sa = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
  float Sb = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
  return interS / (Sa + Sb - interS); // IOU = (A ∩ B) / (A ∪ B)
}

// nms核心函数，计算一个num_boxes * num_boxes的矩阵，矩阵的每个值(i, j)都表示box_i和box_j的IOU是否大于阈值nms_overlap_thresh
__global__ void nms_kernel(int n_boxes, float nms_overlap_thresh,
                           float *dev_boxes, unsigned long long *dev_mask) {
    
  const int row_start = blockIdx.y; //确定当前block的横坐标
  const int col_start = blockIdx.x; //确定当前block的纵坐标

  const int row_size =
        min(n_boxes - row_start * threadsPerBlock, threadsPerBlock); //求当前block的行长度，如果最后一个block不够除，则取余下的，比如ceil(105/25) = 5，105 = 4 * 25 + 5最后一块高为5，此时row_size=5，其余的row_size = 25
  const int col_size =
        min(n_boxes - col_start * threadsPerBlock, threadsPerBlock); //求当前block的列长度

  // 共享内存，加速数据读取，同一个block有共享内存，所以先使用共享内存存下当前block全部需要读取的数据(即box的坐标和置信度)然后就不在dev_boxes里面读数据了，而是读share memory里面的数据
  __shared__ float block_boxes[threadsPerBlock * 5];
  if (threadIdx.x < col_size) {
    block_boxes[threadIdx.x * 5 + 0] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 0];
    block_boxes[threadIdx.x * 5 + 1] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 1];
    block_boxes[threadIdx.x * 5 + 2] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 2];
    block_boxes[threadIdx.x * 5 + 3] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 3];
    block_boxes[threadIdx.x * 5 + 4] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 4];
  }
  //为了保证线程安全，必须等所有的线程都把数据存到share memory以后，统一开始线程
  __syncthreads();

  // 这个if判断去掉多余的thread，保证余下的块可以被正确执行
  // 每个block里面有row_size个线程
  // 每个线程i，for一个col_size的循环，计算该block里面第i个box和该block中每个列box的IOU
  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x; //求当前行box的编号
    const float *cur_box = dev_boxes + cur_box_idx * 5; //box根据编号偏移
    int i = 0;
    unsigned long long t = 0;
    int start = 0;
    for (i = start; i < col_size; i++) { //主循环，求该box和所有列box的IOU，如果满足条件，则使用一个mask把该位置1
      if (devIoU(cur_box, block_boxes + i * 5) > nms_overlap_thresh) {
        t |= 1ULL << i; //掩码操作
      }
    }
    //最后实际生成一个num_boxes * col_blocks的矩阵
    const int col_blocks = DIVUP(n_boxes, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}

//入口函数
void nms_cuda_compute(
    int* keep_out, // 最后输出数组，存box编号
    int *num_out, // 最后输出的keep_out中有效数量
    float* boxes_host, // 待处理box集合，shape[boxes_num, 5]
    int boxes_num, // 输入box数量
    int boxes_dim, // 5 = [hstart, wstart, hend, wend, confidence]
    float nms_overlap_thresh // NMS的阈值
) {

  float* boxes_dev = NULL;
  unsigned long long* mask_dev = NULL;

  const int col_blocks = DIVUP(boxes_num, threadsPerBlock); 

  CUDA_CHECK(cudaMalloc(&boxes_dev,
                        boxes_num * boxes_dim * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(boxes_dev,
                        boxes_host,
                        boxes_num * boxes_dim * sizeof(float),
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&mask_dev,
                        boxes_num * col_blocks * sizeof(unsigned long long)));

  // 定义blocks的数量和每个block的线程数
  dim3 blocks(DIVUP(boxes_num, threadsPerBlock),
              DIVUP(boxes_num, threadsPerBlock));
  dim3 threads(threadsPerBlock);

  // 调用kernel，最后在mask_dev中求出每两个框的IoU是否超过阈值t
  nms_kernel<<<blocks, threads>>>(boxes_num,
                                  nms_overlap_thresh,
                                  boxes_dev,
                                  mask_dev);

  std::vector<unsigned long long> mask_host(boxes_num * col_blocks);
  CUDA_CHECK(cudaMemcpy(&mask_host[0],
                        mask_dev,
                        sizeof(unsigned long long) * boxes_num * col_blocks,
                        cudaMemcpyDeviceToHost));

  std::vector<unsigned long long> remv(col_blocks); // 标记是否移出S
  memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks); // 初始是所有框都在S里面，移出标记都置为0

  // 主循环开始
  int* keep_out_cpu = new int[boxes_num];
  int num_to_keep = 0; 
  // for每一个box编号
  for (int i = 0; i < boxes_num; i++) {
    int nblock = i / threadsPerBlock; //求这个box是在那个block里面计算的
    int inblock = i % threadsPerBlock;//求这个box在block的哪个线程计算的
	// 对于每个box，如果他在S中，则加入结果集，并移出S
    // 并把和他的IOU大于阈值的所有box全部移出S
    if (!(remv[nblock] & (1ULL << inblock))) {
      keep_out_cpu[num_to_keep++] = i; //加入结果集操作
      unsigned long long *p = &mask_host[0] + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv[j] |= p[j]; //移出S操作
      }
    }
  }

  // copy keep_out_cpu to keep_out on gpu
  CUDA_WARN(cudaMemcpy(keep_out, keep_out_cpu, boxes_num * sizeof(int),cudaMemcpyHostToDevice));  

  CUDA_WARN(cudaMemcpy(num_out, &num_to_keep, 1 * sizeof(int),cudaMemcpyHostToDevice));  

  // 释放cuda资源和cpu资源
  CUDA_CHECK(cudaFree(boxes_dev)); 
  CUDA_CHECK(cudaFree(mask_dev));
  delete [] keep_out_cpu;
}

```

