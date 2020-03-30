参考博客：[CTC 原理及实现](https://blog.csdn.net/JackyTintin/article/details/79425866)

- **NOTE: 本篇文章为CSDN博客的注释版，并非原创**
- **NOTE: 本篇文章公式较多，建议复制到typora中观看！**
- **NOTE: 不完全和原博客一致，使用>标记的是自己写的部分**

[TOC]

CTC（ Connectionist Temporal Classification，连接时序分类）是一种用于序列建模的工具，其核心是定义了特殊的**目标函数/优化准则**[1]。


# 1. 算法
这里大体根据 Alex Grave 的开山之作[1]，讨论 CTC 的算法原理，并基于 numpy 从零实现 CTC 的推理及训练算法。

## 1.1 序列问题形式化。
序列问题可以形式化为如下函数：

$$
N_w: (R^m)^T \rightarrow (R^n)^T
$$

其中，序列目标为字符串（词表大小为 $n$），即 $N_w$ 输出为 $n$ 维多项概率分布（e.g. 经过 softmax 处理）。

网络输出为：$y = N_w(x)$，其中，$y_k^t$ $t$ 表示时刻第 $k$ 项的概率。

![](http://princepicbed.oss-cn-beijing.aliyuncs.com/full_collapse_from_audio.svg)
**图1. 序列建模【[src](https://distill.pub/2017/ctc/)】**


虽然并没为限定 $N_w$ 具体形式，下面为假设其了某种神经网络（e.g. RNN）。下面代码示例 toy $N_w$：


```python
import numpy as np

np.random.seed(1111)

T, V = 12, 5 # T表示时间步长，V表示字符集大小（包括blank）
h = 6  # h为隐藏层单元

x = np.random.random([T, h])  # T x h
w = np.random.random([h, V])  # weights, h x V

def softmax(logits):
    max_value = np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits - max_value)
    exp_sum = np.sum(exp, axis=1, keepdims=True)
    dist = exp / exp_sum
    return dist

def toy_nw(x):
    y = np.matmul(x, w)  # T x n 
    y = softmax(y)
    return y

y = toy_nw(x)
print(y)
print(y.sum(1, keepdims=True))
```

    [[ 0.24654511  0.18837589  0.16937668  0.16757465  0.22812766]
     [ 0.25443629  0.14992236  0.22945293  0.17240658  0.19378184]
     [ 0.24134404  0.17179604  0.23572466  0.12994237  0.22119288]
     [ 0.27216255  0.13054313  0.2679252   0.14184499  0.18752413]
     [ 0.32558002  0.13485564  0.25228604  0.09743785  0.18984045]
     [ 0.23855586  0.14800386  0.23100255  0.17158135  0.21085638]
     [ 0.38534786  0.11524603  0.18220093  0.14617864  0.17102655]
     [ 0.21867406  0.18511892  0.21305488  0.16472572  0.21842642]
     [ 0.29856607  0.13646801  0.27196606  0.11562552  0.17737434]
     [ 0.242347    0.14102063  0.21716951  0.2355229   0.16393996]
     [ 0.26597326  0.10009752  0.23362892  0.24560198  0.15469832]
     [ 0.23337289  0.11918746  0.28540761  0.20197928  0.16005275]]
    [[ 1.]
     [ 1.]
     [ 1.]
     [ 1.]
     [ 1.]
     [ 1.]
     [ 1.]
     [ 1.]
     [ 1.]
     [ 1.]
     [ 1.]
     [ 1.]]


## 1.2 align-free 变长映射
上面的形式是输入和输出的一对一的映射。序列学习任务一般而言是多对多的映射关系（如语音识别中，上百帧输出可能仅对应若干音节或字符，并且每个输入和输出之间，也没有清楚的对应关系）。CTC 通过引入一个特殊的 blank 字符（用 % 表示），解决多对一映射问题。

扩展原始词表 $L$ 为 $L^\prime = L \cup \{\text{blank}\}$。对输出字符串，定义操作 $B$：1）合并连续的相同符号；2）去掉 blank 字符。

例如，对于 `aa%bb%%cc`，应用 $B$，则实际上代表的是字符串 "abc"。同理“%a%b%cc%” 也同样代表 "abc"。
$$
B(aa\%bb\%\%cc) = B(\%a\%b\%cc\%) = abc
$$

通过引入blank 及 $B$，可以实现了变长的映射。
$$
L^{\prime T} \rightarrow L^{\le T}
$$

因为这个原因，CTC 只能建模输出长度小于输入长度的序列问题。

## 1.3 似然计算

和大多数有监督学习一样，CTC 使用最大似然标准进行训练。

给定输入 $x$，输出 $l$ 的条件概率为：

$$
p(l|x) =  \sum_{\pi \in B^{-1}(l)} p(\pi|x)
$$

其中，$B^{-1}(l)$ 表示了长度为 $T$ 且示经过 $B$ 结果为 $l$ 字符串的集合。

**CTC 假设输出的概率是（相对于输入）条件独立的**，因此有：
$$
p(\pi|x) = \prod y^t_{\pi_t}, \forall \pi \in L^{\prime T}
$$


然而，直接按上式我们没有办理有效的计算似然值。下面用动态规划解决似然的计算及梯度计算, 涉及前向算法和后向算法。

## 1.4 前向算法

在前向及后向计算中，CTC 需要将输出字符串进行扩展。具体的，$(a_1,\cdots,a_m)$ 每个字符之间及首尾分别插入 blank，即扩展为 $(\%, a_1,\%,a_2, \%,\cdots,\%, a_m,\%)$。下面的 $l$ 为原始字符串，$l^\prime$ 指为扩展后的字符串。

定义
$$
\alpha_t(s) \stackrel{def}{=} \sum_{\pi \in N^T: B(\pi_{1:t}) = l_{1:s}} \prod_{t^\prime=1}^t y^t_{\pi^\prime}
$$

显然有，（印象笔记无法正确加载这个公式转成图片）


$$
\begin{align}
\alpha_1(1) = y_b^1,\\

\alpha_1(2) = y_{l_1}^1,\\
\alpha_1(s) = 0, \forall s > 2\end{align}
$$

![](http://princepicbed.oss-cn-beijing.aliyuncs.com/img_20200330_162319.png)

根据 $\alpha$ 的定义，有如下递归关系：
$$
\alpha_t(s) = \{  \begin{array}{l}
(\alpha_{t-1}(s)+\alpha_{t-1}(s-1)) y^t_{l^\prime_s},\  \  \    if\  l^\prime_s = b \ or\  l_{s-2}^\prime = l_s^{\prime}  \\
(\alpha_{t-1}(s)+\alpha_{t-1}(s-1) + \alpha_{t-1}(s-2)) y^t_{l^\prime_s} \ \ otherwise
\end{array}
$$

> **完整的节点图参考:** https://zhuanlan.zhihu.com/p/43534801
> 节点之间的转换有以下限制：
>
> 1. 转换只能往右下方向，其他方向不允许
> 2. 相同的字符之间起码要有一个空字符（黑色节点的前驱不能是上一个相同字母的黑色节点，比如p前面不能是p）
> 3. 非空字符不能被跳过（白色节点前面的前驱不能是白色节点）
> 4. 起点必须是前两个字符开始
> 5. 重点必须落在结尾两个字符

![](http://princepicbed.oss-cn-beijing.aliyuncs.com/v2-2371b50895b935ecb8f3925a0462e5e5_r.jpg)

### 1.4.1 Case 2
递归公式中 case 2 是一般的情形。如图所示，$t$ 时刻字符为 $s$ 为 blank 时，它可能由于两种情况扩展而来：

1. 重复上一字符，即上个字符也是 a
2. 字符发生转换，即上个字符是非 a 的字符。第二种情况又分为两种情形，
   	- 2.1）上一字符是 blank；
    - 2.2）a 由非 blank 字符直接跳转而来（$B$） 操作中， blank 最终会被去掉，因此 blank 并不是必须的）。

![](http://princepicbed.oss-cn-beijing.aliyuncs.com/cost_regular.svg)
      **图2. 前向算法 Case 2 示例【[src](https://distill.pub/2017/ctc/)】**

### 1.4.2 Case 1
递归公式 case 1 是特殊的情形。
如图所示，$t$ 时刻字符为 $s$ 为 blank （白色节点）时，它只能由于两种情况扩展而来：

1. 重复上一字符，即上个字符也是 blank，
2. 字符发生转换，即上个字符是非 blank 字符。$t$ 时刻字符为 $s$ 为非 blank 时，类似于 case2，但是这时两个相同字符之间的 blank 不能省略（否则无法区分"aa"和"a"），因此，也只有两种跳转情况。

![](http://princepicbed.oss-cn-beijing.aliyuncs.com/cost_no_skip.svg)
**图3. 前向算法 Case 1 【[src](https://distill.pub/2017/ctc/)】**

我们可以利用动态规划计算所有 $\alpha$ 的值，算法时间和空间复杂度为 $O(T * L)$。

似然的计算只涉及乘加运算，因此，CTC 的似然是可导的，可以尝试 tensorflow 或 pytorch 等具有自动求导功能的工具自动进行梯度计算。下面介绍如何手动高效的计算梯度。


```python
def forward(y, labels):
    T, V = y.shape
    L = len(labels)
    alpha = np.zeros([T, L])

    # init
    alpha[0, 0] = y[0, labels[0]]
    alpha[0, 1] = y[0, labels[1]]

    for t in range(1, T):
        for i in range(L):
            s = labels[i]
            
            a = alpha[t - 1, i] 
            if i - 1 >= 0:
                a += alpha[t - 1, i - 1]
            if i - 2 >= 0 and s != 0 and s != labels[i - 2]:
                a += alpha[t - 1, i - 2]
                
            alpha[t, i] = a * y[t, s]
            
    return alpha

labels = [0, 3, 0, 3, 0, 4, 0]  # 0 for blank
alpha = forward(y, labels)
print(alpha)
```

    [[  2.46545113e-01   1.67574654e-01   0.00000000e+00   0.00000000e+00
        0.00000000e+00   0.00000000e+00   0.00000000e+00]
     [  6.27300235e-02   7.13969720e-02   4.26370730e-02   0.00000000e+00
        0.00000000e+00   0.00000000e+00   0.00000000e+00]
     [  1.51395174e-02   1.74287803e-02   2.75214373e-02   5.54036251e-03
        0.00000000e+00   0.00000000e+00   0.00000000e+00]
     [  4.12040964e-03   4.61964998e-03   1.22337658e-02   4.68965079e-03
        1.50787918e-03   1.03895167e-03   0.00000000e+00]
     [  1.34152305e-03   8.51612635e-04   5.48713543e-03   1.64898136e-03
        2.01779193e-03   1.37377693e-03   3.38261905e-04]
     [  3.20028190e-04   3.76301179e-04   1.51214552e-03   1.22442454e-03
        8.74730268e-04   1.06283215e-03   4.08416903e-04]
     [  1.23322177e-04   1.01788478e-04   7.27708889e-04   4.00028082e-04
        8.08904808e-04   5.40783712e-04   5.66942671e-04]
     [  2.69673617e-05   3.70815141e-05   1.81389560e-04   1.85767281e-04
        2.64362267e-04   3.82184328e-04   2.42231029e-04]
     [  8.05153930e-06   7.40568461e-06   6.52280509e-05   4.24527009e-05
        1.34393412e-04   1.47631121e-04   1.86429242e-04]
     [  1.95126637e-06   3.64053019e-06   1.76025677e-05   2.53612828e-05
        4.28581244e-05   5.31947855e-05   8.09585256e-05]
     [  5.18984675e-07   1.37335633e-06   5.65009596e-06   1.05520069e-05
        1.81445380e-05   1.87825719e-05   3.56811933e-05]
     [  1.21116956e-07   3.82213679e-07   1.63908339e-06   3.27248912e-06
        6.69699576e-06   7.59916314e-06   1.27103665e-05]]


最后可以得到似然 $p(l|x) = \alpha_T(|l^\prime|) + \alpha_T(|l^\prime|-1)$。


```python
p = alpha[-1, -1] + alpha[-1, -2]
print(p)
```

```
2.0309529674855637e-05
```

## 1.5 后向计算
类似于前向计算，我们定义后向计算。
首先定义
$$
\beta_t(s)   \stackrel{def}{=} \sum_{\pi \in N^T: B(\pi_{t:T}) = l_{s:|l|}} \prod_{t^\prime=t}^T y^t_{\pi^\prime}
$$

显然，（印象笔记无法正确显示公式，转图片）

$$
\begin{align}
\beta_T(|l^\prime|) = y_b^T,\\

\beta_T(|l^\prime|-1) = y_{l_{|l|}}^T,\\
\beta_T(s) = 0, \forall s < |l^\prime| - 1\end{align}
$$

![](http://princepicbed.oss-cn-beijing.aliyuncs.com/img_20200330_162400.png)

易得如下递归关系：
$$
\beta_t(s) = \{  \begin{array}{l}
(\beta_{t+1}(s)+\beta_{t+1}(s+1)) y^t_{l^\prime_s},\  \  \    if\  l^\prime_s = b \ or\  l_{s+2}^\prime = l_s^{\prime}  \\
(\beta_{t+1}(s)+\beta_{t+1}(s+1) + \beta_{t+1}(s+2)) y^t_{l^\prime_s} 
\end{array}
$$


```python
def backward(y, labels):
    T, V = y.shape
    L = len(labels)
    beta = np.zeros([T, L])

    # init
    beta[-1, -1] = y[-1, labels[-1]]
    beta[-1, -2] = y[-1, labels[-2]]

    for t in range(T - 2, -1, -1):
        for i in range(L):
            s = labels[i]
            
            a = beta[t + 1, i] 
            if i + 1 < L:
                a += beta[t + 1, i + 1]
            if i + 2 < L and s != 0 and s != labels[i + 2]:
                a += beta[t + 1, i + 2]
                
            beta[t, i] = a * y[t, s]
            
    return beta

beta = backward(y, labels)
p = beta[0, 0] + beta[0, 1] # 和alpha得到的结果完全一致
print(beta)
```

    2.0309529674855637e-05
    [[  1.25636660e-05   7.74586366e-06   8.69559539e-06   3.30990037e-06
        2.41325357e-06   4.30516936e-07   1.21116956e-07]
     [  3.00418145e-05   2.09170784e-05   2.53062822e-05   9.96351200e-06
        8.39236521e-06   1.39591874e-06   4.91256769e-07]
     [  7.14014755e-05   4.66705755e-05   7.46535563e-05   2.48066359e-05
        2.77113594e-05   5.27279259e-06   1.93076535e-06]
     [  1.69926001e-04   1.25923340e-04   2.33240296e-04   7.60839197e-05
        9.89830489e-05   1.58379311e-05   8.00005392e-06]
     [  4.20893778e-04   2.03461048e-04   6.84292101e-04   1.72696845e-04
        3.08627225e-04   5.50636993e-05   2.93943967e-05]
     [  4.81953899e-04   8.10796738e-04   1.27731424e-03   8.24448952e-04
        7.48161143e-04   1.99769340e-04   9.02831714e-05]
     [  9.80428697e-04   1.03986915e-03   3.68556718e-03   1.66879393e-03
        2.56724754e-03   5.68961868e-04   3.78457146e-04]
     [  2.40870506e-04   2.30339872e-03   4.81028886e-03   4.75397134e-03
        4.31752827e-03   2.34462771e-03   9.82118206e-04]
     [  0.00000000e+00   1.10150469e-03   1.28817322e-02   9.11579592e-03
        1.35011919e-02   6.24293419e-03   4.49124231e-03]
     [  0.00000000e+00   0.00000000e+00   9.52648414e-03   3.36188472e-02
        2.50664437e-02   2.01536701e-02   1.50427081e-02]
     [  0.00000000e+00   0.00000000e+00   0.00000000e+00   3.93092725e-02
        4.25697510e-02   6.08622868e-02   6.20709492e-02]
     [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
        0.00000000e+00   1.60052748e-01   2.33372894e-01]]


## 1.6 梯度计算
下面，我们利用前向、后者计算的 $\alpha$ 和 $\beta$ 来计算梯度。

根据 $\alpha$、$\beta$ 的定义，我们有：

$$
\alpha_t(s)\beta_t(s) = \sum_{\pi \in  B^{-1}(l):\pi_t=l_s^\prime} y^t_{l_s^\prime} \prod_{t=1}^T y^t_{\pi_t} = y^t_{l_s^\prime} \cdot \sum_{\pi \in  B^{-1}(l):\pi_t=l_s^\prime} \prod_{t=1}^T y^t_{\pi_t}
$$

则

$$
\frac{\alpha_t(s)\beta_t(s)}{ y^t_{l_s^\prime}} = \sum_{\pi \in  B^{-1}(l):\pi_t=l_s^\prime} \prod_{t=1}^T y^t_{\pi_t}  = \sum_{\pi \in  B^{-1}(l):\pi_t=l_s^\prime} p(\pi|x)
$$
于是，可得似然
$$
p(l|x) = \sum_{s=1}^{|l^\prime|} \sum_{\pi \in  B^{-1}(l):\pi_t=l_s^\prime} p(\pi|x) =  \sum_{s=1}^{|l^\prime|}  \frac{\alpha_t(s)\beta_t(s)}{ y^t_{l_s^\prime}}
$$

插播：这里p的表达式中有一个不确定的t，其实不管t取什么值，都可以得到相同的计算结果，我们可以验证一下：

```python
alpha_beta = alpha * beta

for i in range(alpha_beta.shape[0]):
    p = 0.
    for j in range(alpha_beta.shape[1]):
        p += alpha_beta[i, j] / y[i, labels[j]]
    print(p)
```

```
2.0309529674855637e-05
2.0309529674855637e-05
2.030952967485564e-05
2.0309529674855637e-05
2.0309529674855634e-05
2.0309529674855637e-05
2.0309529674855637e-05
2.030952967485564e-05
2.0309529674855637e-05
2.0309529674855637e-05
2.0309529674855637e-05
2.0309529674855637e-05
```

**发现当t取任何值的时候，计算出的$p(l|x)$都相同**，特别的，当$t=0$时，$p(x|l)$刚好就是`beta[0, 0] + beta[0, 1]`，当$t=T$时，`alpha[-1, -1] + alpha[-1, -2]`

> 插播结束

接下来为了反向传播需要计算 $\frac{\partial p(l|x)}{\partial y^t_k}$，观察上式右端求各项，仅有 $s=k$ 的项包含 $y^t_k$，因此，其他项的偏导都为零，不用考虑。于是有：
$$
\frac{\partial p(l|x)}{\partial y^t_k} = \frac{\partial \frac{\alpha_t(k)\beta_t(k)}{ y^t_{k}} }{\partial y^t_k} 
$$

利用除法的求导准则有：
$$
\frac{\partial p(l|x)}{\partial y^t_k} =  \frac{\frac{2 \cdot \alpha_t(k)\beta_t(k)}{ y^t_{k}} \cdot y^t_{k} -  \alpha_t(k)\beta_t(k) \cdot 1}{({y^t_k})^2} = \frac{\alpha_t(k)\beta_t(k)}{({y^t_k})^2}
$$

> **求导中，分子第一项是因为 $\alpha_t(k)\beta_t(k)$ 中包含为两个 $y^t_k$ 乘积项（即 $(y^t_k)^2$，因此对其导数是$2\alpha_t(k)\beta_t(k)$。其他均为与 $y^t_k$ 无关的常数。**

$l$ 中可能包含多个 $k$ 字符，它们计算的梯度要进行累加，因此，最后的梯度计算结果为：
$$
\frac{\partial p(l|x)}{\partial y^t_k} = \frac{1}{{y^t_k}^2} \sum_{s \in lab(l, k)} \alpha_t(s)\beta_t(s)
$$
其中，$lab(s)=\{s: l_s^\prime = k\}$。

一般我们优化似然函数的对数，因此，梯度计算如下：
$$
\frac{\partial \ln(p(l|x))}{\partial y^t_k} =\frac{1}{p(l|x)} \frac{\partial p(l|x)}{\partial y^t_k}
$$
其中，似然值在前向计算中已经求得： $p(l|x) = \alpha_T(|l^\prime|) + \alpha_T(|l^\prime|-1)$。

对于给定训练集 $D$，待优化的目标函数为：
$$
O(D, N_w) = -\sum_{(x,z)\in D} \ln(p(z|x))
$$

得到梯度后，我们可以利用任意优化方法（e.g. SGD， Adam）进行训练。


```python
def gradient(y, labels):
    T, V = y.shape
    L = len(labels)
    
    alpha = forward(y, labels)
    beta = backward(y, labels)
    p = alpha[-1, -1] + alpha[-1, -2]
    
    grad = np.zeros([T, V])
    for t in range(T):
        for s in range(V):
            lab = [i for i, c in enumerate(labels) if c == s]
            for i in lab:
                grad[t, s] += alpha[t, i] * beta[t, i] 
            grad[t, s] /= y[t, s] ** 2
                
    grad /= p
    return grad
    
grad = gradient(y, labels)
print(grad)
```

```
    [[ 2.50911241  0.          0.          2.27594441  0.        ]
     [ 2.25397118  0.          0.          2.47384957  0.        ]
     [ 2.65058465  0.          0.          2.77274592  0.        ]
     [ 2.46136916  0.          0.          2.29678159  0.02303985]
     [ 2.300259    0.          0.          2.37548238  0.10334851]
     [ 2.40271071  0.          0.          2.19860276  0.23513657]
     [ 1.68914157  0.          0.          1.78214377  0.51794046]
     [ 2.32536762  0.          0.          1.75750877  0.92477606]
     [ 1.92883907  0.          0.          1.45529832  1.44239844]
     [ 2.06219335  0.          0.          0.7568118   1.96405515]
     [ 2.07914466  0.          0.          0.33858403  2.35197258]
     [ 2.6816852   0.          0.          0.          2.3377753 ]]
```

将基于前向-后向算法得到梯度与基于数值的梯度比较，以验证实现的正确性。


```python
def check_grad(y, labels, w=-1, v=-1, toleration=1e-3):
    grad_1 = gradient(y, labels)[w, v]
    
    delta = 1e-10
    original = y[w, v]
    
    y[w, v] = original + delta
    alpha = forward(y, labels)
    log_p1 = np.log(alpha[-1, -1] + alpha[-1, -2])
    
    y[w, v] = original - delta
    alpha = forward(y, labels)
    log_p2 = np.log(alpha[-1, -1] + alpha[-1, -2])
    
    y[w, v] = original
    
    grad_2 = (log_p1 - log_p2) / (2 * delta)
    if np.abs(grad_1 - grad_2) > toleration:
        print('[%d, %d]：%.2e' % (w, v, np.abs(grad_1 - grad_2)))

for toleration in [1e-5, 1e-6]:
    print('%.e' % toleration)
    for w in range(y.shape[0]):
        for v in range(y.shape[1]):
            check_grad(y, labels, w, v, toleration)
```

```
1e-05
1e-06
[0, 3]：3.91e-06
[1, 0]：3.61e-06
[1, 3]：2.66e-06
[2, 0]：2.67e-06
[2, 3]：3.88e-06
[3, 0]：4.71e-06
[3, 3]：3.39e-06
[4, 0]：1.24e-06
[4, 3]：4.79e-06
[5, 0]：1.57e-06
[5, 3]：2.98e-06
[6, 0]：5.03e-06
[6, 3]：4.89e-06
[7, 0]：1.05e-06
[7, 4]：4.19e-06
[8, 4]：5.57e-06
[9, 0]：5.95e-06
[9, 3]：3.85e-06
[10, 0]：1.09e-06
[10, 3]：1.53e-06
[10, 4]：3.82e-06
```

可以看到，前向-后向及数值梯度两种方法计算的梯度差异都在 1e-5 以下，误差最多在 1e-6 的量级。这初步验证了前向-后向梯度计算方法原理和实现的正确性。

## 1.7 logits 梯度
在实际训练中，为了计算方便，可以将 CTC 和 softmax 的梯度计算合并，公式如下：

$$
\frac{\partial \ln(p(l|x))}{\partial y^t_k} = y^t_k - \frac{1}{y^t_k \cdot p(l|x)} \sum_{s \in lab(l, k)} \alpha_t(s)\beta_t(s)
$$

这是因为，softmax 的梯度反传公式为：

$$
\frac{\partial \ln(p(l|x))}{\partial u^t_k} = y^t_k (\frac{\partial \ln(p(l|x))}{\partial y^t_k}  - \sum_{j=1}^{V} \frac{\partial \ln(p(l|x))}{\partial y^t_j} y^t_j)
$$

> **为什么是softmax的反传公式是这样参考笔记**：[Softmax函数求导详解](http://blog.prince2015.club/2020/03/27/softmax/)

接合上面两式，有:

$$
\frac{\partial \ln(p(l|x))}{\partial u^t_k} = \frac{1}{y^t_k p(l|x)} \sum_{s \in lab(l, k)} \alpha_t(s)\beta_t(s) - y^t_k
$$


```python
def gradient_logits_naive(y, labels):
    '''
    gradient by back propagation
    '''
    y_grad = gradient(y, labels)
    
    sum_y_grad = np.sum(y_grad * y, axis=1, keepdims=True)
    u_grad = y * (y_grad - sum_y_grad) 
    
    return u_grad

def gradient_logits(y, labels):
    '''
    '''
    T, V = y.shape
    L = len(labels)
    
    alpha = forward(y, labels)
    beta = backward(y, labels)
    p = alpha[-1, -1] + alpha[-1, -2]
    
    u_grad = np.zeros([T, V])
    for t in range(T):
        for s in range(V):
            lab = [i for i, c in enumerate(labels) if c == s]
            for i in lab:
                u_grad[t, s] += alpha[t, i] * beta[t, i] 
            u_grad[t, s] /= y[t, s] * p
                
    u_grad -= y
    return u_grad
    
grad_l = gradient_logits_naive(y, labels)
grad_2 = gradient_logits(y, labels)

print(np.sum(np.abs(grad_l - grad_2)))
```

```
1.59941504485e-15
```

同上，我们利用数值梯度来初步检验梯度计算的正确性：


```python
def check_grad_logits(x, labels, w=-1, v=-1, toleration=1e-3):
    grad_1 = gradient_logits(softmax(x), labels)[w, v]
    
    delta = 1e-10
    original = x[w, v]
    
    x[w, v] = original + delta
    y = softmax(x)
    alpha = forward(y, labels)
    log_p1 = np.log(alpha[-1, -1] + alpha[-1, -2])
    
    x[w, v] = original - delta
    y = softmax(x)
    alpha = forward(y, labels)
    log_p2 = np.log(alpha[-1, -1] + alpha[-1, -2])
    
    x[w, v] = original
    
    grad_2 = (log_p1 - log_p2) / (2 * delta)
    if np.abs(grad_1 - grad_2) > toleration:
        print('[%d, %d]：%.2e, %.2e, %.2e' % (w, v, grad_1, grad_2, np.abs(grad_1 - grad_2)))

np.random.seed(1111)
x = np.random.random([10, 10])
for toleration in [1e-5, 1e-6]:
    print('%.e' % toleration)
    for w in range(x.shape[0]):
        for v in range(x.shape[1]):
            check_grad_logits(x, labels, w, v, toleration)
```

    1e-05
    [0, 6]：-8.00e-02, -8.00e-02, 1.03e-05
    [4, 0]：4.29e-01, 4.29e-01, 1.10e-05
    [5, 6]：-7.59e-02, -7.59e-02, 1.22e-05
    [6, 2]：-1.38e-01, -1.38e-01, 1.23e-05
    [6, 3]：3.33e-01, 3.33e-01, 1.02e-05
    1e-06
    [0, 0]：3.88e-01, 3.88e-01, 7.03e-06
    [0, 1]：-1.59e-01, -1.59e-01, 2.78e-06
    [0, 2]：-8.89e-02, -8.89e-02, 3.47e-06
    [0, 3]：4.57e-01, 4.57e-01, 1.64e-06
    [0, 4]：-6.32e-02, -6.32e-02, 7.19e-06
    [0, 5]：-7.98e-02, -7.98e-02, 5.46e-06
    [0, 6]：-8.00e-02, -8.00e-02, 1.03e-05
    [0, 7]：-1.32e-01, -1.32e-01, 2.21e-06
    [0, 8]：-1.04e-01, -1.04e-01, 7.75e-06
    [0, 9]：-1.38e-01, -1.38e-01, 5.95e-06
    [1, 0]：3.41e-01, 3.41e-01, 2.79e-06
    [1, 1]：-1.18e-01, -1.18e-01, 6.08e-06
    [1, 3]：5.04e-01, 5.04e-01, 4.06e-06
    [1, 4]：-9.96e-02, -9.96e-02, 5.77e-06
    [1, 5]：-8.22e-02, -8.22e-02, 4.03e-06
    [1, 6]：-9.46e-02, -9.46e-02, 4.49e-06
    [1, 7]：-1.49e-01, -1.49e-01, 3.96e-06
    [1, 8]：-1.24e-01, -1.24e-01, 4.96e-06
    [1, 9]：-7.48e-02, -7.47e-02, 5.94e-06
    [2, 0]：3.29e-01, 3.29e-01, 3.47e-06
    [2, 1]：-9.42e-02, -9.42e-02, 1.63e-06
    [2, 2]：-9.17e-02, -9.17e-02, 4.47e-06
    [2, 3]：4.50e-01, 4.50e-01, 2.14e-06
    [2, 5]：-1.07e-01, -1.07e-01, 6.33e-06
    [2, 6]：-5.42e-02, -5.42e-02, 1.71e-06
    [2, 7]：-9.68e-02, -9.68e-02, 7.69e-06
    [2, 9]：-1.21e-01, -1.21e-01, 9.06e-06
    [3, 0]：4.42e-01, 4.42e-01, 9.21e-06
    [3, 1]：-6.71e-02, -6.71e-02, 5.75e-06
    [3, 2]：-1.16e-01, -1.16e-01, 5.26e-06
    [3, 3]：4.03e-01, 4.03e-01, 6.39e-06
    [3, 4]：-1.07e-01, -1.07e-01, 2.42e-06
    [3, 5]：-1.25e-01, -1.25e-01, 8.90e-06
    [3, 6]：-1.17e-01, -1.17e-01, 2.08e-06
    [3, 7]：-1.32e-01, -1.32e-01, 2.21e-06
    [3, 8]：-6.90e-02, -6.90e-02, 1.72e-06
    [3, 9]：-1.13e-01, -1.13e-01, 6.68e-06
    [4, 0]：4.29e-01, 4.29e-01, 1.10e-05
    [4, 1]：-7.17e-02, -7.17e-02, 7.76e-06
    [4, 3]：3.25e-01, 3.25e-01, 2.93e-06
    [4, 4]：-5.91e-02, -5.91e-02, 1.88e-06
    [4, 5]：-7.78e-02, -7.78e-02, 5.82e-06
    [4, 6]：-9.08e-02, -9.08e-02, 8.04e-06
    [4, 7]：-1.12e-01, -1.12e-01, 2.11e-06
    [4, 8]：-1.26e-01, -1.26e-01, 5.92e-06
    [4, 9]：-1.40e-01, -1.40e-01, 4.89e-06
    [5, 0]：1.86e-01, 1.86e-01, 6.60e-06
    [5, 2]：-9.52e-02, -9.52e-02, 5.01e-06
    [5, 3]：4.97e-01, 4.97e-01, 6.92e-06
    [5, 4]：2.50e-02, 2.50e-02, 5.20e-06
    [5, 5]：-7.93e-02, -7.93e-02, 2.78e-06
    [5, 6]：-7.59e-02, -7.59e-02, 1.22e-05
    [5, 7]：-1.05e-01, -1.05e-01, 9.53e-06
    [5, 8]：-1.25e-01, -1.25e-01, 9.07e-06
    [5, 9]：-6.74e-02, -6.74e-02, 3.08e-06
    [6, 0]：1.49e-01, 1.49e-01, 5.17e-06
    [6, 1]：-6.67e-02, -6.66e-02, 4.40e-06
    [6, 2]：-1.38e-01, -1.38e-01, 1.23e-05
    [6, 3]：3.33e-01, 3.33e-01, 1.02e-05
    [6, 4]：1.72e-01, 1.72e-01, 7.79e-06
    [6, 6]：-6.73e-02, -6.73e-02, 7.13e-06
    [6, 7]：-6.08e-02, -6.08e-02, 8.58e-06
    [6, 8]：-9.98e-02, -9.98e-02, 6.14e-06
    [6, 9]：-1.31e-01, -1.31e-01, 8.89e-06
    [7, 0]：2.39e-01, 2.39e-01, 2.53e-06
    [7, 2]：-1.61e-01, -1.61e-01, 1.78e-06
    [7, 3]：1.02e-01, 1.02e-01, 4.73e-06
    [7, 4]：3.63e-01, 3.63e-01, 7.51e-06
    [7, 5]：-6.02e-02, -6.02e-02, 4.63e-06
    [7, 6]：-1.05e-01, -1.05e-01, 3.29e-06
    [7, 8]：-8.83e-02, -8.83e-02, 4.16e-06
    [7, 9]：-6.05e-02, -6.05e-02, 2.89e-06
    [8, 0]：2.92e-01, 2.92e-01, 1.38e-06
    [8, 1]：-9.70e-02, -9.70e-02, 2.27e-06
    [8, 2]：-9.87e-02, -9.87e-02, 4.36e-06
    [8, 4]：5.07e-01, 5.07e-01, 2.30e-06
    [8, 5]：-1.28e-01, -1.28e-01, 7.41e-06
    [8, 6]：-1.09e-01, -1.09e-01, 6.32e-06
    [8, 7]：-1.20e-01, -1.20e-01, 6.01e-06
    [8, 8]：-7.00e-02, -7.00e-02, 1.44e-06
    [8, 9]：-1.55e-01, -1.55e-01, 2.47e-06
    [9, 0]：4.90e-01, 4.90e-01, 2.04e-06
    [9, 1]：-7.69e-02, -7.69e-02, 1.54e-06
    [9, 2]：-9.72e-02, -9.72e-02, 9.59e-06
    [9, 4]：2.61e-01, 2.61e-01, 2.16e-06
    [9, 5]：-9.27e-02, -9.27e-02, 5.07e-06
    [9, 6]：-7.70e-02, -7.70e-02, 1.03e-06
    [9, 7]：-8.30e-02, -8.30e-02, 5.42e-06
    [9, 8]：-1.17e-01, -1.17e-01, 4.26e-06
    [9, 9]：-1.39e-01, -1.39e-01, 3.53e-06


# 2. 数值稳定性

CTC 的训练过程面临数值下溢的风险，特别是序列较大的情况下。下面介绍两种数值上稳定的工程优化方法：1）log 域（许多 CRF 实现的常用方法）；2）scale 技巧（原始论文 [1] 使用的方法）。



## 2.1 log 域计算

log 计算涉及 logsumexp 操作。
[经验表明](https://github.com/baidu-research/warp-ctc)，在 log 域计算，即使使用单精度，也表现出良好的数值稳定性，可以有效避免下溢的风险。稳定性的代价是增加了运算的复杂性——原始实现只涉及乘加运算，log 域实现则需要对数和指数运算。

**这里解释一下为什么要使用log域计算：**

比如计算机最小只能表示0.00001，那么当x = 0.00001, y = 0.00001相乘时，就会发生下溢，但是我们还是要表示出两个数乘积的，因此这里在计算$\alpha$的时候就存储$log(\alpha_t(s))$，这个数字就不会下溢了。

但是还有一个问题，我们在迭代计算$\alpha$的时候不仅需要存储，还需要进行运算，具体地，需要先计算两个或三个数值的和，然后再乘以$y_{l'_s}^{t}$，相当于是已经有了`a=log(x)，b=log(y)，c=log(z)`，需要计算`log((x + y) * z)`,后面一个z比较好处理，就直接是`log(x + y) + log(z) = log(x + y) + c`，但是前面`log(x + y)`，我们不能直接把x和y直接计算出来了（**因为x和y本身超过了计算机表示的范围，只要算出来就是下溢。所以要在不计算出a和b的前提下，算出log(x + y)**），这里使用的是`a + log(1 + exp(b - a))`来代替，原因是：
$$
\ln(x + y) = \ln(x(1 + \frac{y}{x})) = \ln{x} + \ln(1 + e^{\ln y - \ln x}) = a + \ln(1 + e^{b - a})
$$
实现使用以下代码：


```python
ninf = -np.float('inf')

def _logsumexp(a, b):
    '''
    np.log(np.exp(a) + np.exp(b))

    '''
    
    if a < b:
        a, b = b, a
        
    if b == ninf:
        return a
    else:
        return a + np.log(1 + np.exp(b - a)) 
    
def logsumexp(*args):
    '''
    from scipy.special import logsumexp
    logsumexp(args)
    '''
    res = args[0]
    for e in args[1:]:
        res = _logsumexp(res, e)
    return res
```

### 2.1.1 log 域前向算法
基于 log 的前向算法实现如下：


```python
def forward_log(log_y, labels):
    T, V = log_y.shape
    L = len(labels)
    log_alpha = np.ones([T, L]) * ninf

    # init, alpha中只存储log后的概率
    log_alpha[0, 0] = log_y[0, labels[0]]
    log_alpha[0, 1] = log_y[0, labels[1]]

    for t in range(1, T):
        for i in range(L):
            s = labels[i]
            
            a = log_alpha[t - 1, i]
            if i - 1 >= 0:
                a = logsumexp(a, log_alpha[t - 1, i - 1]) # 这一步用到了上面的公式
            if i - 2 >= 0 and s != 0 and s != labels[i - 2]:
                a = logsumexp(a, log_alpha[t - 1, i - 2])
                
            log_alpha[t, i] = a + log_y[t, s]  # 最后使用加法代替乘法，得出的还是log域上的值
            
    return log_alpha

log_alpha = forward_log(np.log(y), labels)
alpha = forward(y, labels)
print(np.sum(np.abs(np.exp(log_alpha) - alpha)))
```

```
8.60881935942e-17
```

### 2.1.2 log 域后向算法
基于 log 的后向算法实现如下：


```python
def backward_log(log_y, labels):
    T, V = log_y.shape
    L = len(labels)
    log_beta = np.ones([T, L]) * ninf

    # init，同上面的代码
    log_beta[-1, -1] = log_y[-1, labels[-1]]
    log_beta[-1, -2] = log_y[-1, labels[-2]]

    for t in range(T - 2, -1, -1):
        for i in range(L):
            s = labels[i]
            
            a = log_beta[t + 1, i] 
            if i + 1 < L:
                a = logsumexp(a, log_beta[t + 1, i + 1]) # log域的加法
            if i + 2 < L and s != 0 and s != labels[i + 2]:
                a = logsumexp(a, log_beta[t + 1, i + 2])
                
            log_beta[t, i] = a + log_y[t, s]  # log域乘法
            
    return log_beta

log_beta = backward_log(np.log(y), labels)
beta = backward(y, labels)
print(np.sum(np.abs(np.exp(log_beta) - beta)))
```

```
1.10399945005e-16
```

### 2.1.3 log 域梯度计算
在前向、后向基础上，也可以在 log 域上计算梯度。（对比不使用log域的梯度计算）


```python
def gradient_log(log_y, labels):
    T, V = log_y.shape
    L = len(labels)
    
    log_alpha = forward_log(log_y, labels)
    log_beta = backward_log(log_y, labels)
    log_p = logsumexp(log_alpha[-1, -1], log_alpha[-1, -2])
    
    log_grad = np.ones([T, V]) * ninf
    for t in range(T):
        for s in range(V):
            lab = [i for i, c in enumerate(labels) if c == s]
            for i in lab:
                log_grad[t, s] = logsumexp(log_grad[t, s], log_alpha[t, i] + log_beta[t, i]) 
            log_grad[t, s] -= 2 * log_y[t, s]
                
    log_grad -= log_p  # Log域除法
    return log_grad
    
log_grad = gradient_log(np.log(y), labels)
grad = gradient(y, labels)
#print(log_grad)
#print(grad)
print(np.sum(np.abs(np.exp(log_grad) - grad)))
```

```
4.97588081849e-14
```

## 2.2 scale

### 2.2.1 前向算法

为了防止下溢，在前向算法的每个时刻，都对计算出的 $\alpha$ 的范围进行缩放：

$$
C_t \stackrel{def}{=} \sum_s\alpha_t(s)
$$

$$
\hat{\alpha}_t = \frac{\alpha_t(s)}{C_t}
$$

缩放后的 $\alpha$，不会随着时刻的积累变得太小。$\hat{\alpha}$ 替代 $\alpha$，进行下一时刻的迭代。

> 这种方法是等同于log域计算的稳定数值计算方法，两者都可以，具体做法相当于是对alpha做t维度上的归一化，针对每个t，求一个sum，然后每个值都除以这个sum，由于alpha的计算只有同一个t上的加法和t与t-1的乘法，所以这个方法是可行的。
>
> 但要注意的是，在求最后概率的时候，由于前面的alpha的值都是经过归一化的，所以，需要把之前每个sum都保存起来，然后用sum的累乘值乘以最后的概率，这样才能把概率还原出来，而这些sum的补偿，本身又可能造成下溢，比如每个值都小于1，那么乘起来还是有下溢的风险，因此，在补偿的时候也要使用log域的加法代替乘法。

```python
def forward_scale(y, labels):
    T, V = y.shape
    L = len(labels)
    alpha_scale = np.zeros([T, L])

    # init
    alpha_scale[0, 0] = y[0, labels[0]]
    alpha_scale[0, 1] = y[0, labels[1]]
    Cs = []
    
    C = np.sum(alpha_scale[0])
    alpha_scale[0] /= C
    Cs.append(C)

    for t in range(1, T):
        for i in range(L):
            s = labels[i]
            
            a = alpha_scale[t - 1, i] 
            if i - 1 >= 0:
                a += alpha_scale[t - 1, i - 1]
            if i - 2 >= 0 and s != 0 and s != labels[i - 2]:
                a += alpha_scale[t - 1, i - 2]
                
            alpha_scale[t, i] = a * y[t, s]
            
        C = np.sum(alpha_scale[t])  # 计算完以后对当前t进行归一化
        alpha_scale[t] /= C
        Cs.append(C)
            
    return alpha_scale, Cs
```

由于进行了缩放，最后计算概率时要时行补偿（使用log域乘法）：
$$
p(l|x) = \alpha_T(|l^\prime|) + \alpha_T(|l^\prime|-1) = (\hat\alpha_T(|l^\prime|) + \hat\alpha_T(|l^\prime|-1) * \prod_{t=1}^T C_t
$$

$$
\ln(p(l|x)) = \sum_t^T\ln(C_t) + \ln(\hat\alpha_T(|l^\prime|) + \hat\alpha_T(|l^\prime|-1))
$$


```python
labels = [0, 1, 2, 0]  # 0 for blank

alpha_scale, Cs = forward_scale(y, labels)
log_p = np.sum(np.log(Cs)) + np.log(alpha_scale[-1][labels[-1]] + alpha_scale[-1][labels[-2]])

alpha = forward(y, labels)
p = alpha[-1, labels[-1]] + alpha[-1, labels[-2]]

print(np.log(p), log_p, np.log(p) - log_p)
```

```
(-13.202925982240107, -13.202925982240107, 0.0)
```

### 2.2.2 后向算法
后向算法缩放类似于前向算法，公式如下：

$$
D_t \stackrel{def}{=} \sum_s\beta_t(s)
$$

$$
\hat{\beta}_t = \frac{\beta_t(s)}{D_t}
$$

```python
def backward_scale(y, labels):
    T, V = y.shape
    L = len(labels)
    beta_scale = np.zeros([T, L])

    # init
    beta_scale[-1, -1] = y[-1, labels[-1]]
    beta_scale[-1, -2] = y[-1, labels[-2]]
    
    Ds = []
    
    D = np.sum(beta_scale[-1,:])
    beta_scale[-1] /= D
    Ds.append(D)

    for t in range(T - 2, -1, -1):
        for i in range(L):
            s = labels[i]
            
            a = beta_scale[t + 1, i] 
            if i + 1 < L:
                a += beta_scale[t + 1, i + 1]
            if i + 2 < L and s != 0 and s != labels[i + 2]:
                a += beta_scale[t + 1, i + 2]
                
            beta_scale[t, i] = a * y[t, s]
            
        D = np.sum(beta_scale[t])
        beta_scale[t] /= D
        Ds.append(D)
            
    return beta_scale, Ds[::-1]

beta_scale, Ds = backward_scale(y, labels)
print(beta_scale)
```

    [[ 0.71362347  0.18910147  0.07964328  0.01763178]
     [ 0.70165268  0.15859852  0.11849423  0.02125457]
     [ 0.67689676  0.165374    0.13221504  0.02551419]
     [ 0.71398181  0.11936432  0.13524265  0.03141122]
     [ 0.70769657  0.13093688  0.12447135  0.0368952 ]
     [ 0.63594568  0.1790638   0.14250065  0.04248987]
     [ 0.63144322  0.1806382   0.13366043  0.05425815]
     [ 0.33926289  0.35149591  0.24988622  0.05935497]
     [ 0.30303623  0.26644554  0.33088584  0.0996324 ]
     [ 0.12510056  0.3297143   0.3956509   0.14953425]
     [ 0.          0.22078343  0.5153114   0.26390517]
     [ 0.          0.          0.550151    0.449849  ]]


### 2.2.3 梯度计算

$$
\frac{\partial \ln(p(l|x))}{\partial y^t_k} = \frac{1}{p(l|x)} \frac{\partial p(l|x)}{\partial y^t_k} = \frac{1}{p(l|x)} \frac{1}{{y^t_k}^2} \sum_{s \in lab(l, k)} \alpha_t(s)\beta_t(s) 
$$

考虑到
$$
p(l|x) =  \sum_{s=1}^{|l^\prime|}  \frac{\alpha_t(s)\beta_t(s)}{ y^t_{l_s^\prime}} 
$$
以及
$$
\alpha_t(s) = \hat\alpha_t(s) \cdot \prod_{k=1}^t C_k
$$

$$
\beta_t(s) = \hat\beta_t(s) \cdot \prod_{k=t}^T D_k
$$

$$
\frac{\partial \ln(p(l|x))}{\partial y^t_k} = \frac{1}{\sum_{s=1}^{|l^\prime|} \frac{\hat\alpha_t(s)  \hat\beta_t(s)}{y^t_{l^\prime_s}}} \frac{1}{{y^t_k}^2} \sum_{s \in lab(l, k)} \hat\alpha_t(s)  \hat\beta_t(s)
$$

式中最右项中的各个部分我们都已经求得。梯度计算实现如下：


```python
def gradient_scale(y, labels):
    T, V = y.shape
    L = len(labels)
    
    alpha_scale, _ = forward_scale(y, labels)
    beta_scale, _ = backward_scale(y, labels)
    
    grad = np.zeros([T, V])
    for t in range(T):
        for s in range(V):
            lab = [i for i, c in enumerate(labels) if c == s]
            for i in lab:
                grad[t, s] += alpha_scale[t, i] * beta_scale[t, i]
            grad[t, s] /= y[t, s] ** 2
         
        # normalize factor
        z = 0
        for i in range(L):
            z += alpha_scale[t, i] * beta_scale[t, i] / y[t, labels[i]]
        grad[t] /= z
        
    return grad
    
labels = [0, 3, 0, 3, 0, 4, 0]  # 0 for blank
grad_1 = gradient_scale(y, labels)
grad_2 = gradient(y, labels)
print(np.sum(np.abs(grad_1 - grad_2)))
```

```
6.86256607096e-15
```

### 2.2.4 logits 梯度
类似于 y 梯度的推导，logits 梯度计算公式如下：
$$
\frac{\partial \ln(p(l|x))}{\partial u^t_k} = \frac{1}{y^t_k Z_t} \sum_{s \in lab(l, k)} \hat\alpha_t(s)\hat\beta_t(s) - y^t_k
$$
其中，
$$
Z_t \stackrel{def}{=} \sum_{s=1}^{|l^\prime|} \frac{\hat\alpha_t(s)\hat\beta_t(s)}{y^t_{l^\prime_s}}
$$

# 4. 工具

- [warp-ctc](https://github.com/baidu-research/warp-ctc) 是百度开源的基于 CPU 和 GPU 的高效并行实现。warp-ctc 自身提供 C 语言接口，对于流利的机器学习工具（[torch](https://github.com/baidu-research/warp-ctc/tree/master/torch_binding)、 [pytorch](https://github.com/SeanNaren/deepspeech.pytorch) 和 [tensorflow](https://github.com/baidu-research/warp-ctc/tree/master/tensorflow_binding)、[chainer](https://github.com/jheymann85/chainer_ctc)）都有相应的接口绑定。

- [cudnn 7](https://developer.nvidia.com/cudnn) 以后开始提供 CTC 支持。

- Tensorflow 也原生支持 [CTC loss](https://www.tensorflow.org/api_docs/python/tf/nn/ctc_loss)，及 greedy 和 beam search 解码器。

# 小结
1. CTC 可以建模无对齐信息的多对多序列问题（输入长度不小于输出），如语音识别、连续字符识别 [3,4]。
2. CTC 不需要输入与输出的对齐信息，可以实现端到端的训练。
3. CTC 在 loss 的计算上，利用了整个 labels 序列的全局信息，某种意义上相对逐帧计算损失的方法，"更加区分性"。

# References
1. Graves et al. [Connectionist Temporal Classification : Labelling Unsegmented Sequence Data with Recurrent Neural Networks](ftp://ftp.idsia.ch/pub/juergen/icml2006.pdf).
2. Hannun et al. [First-Pass Large Vocabulary Continuous Speech Recognition using Bi-Directional Recurrent DNNs](https://arxiv.org/abs/1408.2873).
3. Graves et al. [Towards End-To-End Speech Recognition with Recurrent Neural Networks](http://jmlr.org/proceedings/papers/v32/graves14.pdf).
4. Liwicki et al. [A novel approach to on-line handwriting recognition based on bidirectional long short-term memory networks](https://www.cs.toronto.edu/~graves/icdar_2007.pdf).
5. Zenkel et al. [Comparison of Decoding Strategies for CTC Acoustic Models](https://arxiv.org/abs/1708.004469).
5. Huannun. [Sequence Modeling with CTC](https://distill.pub/2017/ctc/).


