[TOC]

# 3. CTC解码
训练和的 $N_w$ 可以用来预测新的样本输入对应的输出字符串，这涉及到解码。
按照最大似然准则，最优的解码结果为：
$$
h(x) = \underset{l \in L^{\le T}}{\mathrm{argmax}}\ p(l|x)
$$

然而，上式不存在已知的高效解法。下面介绍几种实用的近似破解码方法。

## 3.1 贪心搜索 （greedy search）
虽然 $p(l|x)$ 难以有效的计算，但是由于 CTC 的独立性假设，对于某个具体的字符串 $\pi$（去 blank 前），确容易计算：
$$
p(\pi|x) = \prod_{k=1}^T p(\pi_k|x)
$$

因此，我们放弃寻找使 $p(l|x)$ 最大的字符串，退而寻找一个使 $p(\pi|x)$ 最大的字符串，即：

$$
h(x) \approx B(\pi^\star)
$$
其中，
$$
\pi^\star = \underset{\pi \in N^T}{\mathrm{argmax}}\ p(\pi|x)
$$

简化后，解码过程（构造 $\pi^\star$）变得非常简单（基于独立性假设）： 在每个时刻输出概率最大的字符:
$$
\pi^\star = cat_{t=1}^T(\underset{s \in L^\prime}{\mathrm{argmax}}\ y^t_s)
$$





```python
def remove_blank(labels, blank=0):
    new_labels = []
    
    # combine duplicate
    previous = None
    for l in labels:
        if l != previous:
            new_labels.append(l)
            previous = l
            
    # remove blank     
    new_labels = [l for l in new_labels if l != blank]
    
    return new_labels

def insert_blank(labels, blank=0):
    new_labels = [blank]
    for l in labels:
        new_labels += [l, blank]
    return new_labels

def greedy_decode(y, blank=0):
    raw_rs = np.argmax(y, axis=1)
    rs = remove_blank(raw_rs, blank)
    return raw_rs, rs

np.random.seed(1111)
y = softmax(np.random.random([20, 6]))
rr, rs = greedy_decode(y)
print(rr)
print(rs)
```

```
    [1 3 5 5 5 5 1 5 3 4 4 3 0 4 5 0 3 1 3 3]
    [1, 3, 5, 1, 5, 3, 4, 3, 4, 5, 3, 1, 3]
```

## 3.2 束搜索（Beam Search）
显然，贪心搜索的性能非常受限。例如，它不能给出除最优路径之外的其他其优路径。很多时候，如果我们能拿到 nbest 的路径，后续可以利用其他信息来进一步优化搜索的结果。束搜索能近似找出 top 最优的若干条路径。


```python
def beam_decode(y, beam_size=10):
    T, V = y.shape
    log_y = np.log(y)
    
    beam = [([], 0)]
    for t in range(T):  # for every timestep
        new_beam = []
        for prefix, score in beam:
            for i in range(V):  # for every state
                new_prefix = prefix + [i]
                new_score = score + log_y[t, i]
                
                new_beam.append((new_prefix, new_score))
                
        # top beam_size
        new_beam.sort(key=lambda x: x[1], reverse=True)
        beam = new_beam[:beam_size]
        
    return beam
    
np.random.seed(1111)
y = softmax(np.random.random([20, 6]))
beam = beam_decode(y, beam_size=100)
for string, score in beam[:20]:
    print(remove_blank(string), score)
```

```
    ([1, 3, 5, 1, 5, 3, 4, 3, 4, 5, 3, 1, 3], -29.261797539205567)
    ([1, 3, 5, 1, 5, 3, 4, 3, 3, 5, 3, 1, 3], -29.279020152518033)
    ([1, 3, 5, 1, 5, 3, 4, 2, 3, 4, 5, 3, 1, 3], -29.300726142201842)
    ([1, 5, 1, 5, 3, 4, 3, 4, 5, 3, 1, 3], -29.310307014773972)
    ([1, 3, 5, 1, 5, 3, 4, 2, 3, 3, 5, 3, 1, 3], -29.317948755514308)
    ([1, 5, 1, 5, 3, 4, 3, 3, 5, 3, 1, 3], -29.327529628086438)
    ([1, 3, 5, 1, 5, 4, 3, 4, 5, 3, 1, 3], -29.331572723457334)
    ([1, 3, 5, 5, 1, 5, 3, 4, 3, 4, 5, 3, 1, 3], -29.332631809924511)
    ([1, 3, 5, 4, 1, 5, 3, 4, 3, 4, 5, 3, 1, 3], -29.334649090836038)
    ([1, 3, 5, 1, 5, 3, 4, 3, 4, 5, 3, 1, 3], -29.33969505198154)
    ([1, 3, 5, 2, 1, 5, 3, 4, 3, 4, 5, 3, 1, 3], -29.339823066915415)
    ([1, 3, 5, 1, 5, 4, 3, 3, 5, 3, 1, 3], -29.3487953367698)
    ([1, 5, 1, 5, 3, 4, 2, 3, 4, 5, 3, 1, 3], -29.349235617770248)
    ([1, 3, 5, 5, 1, 5, 3, 4, 3, 3, 5, 3, 1, 3], -29.349854423236977)
    ([1, 3, 5, 1, 5, 3, 4, 3, 4, 5, 3, 3], -29.350803198551016)
    ([1, 3, 5, 4, 1, 5, 3, 4, 3, 3, 5, 3, 1, 3], -29.351871704148504)
    ([1, 3, 5, 1, 5, 3, 4, 3, 3, 5, 3, 1, 3], -29.356917665294006)
    ([1, 3, 5, 2, 1, 5, 3, 4, 3, 3, 5, 3, 1, 3], -29.357045680227881)
    ([1, 3, 5, 1, 5, 3, 4, 5, 4, 5, 3, 1, 3], -29.363802591012263)
    ([1, 5, 1, 5, 3, 4, 2, 3, 3, 5, 3, 1, 3], -29.366458231082714)
```

## 3.3 前缀束搜索（Prefix Beam Search）
直接的束搜索的一个问题是，在保存的 top N 条路径中，可能存在多条实际上是同一结果（经过去重复、去 blank 操作）。这减少了搜索结果的多样性。下面介绍的前缀搜索方法，在搜索过程中不断的合并相同的前缀[2]。参考 [gist](https://gist.github.com/awni/56369a90d03953e370f3964c826ed4b0)，前缀束搜索实现如下：


```python
from collections import defaultdict

def prefix_beam_decode(y, beam_size=10, blank=0):
    T, V = y.shape
    log_y = np.log(y)
    
    beam = [(tuple(), (0, ninf))]  # blank, non-blank
    for t in range(T):  # for every timestep
        new_beam = defaultdict(lambda : (ninf, ninf))
             
        for prefix, (p_b, p_nb) in beam:
            for i in range(V):  # for every state
                p = log_y[t, i]
                
                if i == blank:  # propose a blank
                    new_p_b, new_p_nb = new_beam[prefix]
                    new_p_b = logsumexp(new_p_b, p_b + p, p_nb + p)
                    new_beam[prefix] = (new_p_b, new_p_nb)
                    continue
                else:  # extend with non-blank
                    end_t = prefix[-1] if prefix else None
                    
                    # exntend current prefix
                    new_prefix = prefix + (i,)
                    new_p_b, new_p_nb = new_beam[new_prefix]
                    if i != end_t:
                        new_p_nb = logsumexp(new_p_nb, p_b + p, p_nb + p)
                    else:
                        new_p_nb = logsumexp(new_p_nb, p_b + p)
                    new_beam[new_prefix] = (new_p_b, new_p_nb)
                    
                    # keep current prefix
                    if i == end_t:
                        new_p_b, new_p_nb = new_beam[prefix]
                        new_p_nb = logsumexp(new_p_nb, p_nb + p)
                        new_beam[prefix] = (new_p_b, new_p_nb)
                
        # top beam_size
        beam = sorted(new_beam.items(), key=lambda x : logsumexp(*x[1]), reverse=True)
        beam = beam[:beam_size]
        
    return beam

np.random.seed(1111)
y = softmax(np.random.random([20, 6]))
beam = prefix_beam_decode(y, beam_size=100)
for string, score in beam[:20]:
    print(remove_blank(string), score)
```

    ([1, 5, 4, 1, 3, 4, 5, 2, 3], (-18.189863809114193, -17.613677981426175))
    ([1, 5, 4, 5, 3, 4, 5, 2, 3], (-18.19636512622969, -17.621013424585406))
    ([1, 5, 4, 1, 3, 4, 5, 1, 3], (-18.317018960331531, -17.666629973270073))
    ([1, 5, 4, 5, 3, 4, 5, 1, 3], (-18.323388267369936, -17.674125139073176))
    ([1, 5, 4, 1, 3, 4, 3, 2, 3], (-18.415808498759556, -17.862744326248826))
    ([1, 5, 4, 1, 3, 4, 3, 5, 3], (-18.366422766638632, -17.898463479112884))
    ([1, 5, 4, 5, 3, 4, 3, 2, 3], (-18.42224294936932, -17.870025672291458))
    ([1, 5, 4, 5, 3, 4, 3, 5, 3], (-18.372199113900191, -17.905130493229173))
    ([1, 5, 4, 1, 3, 4, 5, 4, 3], (-18.457066311773847, -17.880630315602037))
    ([1, 5, 4, 5, 3, 4, 5, 4, 3], (-18.462614293487096, -17.88759583852546))
    ([1, 5, 4, 1, 3, 4, 5, 3, 2], (-18.458941701567706, -17.951422824358747))
    ([1, 5, 4, 5, 3, 4, 5, 3, 2], (-18.464527031120184, -17.958629487208658))
    ([1, 5, 4, 1, 3, 4, 3, 1, 3], (-18.540857550725587, -17.920589910093689))
    ([1, 5, 4, 5, 3, 4, 3, 1, 3], (-18.547146092248852, -17.928030266681613))
    ([1, 5, 4, 1, 3, 4, 5, 3, 2, 3], (-19.325467801462263, -17.689203224408899))
    ([1, 5, 4, 5, 3, 4, 5, 3, 2, 3], (-19.328748799764973, -17.694105969982637))
    ([1, 5, 4, 1, 3, 4, 5, 3, 4], (-18.79699026165903, -17.945090229238392))
    ([1, 5, 4, 5, 3, 4, 5, 3, 4], (-18.803585534273239, -17.95258394264377))
    ([1, 5, 4, 3, 4, 3, 5, 2, 3], (-19.181531846082809, -17.859420073785095))
    ([1, 5, 4, 1, 3, 4, 5, 2, 3, 2], (-19.439349296385199, -17.884502168470895))


## 3.4 其他解码方法
上述介绍了基本解码方法。实际中，搜索过程可以接合额外的信息，提高搜索的准确度（例如在语音识别任务中，加入语言模型得分信息, [前缀搜索+语言模型](https://github.com/PaddlePaddle/DeepSpeech/blob/develop/decoders/decoders_deprecated.py
)）。

本质上，CTC 只是一个训练准则。训练完成后，$N_w$ 输出一系列概率分布，这点和常规基于交叉熵准则训练的模型完全一致。因此，特定应用领域常规的解码也可以经过一定修改用来 CTC 的解码。例如在语音识别任务中，利用 CTC 训练的声学模型可以无缝融入原来的 WFST 的解码器中[5]（e.g. 参见 [EESEN](https://github.com/srvk/eesen)）。

此外，[1] 给出了一种利用 CTC 顶峰特点的启发式搜索方法。