## Moving Average Computation
Average:

- Simple Moving Average (SMA)
- Exponential Moving Average (EMA)

Given data points $N$ ($N$ is data growth with length $k$)

- SMA 
$$
	\frac{1}{n} \sum_{k=1}^{n}{N_k}
$$

- SMA

```python
def SMA(N):
	runing_mean = 0
	decay = 0.99
	for i in range(n):
		 runing_mean = running_mean * decay + N[i]
	return running_mean 
```

In `Batch Normalization (BN)` when testing with one testing set, we need to use SMA to estimate the population `Mean` and `Var`