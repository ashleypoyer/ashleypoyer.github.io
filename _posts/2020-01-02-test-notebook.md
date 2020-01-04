---
layout: post
title:  "A Test Post Made from a Jupyter Notebook"
mathjax: true
date:   2020-01-01 14:03:14 -0500
description: This is a post made from a jupyter notebook. Come check this out!
categories: notebooks 
---


<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>


## Simple Linear Regression

### Ordinary Least Squares Approach (OLS)

#### Ordinary Least Squares Approach (OLS)

```python
import numpy as np
# here is a comment
def function(param):
    param += 1
    return param

print("Some string!")

for i in range(n):
    for j in range(m):
        if param != True:
            break
```

$$
\begin{equation*}
\mathbf{V}_1 \times \mathbf{V}_2 =  \begin{vmatrix}
\mathbf{i} & \mathbf{j} & \mathbf{k} \\
\frac{\partial X}{\partial u} &  \frac{\partial Y}{\partial u} & 0 \\
\frac{\partial X}{\partial v} &  \frac{\partial Y}{\partial v} & 0
\end{vmatrix}
\end{equation*}
$$


$$ 
\begin{gather*}
X = 
\begin{bmatrix}
X_{11} & X_{12} & \cdots & X_{1k} \\ 
X_{21} & X_{22} & \cdots & X_{2k} \\
\vdots & \vdots & \ddots & \vdots \\
X_{n1} & X_{n2} & \cdots & X_{nk}
\end{bmatrix}, &
y = 
\begin{bmatrix}
y_1 \\
y_2 \\
\vdots \\
y_n
\end{bmatrix}, &
\beta = 
\begin{bmatrix}
\beta_1 \\
\beta_2 \\
\vdots \\
\beta_k
\end{bmatrix}
&& \text {where $X_{i1}$ = 1 so $k$ = number of regressors + 1} 
\end{gather*}
$$

<div lang="latex">x_i + y_j</div>

```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid')
```

Here is some inline MathJax $$ x_i $$ and $$y_j$$.

```python
def scatter_data(X, y, title=None):
    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(X, y, c='dodgerblue')
    ax.xaxis.label.set_size(16)
    ax.yaxis.label.set_size(16)
    ax.set_title('Simple Linear Regression with Toy Data', fontsize=14)
    ax.set(xlabel="x1", ylabel="y")
    if title != None:
        ax.set_title(title, fontsize=14)
```


```python
X, y = make_regression(n_samples=100, n_features=1, noise=1, random_state=18)
y = y.reshape(100,1)

scatter_data(X,y)
```


![png]({{ "/images/test_notebook_3_0.png" }})


```python
X = X.reshape(100,)
y = y.reshape(100,)
n = X.shape[0]
```

Let's test out MathJax!:

Given our objective function 

$$S\left(\hat{\alpha}, \hat{\beta} \right ) = \min_{\hat{\alpha}, \hat{\beta}} \sum_{i=1}^{n} \left ( \varepsilon_{i} \right )^{2}$$ 

we can solve for $$\alpha$$ and $$\beta$$ as follows:


$$S\left(\hat{\alpha}, \hat{\beta} \right ) =  \min_{\hat{\alpha}, \hat{\beta}} \sum_{i=1}^{n} \left (y_{i} - \hat{\alpha} - \hat{\beta}\cdot x_{i} \right)^2$$


$$\frac{\partial S\left(\hat{\alpha}, \hat{\beta} \right )}{\partial\alpha} = -2 \sum_{i=1}^{n} \left (y_{i} - \hat{\alpha} - \hat{\beta}\cdot x_{i} \right) = 0$$

$$\hat{\alpha }= \bar{y} - \hat{\beta} \cdot \bar{x}$$


$$\frac{\partial S\left(\hat{\alpha}, \hat{\beta} \right )}{\partial\beta} = -2 \sum_{i=1}^{n} \left (y_{i} - \hat{\alpha} - \hat{\beta}\cdot x_{i} \right)\cdot x_{i} = 0$$


$${\hat {\beta }}={\frac {\sum {x_{i}y_{i}}-{\frac {1}{n}}\sum {x_{i}}\sum {y_{i}}} {\sum {x_{i}^{2}}-{\frac {1}{n}}(\sum {x_{i}})^{2}}} = {\frac {\operatorname {Cov} [x,y]}{\operatorname {Var} [x]}}$$

## <font color=k> scikit-learn Implementation (OLS) </font>


```python
X = X.reshape(-1,1)
lr = LinearRegression().fit(X,y)
print("intercept_: {:.4f} coef_: {:.4f}".format(lr.intercept_, lr.coef_[0]))
```

<p style="font-family:Monaco, Menlo, Consolas, Courier New, DotumChe, monospace;font-size:16px;"> intercept_: -0.0605 coef_: 4.1123 </p> 



```python
fig, ax = plt.subplots(figsize=(8,6))
ax.scatter(X,y, c='dodgerblue', label="Toy Data")
x = np.linspace(X.min(), X.max(), num=X.shape[0])
lr_pred = lr.intercept_ + lr.coef_*x
ax.plot(x, lr_pred, c='black', linewidth=2, label='predictions', alpha=0.7)
ax.set(xlabel="x1", ylabel="y") 
ax.xaxis.label.set_size(16)
ax.yaxis.label.set_size(16)
ax.set_title("scikit-learn LinearRegression Model Predictions", fontsize=14)
ax.legend()
```





![png]({{ "/images/test_notebook_11_1.png" }})

