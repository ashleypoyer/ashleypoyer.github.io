---
layout: post
title:  "Naive Bayes Classifiers"
mathjax: true
date:   2019-09-22 
description: How Naive Bayes classifiers work and implementing some common variants, Gaussian, Bernoulli, and Multinomial. Using term-frequency (tf) and term frequency-inverse document frequency (tf-idf) vectors with Multinomial Naive Bayes. Lastly, an example on classifying Amazon product reviews.
categories: [Machine Learning]
---

<style>

p.code {
  /* font-family:Monaco, Menlo, Consolas, Courier New, DotumChe, monospace; */
  font-family: Courier, monospace;
  font-size:17px;
}


p, blockquote, ul, ol, dl, li, table, pre {
margin: 15px 0;}

table { font-size: 14px; width: 800px;
  padding: 0; }
  table tr {
    border-top: 1px solid #cccccc;
    background-color: white;
    margin: 0;
    padding: 0; }
    table tr:nth-child(2n) {
      background-color: #f8f8f8; }
      table tr th {
      font-weight: bold;
      border: 1px solid #cccccc;
      text-align: left;
      margin: 0;
      padding: 6px 13px; }
    table tr td {
      border: 1px solid #cccccc;
      text-align: left;
      margin: 0;
      padding: 6px 13px; }
    table tr th :first-child, table tr td :first-child {
      margin-top: 0; }
    table tr th :last-child, table tr td :last-child {
margin-bottom: 0; }


img[alt=norm] { width: 1000px; display: block; margin: 0 auto;}

</style>

<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>


Naive Bayes classifiers are a set of supervised learning algorithms which work by applying **Bayes Theorem**. For example, if we were interested in predicting a class label $$y_j$$ from a set of input features $$\left \{ {x_1, \dots , x_n} \right \}$$ then Bayes Theorem would give us the probability of an observation $$i$$ belonging to class $$y_j$$ given a set of input features $$x_i$$ as:


$$P(y_j \mid x_1, \dots, x_n) = \frac{P(y_j) P(x_1, \dots x_n \mid y_j)} {P(x_1, \dots, x_n)} \hspace{2em}  \text{where $x_i = \left \{ {x_{i1}, \dots , x_{in}} \right \}$}$$


Additionally, Naive Bayes classifiers make a couple big assumptions. The first assumption is that the samples are independent and identically distributed (i.i.d.). The second is a pretty *naive* assumption that the features in $$x_i$$ are conditionally independent of one another. However, under this assumption we can conveniently re-write $$P(x_1, \ldots, x_n \mid y_j)$$ as follows:

$$P(x_i \mid y_j) = P(x_1 \mid y_j) \cdot P(x_2 \mid y_j) \cdot \ldots \cdot P(x_n \mid y_j) =  \prod_{i=1}^{n} P( x_i \mid y_j)$$

Which leaves us with this simplification:
                             
$$P(y_j \mid x_1, \dots, x_n) = \frac{P(y_j) \prod_{i=1}^{n} P(x_i \mid y_j)} {P(x_1, \dots, x_n)}$$  

Furthermore, since $$P(x_1, \ldots, x_n)$$ is simply a scalar we can drop it from the decision rule as it will not affect anything:

$$
\begin{align}
\begin{aligned}P(y_j \mid x_1, \dots, x_n) \propto P(y_j) \prod_{i=1}^{n} P(x_i \mid y_j)\\
\hat{y_j} = {\underset {y_j}{\operatorname {arg\,max} }}\,P(y_j) \prod_{i=1}^{n} P(x_i \mid y_j)
\end{aligned}
\end{align}
$$                          

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
```

##  Gaussian Naive Bayes

For `GaussianNB` the distribution of each feature is assumed to be Gaussian. Hence, the likelihood of each feature $$x_{i}$$ can be desbribed by:

$$P(x_i \mid y) = \frac{1}{\sqrt{2\pi\sigma^2_y}} \exp\left(-\frac{(x_i - \mu_y)^2}{2\sigma^2_y}\right)$$


```python
from sklearn.naive_bayes import GaussianNB

iris = datasets.load_iris()
X, y = iris.data, iris.target
names = iris.feature_names 
names.append('target')
data = pd.DataFrame(np.c_[X,y], columns=names)
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
from scipy.stats import norm

colors =["skyblue", "teal", "olive"]
f, axes = plt.subplots(1, 3, figsize=(15,5), sharex=True)
for i, ax in enumerate(axes.flatten()):
    sns.distplot(data.iloc[:, i] , color=colors[i], ax=ax, fit=norm, kde=False)
```



![png]({{ "/images/NaiveBayesClassifiers_5_1.png" }})



```python
gnb = GaussianNB().fit(X, y)

print("Model Accuracy: {:.2f}".format(gnb.score(X, y)))
```

<p class="code"> Model Accuracy: 0.96 </p>


### Another example on the wine quality dataset:


```python
wine = datasets.load_wine()
X, y = wine.data, wine.target
names = wine.feature_names
names.append('target')
data = pd.DataFrame(np.c_[X,y], columns=names)
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>alcohol</th>
      <th>malic_acid</th>
      <th>ash</th>
      <th>alcalinity_of_ash</th>
      <th>magnesium</th>
      <th>total_phenols</th>
      <th>flavanoids</th>
      <th>nonflavanoid_phenols</th>
      <th>proanthocyanins</th>
      <th>color_intensity</th>
      <th>hue</th>
      <th>od280/od315_of_diluted_wines</th>
      <th>proline</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>14.23</td>
      <td>1.71</td>
      <td>2.43</td>
      <td>15.6</td>
      <td>127.0</td>
      <td>2.80</td>
      <td>3.06</td>
      <td>0.28</td>
      <td>2.29</td>
      <td>5.64</td>
      <td>1.04</td>
      <td>3.92</td>
      <td>1065.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>13.20</td>
      <td>1.78</td>
      <td>2.14</td>
      <td>11.2</td>
      <td>100.0</td>
      <td>2.65</td>
      <td>2.76</td>
      <td>0.26</td>
      <td>1.28</td>
      <td>4.38</td>
      <td>1.05</td>
      <td>3.40</td>
      <td>1050.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13.16</td>
      <td>2.36</td>
      <td>2.67</td>
      <td>18.6</td>
      <td>101.0</td>
      <td>2.80</td>
      <td>3.24</td>
      <td>0.30</td>
      <td>2.81</td>
      <td>5.68</td>
      <td>1.03</td>
      <td>3.17</td>
      <td>1185.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14.37</td>
      <td>1.95</td>
      <td>2.50</td>
      <td>16.8</td>
      <td>113.0</td>
      <td>3.85</td>
      <td>3.49</td>
      <td>0.24</td>
      <td>2.18</td>
      <td>7.80</td>
      <td>0.86</td>
      <td>3.45</td>
      <td>1480.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13.24</td>
      <td>2.59</td>
      <td>2.87</td>
      <td>21.0</td>
      <td>118.0</td>
      <td>2.80</td>
      <td>2.69</td>
      <td>0.39</td>
      <td>1.82</td>
      <td>4.32</td>
      <td>1.04</td>
      <td>2.93</td>
      <td>735.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
gnb = GaussianNB().fit(X, y)

print("Model Accuracy: {:.2f}".format(gnb.score(X, y)))
```

<p class="code"> Model Accuracy: 0.99 </p>


##  Bernoulli Naive Bayes 

For `BernoulliNB` the data is assumed to follow a multivariate Bernoulli distribution. Meaning, each feature is expressed as a binary vector. Thus, we have:

$$P(x_i \mid y) = P(i \mid y)x_i + (1 - P(i \mid y)) (1 - x_i)$$


```python
from sklearn.naive_bayes import BernoulliNB
```


```python
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=200, n_features=2, n_informative=2, n_redundant=0, 
                           random_state=12)
```


```python
fig, ax = plt.subplots(figsize=(9,7))
ax.scatter(X[:,0], X[:,1], c=y, s=60)
```


![png]( {{ "/images/NaiveBayesClassifiers_13_1.png" }})



```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
```


```python
bnb = BernoulliNB(binarize=0.0)
bnb.fit(X_train, y_train)

print(bnb.score(X_train, y_train))
print(bnb.score(X_test, y_test))
```

<p class="code"> 0.9534 <br> 0.9 </p>


### Visualizing Predictions of the BernoulliNB Classifier


```python
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['purple', 'yellow'])

fig, ax = plt.subplots(figsize=(9, 7))
ax.scatter(X_train[:,0], X_train[:,1], c=y_train, s=70, marker='o')
y_pred = bnb.predict(X_test)
ax.scatter(X_test[:,0], X_test[:,1], c=y_pred, s=70, marker='^')

h = 0.01 # step size in the mesh
xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
x_array = np.arange(xmin, xmax, h)
y_array = np.arange(ymin, ymax, h)
xx, yy = np.meshgrid(x_array, y_array)
Z = bnb.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
ax.contourf(xx, yy, Z, cmap=cmap, alpha=0.2)
ax.contour(xx, yy, Z, colors='k', linewidths=0.2)
ax.set(xlabel="Feature 0", ylabel="Feature 1")
```


![png]( {{ "/images/NaiveBayesClassifiers_17_1.png" }})



```python
from sklearn.preprocessing import Binarizer

transformer = Binarizer().fit(X[:,1].reshape(-1,1))

binary_features = transformer.transform(X[:,1].reshape(-1,1))

fig, ax = plt.subplots(figsize=(8,7))
sns.distplot(binary_features, hist=False, color="k")
ax.set(title="Distribution of Feature 1 (Bernoulli)")
ax.set(xlabel="Feature 1")
```



![png]( {{ "/images/NaiveBayesClassifiers_18_1.png" }})


## Multinomial Naive Bayes 

For `MultinomialNB` each feature is assumed to follow a multinomial distribtuion. The distribution is parametrized by vectors $$\theta_y = (\theta_{y1},\ldots,\theta_{yn})$$ where $$\theta_{yi} = {P}(x_i \mid y_j)$$ for each feature $$i$$ given $$n$$ features.


The likelihood of each feature in the feature vector, that is, $${P}(x_i \mid y_j)$$ can be approximated via the maximum likelihood estimate (MLE) which is equivalent to the relative frequency count. Therefore, $${P}(x_i \mid y_j)$$ can be calculated as:


$${P}(x_i \mid y_j) = \frac{N_{x_i, y_j}}{N_{y_j}}$$


* $$N_{x_i, y_j}$$ = the number of times feature $$i$$ appears in samples from class $$y_j$$
* $$N_{y_j}$$ = the total count of all features in class $$y_j$$

<br>

### Additive Smoothing 


**The Zero-Frequency Problem**: If a feature does not exist in the training set $$T$$ then the class conditional probability will be 0 and as a result $${P}(x_i \mid y_j)$$ will be 0. The solution to this problem is called **additive smoothing**. A smoothing term $$\alpha$$ is added to $${P}(x_i \mid y_j)$$ to prevent obtaining a 0 probability for a given feature.


To summarize we have:

$$
\begin{align*}
\theta_y = (\theta_{y1},\ldots,\theta_{yn}) &&\text{$n$ = # of features}
\end{align*}
$$

$$\theta_{yi} = {P}(x_i \mid y_j) = \frac{N_{x_i, y_j}+\alpha}{N_{y_j} + \alpha \, n}  \quad \text{for $i \in \left \{1,2, \ldots, n \right \}$}$$

*Setting $$\alpha = 1$$ is known as Laplace smoothing and setting $$\alpha < 1$$ is known as Lidstone smoothing.*

<br>

# Popular Multinomial NB Use-Case: Text Classification

There are two common use cases for the Multinomial Naive Bayes variant which involve document classifications tasks. The first uses term-frequency vectors and the second uses Term frequency-inverse document frequency vectors. These will be elaborated on in the following sections.

Before classifying any documents one needs to determine which words in the collection of documents are important for the classifier to consider, in other words, what are the informative words that would allow your classifier to best discriminate between the documents? This list of terms is commonly referred to as the **vocabulary** in document classification.

<br>

### Term Frequency (Tf Vectors)

Suppose we are given a vocabulary with $$m$$ terms and a collection $$D$$ containing $$n$$ documents. Then, a tf-vector for a document $$d_i$$ is simply a vector which holds the term frequency $$\forall t_j \in \text{vocab}$$ appearing in document $$d_i$$

To summarize:

$$\text{vocab} = \left \{ t_1, t_2, \cdots, t_m \right \}$$

$$D = {d_i} \quad \text{for}\, i \in \left \{1,2, \cdots, n \right \}$$

$$tf(t_j,d_i) = \textrm{number of times term $t_j$ appears in document $d_i$}$$

$$\big \langle tf_{d_i} \big \rangle = \big \langle tf(t_j,d_i) \big \rangle \quad \text{for}\, j \in \left \{1,2, \cdots, m \right \}$$

The feature matrix $$X$$ for a collection of documents $$D$$ using tf-vectors would be:


$$X = \begin{bmatrix}
\big \langle tf(t_j,d_1) \big \rangle  \\ 
\big \langle tf(t_j,d_2) \big \rangle   \\ 
\vdots \\
\big \langle tf(t_j,d_n) \big \rangle    
\end{bmatrix} \quad \text{note that $X$ is an $n \times m$ matrix}$$

<br>

### Term Frequency - Inverse Document Frequency  (Tf-idf Vectors)

An obvious issue that arises when using simple term frequency counts to represent a document is that documents with higher frequencies of very common words will be inaccurately emphasized. Tf-idf resolves this by weighting each term by its inverse document frequency (idf), that is, how often that term appears in the collection of documents. As a result, the term frequency of terms that are less common will be given a higher weight and the term frequency of more common terms will recieve a smaller weight.

Let the inverse document frequency of some term $$t_j$$ be defined as:

$$idf(t_j) = \log \left ( \frac{N_d}{N_d(t_j)} \right )$$


* $$N_{d}$$ = the number of documents
* $$N_{d}(t_j)$$ = the number of documents that contain term $$t_j$$


Then the Tf-idf vector for a document $$d_i$$ is given by:

$$\text{Tf-idf} = \big \langle tf(t_j,d_i) \cdot idf(t_j) \big \rangle \quad \text{for}\, j \in \left \{1,2, \cdots, m \right \}$$

<br> 

### Example: Classifying Amazon Reviews

```python
amazon_reviews = pd.read_csv("amazon_cells_labelled.csv", header=None)
```


```python
amazon_reviews.head(15)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>So there is no way for me to plug it in here i...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Good case</td>
      <td>Excellent value.</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Great for the jawbone.</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Tied to charger for conversations lasting more...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>The mic is great.</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>I have to jiggle the plug to get it to line up...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>If you have several dozen or several hundred c...</td>
      <td>then imagine the fun of sending each of them ...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>If you are Razr owner...you must have this!</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Needless to say</td>
      <td>I wasted my money.</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>What a waste of money and time!.</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>10</th>
      <td>And the sound quality is great.</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>11</th>
      <td>He was very impressed when going from the orig...</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12</th>
      <td>If the two were seperated by a mere 5+ ft I st...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Very good quality though</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14</th>
      <td>The design is very odd</td>
      <td>as the ear "clip" is not very comfortable at ...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



### A little bit of data cleaning


```python
X = []
y = []

for i, record in enumerate(amazon_reviews.iterrows()):
    for j in range(len(record[1])):
        if (pd.isnull(record[1][j]) == True):
            sentiment = record[1][j-1]
            reviews = record[1][:j-1]
            break
        elif j == (len(record[1]) - 1):
            sentiment = record[1][j]
            reviews = record[1][:j]
    
    reviews = reviews.str.cat(sep='')
    X.append(reviews)
    y.append(sentiment)
```


```python
ar = pd.DataFrame({'review': X, 'sentiment': y}, index=np.arange(len(amazon_reviews.index)))
ar["review"] = ar['review'].str.replace('[^\w\s]','').str.lower()
ar.sentiment = ar.sentiment.astype(int)
```


```python
ar.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>review</th>
      <th>sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>so there is no way for me to plug it in here i...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>good case excellent value</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>great for the jawbone</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>tied to charger for conversations lasting more...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>the mic is great</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>i have to jiggle the plug to get it to line up...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>if you have several dozen or several hundred c...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>if you are razr owneryou must have this</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>needless to say i wasted my money</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>what a waste of money and time</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### Creating the Term Frequency Feature Matrix (`X_tf`) 


```python
reviews = ar.review
y = ar.sentiment

# split reviews into terms
review_words = [review.split() for review in reviews]


vocab = sorted(set(sum(review_words, [])))
vocab_dict = {k:i for i,k in enumerate(vocab)}

X_tf = np.zeros((len(reviews), len(vocab)), dtype=int)

for i, review in enumerate(review_words):
    for word in review:
        X_tf[i, vocab_dict[word]] += 1

X = X_tf
```


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
```


```python
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train, y_train)

print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))
```
<p class="code"> 0.9675 <br> 0.84 </p>


### Creating the Tf-idf Feature Matrix (`X_tfidf`) 


```python
idf = np.log(X_tf.shape[0]/X_tf.astype(bool).sum(axis=0))

X_tfidf = X_tf * idf
X = X_tfidf

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
```


```python
clf = MultinomialNB()
clf.fit(X_train, y_train)

print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))
```

<p class="code"> 0.98625 <br> 0.805 </p>

