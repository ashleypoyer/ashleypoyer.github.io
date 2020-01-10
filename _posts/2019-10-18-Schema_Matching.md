---
layout: post
title:  "Schema Matching"
mathjax: true
date:   2019-10-18 
description: description here.
categories: [Data Integration]
---

<style>

.highlight_info {
  width:300px;
  margin: auto;
  background-color: #FFFFCC;
}

p.code {
  font-family: Courier, monospace;
  font-size:17px;
}

p, blockquote, ul, ol, dl, li, table, pre {
margin: 15px 0;}

table { font-size: 16px; width: 800px;
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
      text-align: center;
      margin: 0;
      padding: 2px 5px;
      font-size: 16px;}
    table tr td {
      border: 1px solid #cccccc;
      text-align: left;
      margin: 0;
      padding: 2px 5px; }
    table tr th :first-child, table tr td :first-child {
      margin-top: 0; }
    table tr th :last-child, table tr td :last-child {
margin-bottom: 0; }

</style>

<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>


Suppose we are given schemas $$S = \left \{s_0, s_1, s_2, \ldots s_n \right \}$$ and $$T = \left \{t_0, t_1, t_2, \ldots t_m \right \}$$. 

*Note: $$S$$ and $$T$$ refer to the set of elements (attributes) in the schemas.*

<br>

The elements of $$S$$ can be enumerated as:

<br>

$$s_{i} \in S \quad i \in \left \{0, 1, 2, \ldots n \right \}$$ 

<br>

The same can be done for the elements of $$T$$: 

<br>

$$t_{j} \in T \quad  j \in \left \{ 0, 1, 2, \ldots m \right \}$$

<br>

The **goal** in **schema matching** is to find the elements in $$S$$ and $$T$$ which are a match, 
that is, find $$s_i$$ and $$t_j$$ s.t. $$\left (s_i, t_j \right ) = \text{match}$$. This is done by creating a **matcher** which takes us input schemas $$S$$ and $$T$$ and outputs a **similarity matrix** which holds a **similarity score** $$\forall \left (s_i, t_j \right )$$ pairs.

<br>

$$\longrightarrow \texttt{matcher}(S,T) = \texttt{Similarity Matrix}$$

<br>

$$
\begin{gather*}
\texttt{Similarity Matrix} =
\begin{bmatrix}
\texttt{sim}_{00} & \texttt{sim}_{01} & \ldots & \texttt{sim}_{0m}\\ 
\texttt{sim}_{10} & \texttt{sim}_{11} &  \ldots & \texttt{sim}_{1m}\\ 
\vdots & \vdots &  \ddots & \vdots \\
\texttt{sim}_{n0} & \texttt{sim}_{n1} & \ldots & \texttt{sim}_{nm}
\end{bmatrix} \quad \text{where} & \texttt{sim}_{ij} = \texttt{sim score}(s_i, t_j)
\end{gather*}
$$

<br>

We can then establish some **threshold** $$t$$ so that $$(s_i, t_j) = \text{match}$$ i.f.f $$\text{simscore}(s_i, t_j)$$ >= $$t$$

<br>

```python
import numpy as np
import pandas as pd 
```

## K-grams


```python
def K_grams(X, k):
    X = X.center((len(X)+2)) # add space at the begining and end of the string
    kgrams = []
    for i, x in enumerate(X):
        if len(X[i:]) >= k:
            kgrams.append(X[i:i+k])
    return(kgrams)
```

## Jaccard Similarity 


```python
def jaccard_sim(X, Y, k):
    X_k = set(K_grams(X, k))
    Y_k = set(K_grams(Y, k))
    XinterY = X_k.intersection(Y_k)
    XunionY = X_k.union(Y_k)
    return(len(XinterY )/len(XunionY))
```

<br> For this schema matching example we will be using two different types of matchers: **name-based** and **instance-based**. We will be using three different datasets. The first dataset will be called `chicago` and contains demographic information on the different neighborhoods in Chicago. Its schema is: 

<br> `{'CommunityAreaNumber', 'CommunityAreaName', 'PercentofHousingCrowded', 'PercentHouseholdsBelowPoverty', 'PercentAged16OverUnemployed', 'PercentAged25PlusWithoutHighSchoolDiploma', 'PercentAgedUnder18orOver64 ', 'PerCapitaIncome', 'HardshipIndex'}`

<br> The second dataset will be called `zipcodes` and it contains zipcodes for every neighborhood in Chicago. Its schema is:

<br> `{'CommunityArea', 'Zipcode'}`

<br> The third dataset will be called `schools` and it contains some basic information about the public schools in Chicago like geographical location, level of attendance, test scores, etc. Its schema is:

<br> `{'School ID', 'Name of School', 'Street Address', 'City', 'State', 'Zipcode', 'Phone Number', , 'NWEA Reading Growth Percentile All Grades', ..., 'NWEA Reading Growth Percentile Grade 3', 'Suspensions Per 100 students 2013','One-Year Drop Out Rate Percentage 2013',  'X Coordinate', 'Y Coordinate', 'Longitude', 'Latitude', 'Location'}`

<br> *Note:* This example is to illustrate the concept of schema matching so it is obvious what the matches should be; however, in practice schemas will be far more complex and as a result more complex matchers will be needed.


#### Reading in the datasets:


```python
chicago = pd.read_csv("data/ChicagoDataPortal.csv")
chicago.columns = ['CommunityAreaNumber', 'CommunityAreaName',
       'PercentofHousingCrowded', 'PercentHouseholdsBelowPoverty',
       'PercentAged16OverUnemployed',
       'PercentAged25PlusWithoutHighSchoolDiploma',
       'PercentAgedUnder18orOver64 ', 'PerCapitaIncome', 'HardshipIndex']
zipcodes = pd.read_csv("data/Zipcodes.csv", header=None)
zipcodes.columns = ['CommunityArea', 'Zipcode']
schools = pd.read_csv("data/chicagopublicschools2014.csv")
```


## Name-based Matchers 

Name-based matchers use the **names** of the elements in each schema to determine a match. Each name is a string so we can use a string similarity metric to calculate the similarity score.

<br> Thus for name-based matchers we have the following setup, 
<br>
<br>

$$\forall s_i \in S \quad s_i = \text{string}$$ 

and

$$\forall t_j \in T \quad t_j = \text{string}$$

<br>

$$sim score(s_i, t_j) = JaccardSimilarity(s_{ik}, t_{jk}) = \frac{\left | s_{ik} \cap t_{jk} \right |}{\left | s_{ik} \cup t_{jk} \right |}$$

*Note:* $$s_{ik}$$ and $$t_{jk}$$ refer to the k-gram (also known as n-gram) sets of the respective elements. For more information on k-grams and string similarity you can refer to my earlier [<u>notebook</u>]({% post_url 2019-10-06-String_Similarity_Algorithms%}) on string similarity algorithms 

```python
def name_based(S, T, k=8):
    n = len(S.columns)
    m = len(T.columns)
    Sim = np.zeros((n,m))
    s_i = []
    t_j = []

    for i in range(n):
        for j in range(m):
            s_i.append(S.columns[i])
            t_j.append(T.columns[j])
            score = jaccard_sim(S.columns[i], T.columns[j], k=k)
            Sim[i, j] = score
    return(Sim, s_i, t_j)
```


```python
""" function to retrive the similarity matrix of a matcher and format
the results into an easily readable dataframe """

def matcher_results(matcher, S, T, threshold=0.3):
    Sim, s_i, t_j = matcher(S, T)
    sim_scores = Sim.ravel()
    match = np.where(sim_scores > threshold, 1, 0)
    data = {'s_i': s_i, 't_j': t_j, 'sim score': sim_scores, 'match':match}
    df = pd.DataFrame(data, columns = data.keys())
    return Sim, df
```

Let $$S$$ = `chicago` and $$T$$ = `zipcodes`


```python
Sim, df = matcher_results(name_based, S=chicago, T=zipcodes)
Sim
```

<p class="code">array([[0.46666667, 0.        ], <br>
&emsp; &emsp; &emsp; &emsp;[0.53846154, 0.    ], <br>
&emsp; &emsp; &emsp; &emsp;[0.        , 0.      &emsp; &emsp; &emsp; &emsp;], <br>
&emsp; &emsp; &emsp; &emsp;[0.        , 0.      &emsp; &emsp; &emsp; &emsp;], <br>
&emsp; &emsp; &emsp; &emsp;[0.        , 0.      &emsp; &emsp; &emsp; &emsp;], <br>
&emsp; &emsp; &emsp; &emsp;[0.        , 0.      &emsp; &emsp; &emsp; &emsp;], <br>
&emsp; &emsp; &emsp; &emsp;[0.        , 0.      &emsp; &emsp; &emsp; &emsp;], <br>
&emsp; &emsp; &emsp; &emsp;[0.        , 0.      &emsp; &emsp; &emsp; &emsp;], <br>
&emsp; &emsp; &emsp; &emsp;[0.        , 0.      &emsp; &emsp; &emsp; &emsp;]])
</p>


```python
df.head()
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
      <th>s_i</th>
      <th>t_j</th>
      <th>sim score</th>
      <th>match</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CommunityAreaNumber</td>
      <td>CommunityArea</td>
      <td>0.466667</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CommunityAreaNumber</td>
      <td>Zipcode</td>
      <td>0.000000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CommunityAreaName</td>
      <td>CommunityArea</td>
      <td>0.538462</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CommunityAreaName</td>
      <td>Zipcode</td>
      <td>0.000000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PercentofHousingCrowded</td>
      <td>CommunityArea</td>
      <td>0.000000</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



```python
df[df.match == 1]
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
      <th>s_i</th>
      <th>t_j</th>
      <th>sim score</th>
      <th>match</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CommunityAreaNumber</td>
      <td>CommunityArea</td>
      <td>0.466667</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CommunityAreaName</td>
      <td>CommunityArea</td>
      <td>0.538462</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Let $$S$$ = `zipcodes` and $$T$$ = `schools`


```python
Sim, df = matcher_results(name_based, S=zipcodes, T=schools)
Sim
```

<p class="code">
    array([[0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , <br>
    &emsp; &emsp; &emsp; &emsp; 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , <br>
    &emsp; &emsp; &emsp; &emsp; 0.    , 0.    , 0.    , 0.    , 0.    , 0.1875, 0.    , 0.    , <br>
    &emsp; &emsp; &emsp; &emsp; 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , <br>
    &emsp; &emsp; &emsp; &emsp; 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , <br>
    &emsp; &emsp; &emsp; &emsp; 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , <br>
    &emsp; &emsp; &emsp; &emsp; 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , <br>
    &emsp; &emsp; &emsp; &emsp; 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , <br>
    &emsp; &emsp; &emsp; &emsp; 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , <br>
    &emsp; &emsp; &emsp; &emsp; 0.    , 0.    ], <br>
    &emsp; &emsp; &emsp; &emsp;[0.    , 0.    , 0.    , 0.    , 0.    , 1.    , 0.    , 0.    , <br>
    &emsp; &emsp; &emsp; &emsp; 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , <br>
    &emsp; &emsp; &emsp; &emsp; 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , <br>
    &emsp; &emsp; &emsp; &emsp; 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , <br>
    &emsp; &emsp; &emsp; &emsp; 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , <br>
    &emsp; &emsp; &emsp; &emsp; 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , <br>
    &emsp; &emsp; &emsp; &emsp; 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , <br>
    &emsp; &emsp; &emsp; &emsp; 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , <br>
    &emsp; &emsp; &emsp; &emsp; 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , <br>
    &emsp; &emsp; &emsp; &emsp; 0.    , 0.    ]])
</p>



```python
df.head()
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
      <th>s_i</th>
      <th>t_j</th>
      <th>sim score</th>
      <th>match</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CommunityArea</td>
      <td>School ID</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CommunityArea</td>
      <td>Name of School</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CommunityArea</td>
      <td>Street Address</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CommunityArea</td>
      <td>City</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CommunityArea</td>
      <td>State</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[df.match == 1]
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
      <th>s_i</th>
      <th>t_j</th>
      <th>sim score</th>
      <th>match</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>79</th>
      <td>Zipcode</td>
      <td>Zipcode</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.isnull(chicago).sum()
chicago = chicago.drop(pd.isnull(chicago).any(1).nonzero()[0])
#pd.isnull(zipcodes).sum()
#pd.isnull(schools).sum()
col_to_drop = np.where(pd.isnull(schools).sum() > 40)[0]
schools = schools.drop(schools.columns[col_to_drop], axis=1)
schools = schools.drop(pd.isnull(schools).any(1).nonzero()[0])
```


```python
chicago = chicago.astype(str)
zipcodes = zipcodes.astype(str)
schools = schools.astype(str)
```

# Instance-based Matchers 

Instance-based matchers determine the similarity between the sets of **data instances** of each element. Then we can measure the similarity of the data instances by calculating the **overlap** of the sets.

<br> For instance-based matchers we have the following setup, 

<br>
$$\text{Let} \left | s_i \right | = p  \text{ and }  \left | t_j \right | = q$$


$$\forall s_i \in S$$

$$\quad s_i = \left \{ d_{ik} \right \} \quad k \in \left \{0, 1, 2, \ldots p \right \}$$

$$\left \{ d_{ik} \right \} = \left \{ d_0, d_1, d_2, \ldots, d_p \right \} $$

<br>

$$\forall t_j \in T$$

$$\quad t_j = \left \{ d_{jl} \right \} \quad l \in \left \{0, 1, 2, \ldots q \right \}$$

$$\left \{ d_{jl} \right \} = \left \{ d_0, d_1, d_2, \ldots, d_q \right \} $$


* $$p$$ is the number of data instances in $$s_i$$
* $$q$$ is the number of data instances in $$t_j$$
* $$d_{ik}$$ is the set of data instances for $$s_i$$ where $$k$$ is the index on $$s_i$$
* $$d_{jl}$$ is the set of data instances for $$t_j$$ where $$l$$ is the index on $$t_j$$

<br>

$$sim score(s_i, t_j) = Overlap(d_{ik}, d_{jl}) = \frac{\left | d_{ik} \cap d_{jl} \right |}{ min \left ( \left | d_{ik} \right |, \left | d_{jl} \right | \right )}$$

<br>

```python
def overlap_measure(d_i, d_j):
    inter = d_i.intersection(d_j)
    cardinality_inter = len(inter)
    overlap = cardinality_inter/min(len(d_i), len(d_j))
    return(overlap)
```


```python
def instance_based(S, T):
    s_i_order = []
    t_j_order = []
    n = len(S.columns)
    m = len(T.columns)
    Sim = np.zeros((n,m))

    for i in range(n):
        for j in range(m):
            s_i_order.append(S.columns[i])
            t_j_order.append(T.columns[j])
            d_i = set(S[S.columns[i]]) # data instances of elem s_i
            d_j = set(T[T.columns[j]]) # data instances of elem t_j
            overlap = overlap_measure(d_i, d_j)
            Sim[i, j] = overlap

    return(Sim, s_i_order, t_j_order)  
```

Let $$S$$ = `chicago` and $$T$$ = `zipcodes`


```python
Sim, df = matcher_results(instance_based, S=chicago, T=zipcodes)
Sim
```

<p class="code">
array([[0.        , 0.&emsp; &emsp; &emsp; &emsp;      ], <br>
&emsp; &emsp; &emsp; &emsp;[0.36363636, 0.        ], <br>
&emsp; &emsp; &emsp; &emsp;[0.        , 0.&emsp; &emsp; &emsp; &emsp;         ], <br>
&emsp; &emsp; &emsp; &emsp;[0.        , 0.&emsp; &emsp; &emsp; &emsp;         ], <br>
&emsp; &emsp; &emsp; &emsp;[0.        , 0.&emsp; &emsp; &emsp; &emsp;         ], <br>
&emsp; &emsp; &emsp; &emsp;[0.        , 0.&emsp; &emsp; &emsp; &emsp;         ], <br>
&emsp; &emsp; &emsp; &emsp;[0.        , 0.&emsp; &emsp; &emsp; &emsp;         ], <br>
&emsp; &emsp; &emsp; &emsp;[0.        , 0.&emsp; &emsp; &emsp; &emsp;         ], <br>
&emsp; &emsp; &emsp; &emsp;[0.        , 0.&emsp; &emsp; &emsp; &emsp;         ]])

</p>


```python
df.head()
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
      <th>s_i</th>
      <th>t_j</th>
      <th>sim score</th>
      <th>match</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CommunityAreaNumber</td>
      <td>CommunityArea</td>
      <td>0.000000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CommunityAreaNumber</td>
      <td>Zipcode</td>
      <td>0.000000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CommunityAreaName</td>
      <td>CommunityArea</td>
      <td>0.363636</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CommunityAreaName</td>
      <td>Zipcode</td>
      <td>0.000000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PercentofHousingCrowded</td>
      <td>CommunityArea</td>
      <td>0.000000</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[df.match == 1]
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
      <th>s_i</th>
      <th>t_j</th>
      <th>sim score</th>
      <th>match</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>CommunityAreaName</td>
      <td>CommunityArea</td>
      <td>0.363636</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Let $$S$$ = `zipcodes` and $$T$$ = `schools`


```python
Sim, df = matcher_results(instance_based, S=zipcodes, T=schools)
Sim
```


<p class="code">
    array([[0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , <br>
&emsp; &emsp; &emsp; &emsp; 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , <br>
&emsp; &emsp; &emsp; &emsp; 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , <br>
&emsp; &emsp; &emsp; &emsp; 0.  ], <br>
&emsp; &emsp; &emsp; &emsp; [0.  , 0.  , 0.  , 0.  , 0.  , 0.76, 0.  , 0.  , 0.  , 0.  , 0.  , <br>
&emsp; &emsp; &emsp; &emsp; 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , <br>
&emsp; &emsp; &emsp; &emsp; 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , <br>
&emsp; &emsp; &emsp; &emsp; 0.  ]])
</p>


```python
df.head()
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
      <th>s_i</th>
      <th>t_j</th>
      <th>sim score</th>
      <th>match</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CommunityArea</td>
      <td>School ID</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CommunityArea</td>
      <td>Name of School</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CommunityArea</td>
      <td>Street Address</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CommunityArea</td>
      <td>City</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CommunityArea</td>
      <td>State</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[df.match == 1]
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
      <th>s_i</th>
      <th>t_j</th>
      <th>sim score</th>
      <th>match</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>39</th>
      <td>Zipcode</td>
      <td>Zipcode</td>
      <td>0.76</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## Combining Matchers 


<h3> $$\mathit{combined(i, j)} = \bigg[ \sum_{m=1}^{k} \mathit{matcherScore}(m, i, j) \bigg] / k$$ </h3>


* $$\mathit{matcherScore(m, i, j)}$$ is the similarity score between $$s_i$$ and $$t_j$$ as produced by the $$m$$<sup>th</sup> matcher

<br>

```python
matchers = [name_based, instance_based]

def combined(S, T, matchers):
    k = len(matchers)
    n = len(S.columns)
    m = len(T.columns) # m here is the number of elements in T
    matcher_scores = np.zeros((n*m,1))
    
    for m in range(k): # now m is the number of matchers 
        scores = matchers[m](S, T)[0].ravel()
        matcher_scores = np.append(matcher_scores, scores[:, np.newaxis], axis=1)
        combined_scores = np.sum(matcher_scores, axis=1)/k
        
    return(combined_scores)
```



```python
""" 
function to retrieve the corresponding s_i, t_j elements for each score
since combined() did not store it
""" 
def order(S,T):
    s_i = []
    t_j = []
    for i in range(len(S.columns)):
        for j in range(len(T.columns)):
            s_i.append(S.columns[i])
            t_j.append(T.columns[j])
    return(s_i, t_j)

def combined_results(S, T, matchers, threshold=0.4):
    combined_scores = combined(S, T, matchers).ravel()
    match = np.where(combined_scores > threshold, 1, 0)
    s_i, t_j = order(S, T)
    data = {'s_i': s_i, 't_j': t_j, 'combined score': combined_scores, 'match': match}
    df = pd.DataFrame(data, columns = data.keys())
    return df
```

Let $$S$$ = `chicago` and $$T$$ = `zipcodes`


```python
df = combined_results(S=chicago, T=zipcodes, matchers=matchers)
df.head()
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
      <th>s_i</th>
      <th>t_j</th>
      <th>combined score</th>
      <th>match</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CommunityAreaNumber</td>
      <td>CommunityArea</td>
      <td>0.233333</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CommunityAreaNumber</td>
      <td>Zipcode</td>
      <td>0.000000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CommunityAreaName</td>
      <td>CommunityArea</td>
      <td>0.451049</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CommunityAreaName</td>
      <td>Zipcode</td>
      <td>0.000000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PercentofHousingCrowded</td>
      <td>CommunityArea</td>
      <td>0.000000</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[df.match == 1]
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
      <th>s_i</th>
      <th>t_j</th>
      <th>combined score</th>
      <th>match</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>CommunityAreaName</td>
      <td>CommunityArea</td>
      <td>0.451049</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


Let $$S$$ = `zipcodes` and $$T$$ = `schools`

```python
df = combined_results(S=zipcodes, T=schools, matchers=matchers)
df.head()
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
      <th>s_i</th>
      <th>t_j</th>
      <th>combined score</th>
      <th>match</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CommunityArea</td>
      <td>School ID</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CommunityArea</td>
      <td>Name of School</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CommunityArea</td>
      <td>Street Address</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CommunityArea</td>
      <td>City</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CommunityArea</td>
      <td>State</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[df.match == 1]
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
      <th>s_i</th>
      <th>t_j</th>
      <th>combined score</th>
      <th>match</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>39</th>
      <td>Zipcode</td>
      <td>Zipcode</td>
      <td>0.88</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


