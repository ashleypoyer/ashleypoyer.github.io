---
layout: post
title:  "String Similarity Algorithms"
mathjax: true
date:   2019-10-06 
description: description here.
categories: [Miscellaneous]
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


img[alt=ngrams] {width: 500px; display: block; margin: 0 auto;}

img[alt=edit_dist] { width: 500px; display: block; margin: 0 auto;}

</style>

<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>



## N-grams

<br>

<!-- 
<img src="data/figure5-1.jpg" height="400" width="400"> *Figure displaying the N-grams concept [1]* </img> -->

![ngrams]({{ "/images/figure5-1.jpg"}})

*Figure displaying the N-grams concept [1]*

<br> 

```python
""" This implementation adds a space to the beginning and end of the string before 
extracting the n-grams but the concept illustrated above remains the same. """

def N_grams(X, n):
    X = X.center((len(X)+2)) # adding spaces
    ngrams = []
    for i, x in enumerate(X):
        if len(X[i:]) >= n:
            ngrams.append(X[i:i+n])
    return(ngrams)
```
<br>

### N-grams Example 

Suppose we have two strings $$X$$ = `"Apple Corporation, CA"` and $$Y$$ = `"Apple Corp, CA"`. The corresponding **N-grams** for `n = 3` for each string would be:

```python
X = "Apple Corporation, CA"
Y = "Apple Corp, CA"

X_n = N_grams(X, n=3)
Y_n = N_grams(Y, n=3)

print("N-grams for string X: \n")
for ngram in X_n:
    print("'" + ngram + "'", end=" ")
    
print("\n")  
print("N-grams for string Y: \n")
for ngram in Y_n:
    print("'" + ngram + "'", end=" ")
```

<p class="code"> 
N-grams for string X: <br> <br>
' Ap' 'App' 'ppl' 'ple' 'le ' 'e C' ' Co' 'Cor' 'orp' 'rpo' 'por' 'ora' 'rat' 'ati' 'tio' 'ion' 'on,' 'n, ' ', C' ' CA' 'CA ' 
</p>   
 
<br>

<p class="code">  N-grams for string Y: <br> <br>
' Ap' 'App' 'ppl' 'ple' 'le ' 'e C' ' Co' 'Cor' 'orp' 'rp,' 'p, ' ', C' ' CA' 'CA ' 
</p>
  

## Jaccard Similarity <br>

To determine how similar two strings $$X$$ and $$Y$$ are we can transform $$X$$ and $$Y$$ into their respective N-gram sets. Then we can use a set similarity metric like **Jaccard Similarity** to measure the similarity between the two sets, that is, the similarity between corresponding strings $$X$$ and $$Y$$.

<br> *Formulation:*
<br>

$$Jaccard Similarity(X_n, Y_n) =  \frac{\left | X_n \cap Y_n \right |}{\left | X_n \cup Y_n \right |}$$

<br>
* $$X_n$$ is the set of N-grams for string $$X$$
* $$Y_n$$ is the set of N-grams for string $$Y$$

<br>

```python
def jaccard_sim(X, Y, n):
    X_n = set(N_grams(X, n))
    Y_n = set(N_grams(Y, n))
    XinterY = X_n.intersection(Y_n)
    XunionY = X_n.union(Y_n)
    return(len(XinterY )/len(XunionY))
```

### Jaccard Similarity Example 


```python
X = 'Apple Corporation, CA'
Y = 'Apple Corp, CA'

print("Jaccard Similarity: {:.4f}".format(jaccard_sim(X, Y, n=3)))
```

<p class="code"> Jaccard Similarity: 0.5217</p>
    

Notice how the Jaccard Similiarty between $$X$$ and $$Y$$ changes as the value of $$n$$ changes.


```python
for i in range(1,6):
    print("JaccSim(X, Y) for n = {}: {:.2f}".format(i, jaccard_sim(X,Y, n=i)))
```

<p class="code"> JaccSim(X, Y) for n = 1: 0.69 <br>   
JaccSim(X, Y) for n = 2: 0.62 <br>
JaccSim(X, Y) for n = 3: 0.52 <br>
JaccSim(X, Y) for n = 4: 0.43 <br>
JaccSim(X, Y) for n = 5: 0.35 <br>
</p>


## Edit (Levenshtein) Distance 

The **Edit Distance** between $$X$$ and $$Y$$ is the minimum number of edits to string $$X$$ to be transformed into string $$Y$$.

Suppose the length of $$X = n$$, in other words, $$X$$ is composed of $$n$$ characters. Additionally, suppose the length of $$Y = m$$, that is, $$Y$$ has $$m$$ characters. We can define $$X$$ and $$Y$$ in terms of their characters as follows:

- $$X = x_0 x_1 x_2 \ldots x_n$$ <br>
- $$Y = y_0 y_1 y_2 \ldots y_m$$. 

*Note:* Going forward we will use $$i$$ to index $$X$$ and $$j$$ to index $$Y$$.


The algorithm works by considering all possible length-i prefixes of $$X$$ and length-j prefixes of $$Y$$ with the following recurrence equations (we will use a matrix to store the values):


### Initialization 

* $$d(i, 0) = i$$ <br>
* $$d(0, j) = j$$ <br> 


### Recurrence Equations 

\\( \quad \quad  \quad \quad \quad \quad \forall i \in \\) { \\( 0, 1, 2, \ldots n \\) } 
<br> 

\\( \quad \quad \quad \quad  \quad \quad  \quad \quad  \quad \forall j \in \\) { \\( 0, 1, 2, \ldots m \\) } 


$$
\quad \quad \quad d(i, j) = \left\{
        \begin{array}{ll}
             d(i-1, j-1) + c(x_i, y_j)  \\
             d(i-1, j) + 1 \\
             d(i, j-1) + 1
        \end{array}
    \right.
$$

<br>   

$$
c(x_i, y_j) = \left\{
    \begin{array}{ll}
            0  & \quad  x_i = y_j \\
            1 & \quad \text{otherwise}
    \end{array}
\right.
$$
    
<br> 

$$d(n, m)$$ will be the **Edit Distance** between $$X$$ and $$Y$$.

### Example

<!-- 
<img src="data/edit_dist_example.png" height="300" width="300"> *The arrows refer to the optimal alignment of the strings so those can be ignored* </img> -->

![edit_dist]({{ "/images/edit_dist_example.png"}})

*The arrows refer to the optimal alignment of the strings so those can be ignored*

### Implementation with NumPy!


```python
import numpy as np
```


```python
def edit_distance(X, Y):
    n = len(X)
    m = len(Y)
    dist_matrix = np.zeros((n + 1, m + 1), dtype=int)
    first_row = np.arange(m+1)
    first_col = np.arange(n+1)
    # initialzie the first row and col in the matrix
    dist_matrix[0,:] = first_row
    dist_matrix[:,0] = first_col
    
    for i in range(1,n+1):
        for j in range(1,m+1):
            upper = dist_matrix[i-1,j]
            left = dist_matrix[i,j-1]
            upper_left = dist_matrix[i-1, j-1]
            
            if X[i-1] == Y[j-1]:
                delta = 0
                dist_matrix[i,j] = min(upper, left, upper_left) + delta
                    
            else:
                delta = 1
                dist_matrix[i,j] = min(upper, left, upper_left) + delta
    return(dist_matrix[n,m], dist_matrix)
```

*Another Note:* `delta` refers to \\( \rightarrow \delta(c(x_i, y_j)) \\)

<br>

### Algorithm in Action


```python
x = 'dva'
y = 'dave'

edit_dist, dist_matrix = edit_distance(x, y)
print("Distance Matrix: \n {}".format(dist_matrix))
print("\n")
print("Edit Disance: {}".format(edit_dist))
```

<p class="code">
Distance Matrix: <br>
[[0 1 2 3 4] <br>
[1 0 1 2 3] <br>
[2 1 1 1 2] <br>
[3 2 1 2 2]] <br>
</p>
    
    
<p class="code"> Edit Disance: 2 </p>


## Edit Similarity 


$$EditSimilarity(X, Y) = 1 - \frac{ EditDistance(X, Y) }{ max(length(X), length(Y))}$$ 


```python
def edit_similarity(X,Y):
    edit_dist = edit_distance(X,Y)[0]
    return(1-(edit_dist/max(len(X), len(Y))))
```


```python
X = 'Apple Corporation, CA'
Y = 'Apple Corp, CA'

print("Edit Similarity: {:.4f}".format(edit_similarity(X, Y)))
```

<p class="code">  Edit Similarity: 0.6667 </p>


## References  <br>

[1] Gustavsson, J. (1996). *Figure 5-1*. [Image] Text Categorization Using Acquaintance. http://plaza.ufl.edu/jgu/public_html/C-uppsats/cup.html
