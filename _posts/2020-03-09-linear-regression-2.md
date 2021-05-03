---
layout: post
mathjax: true
title: Linear regression 2
subtitle: The multivariate case
#cover-img: /assets/img/path.jpg
#thumbnail-img: /assets/img/linear-regression.png
#share-img: /assets/img/path.jpg
tags: [linear regression, machine learning, linear algebra]
# share-title: true
---

The discussion in the [previous post]({% post_url 2020-03-08-linear-regression-1 %}) can be generalized to the case where a linear model

$$\begin{align*}
\boldsymbol{g}_{\boldsymbol{\Omega}}\colon \mathbb{R}^n & \to \mathbb{R}^m\\
\boldsymbol{x} & \mapsto \boldsymbol{\Omega}^T \boldsymbol{x}
\end{align*}$$

should be estimated. In this case, $$\boldsymbol{\Omega}$$ is an $$\left(n, m\right)$$ matrix and the $$i$$-th component of $$\boldsymbol{g}_{\boldsymbol{\Omega}}$$ can be seen as the scalar map $$g_{\boldsymbol{\Omega}_i}$$, where $$\boldsymbol{\Omega}_i$$ is the $$i$$-th column of $$\boldsymbol{\Omega}$$. Namely,

$${\boldsymbol{g}_{\boldsymbol{\Omega}}}_i\left(\boldsymbol{x}\right) = g_{\boldsymbol{\Omega}_i}\left(\boldsymbol{x}\right) = {\boldsymbol{\Omega}_i}^T \boldsymbol{x}\,.$$

In particular, given a sample $$\left(\boldsymbol{x}_1, \boldsymbol{y}_1\right), \ldots, \left(\boldsymbol{x}_k, \boldsymbol{y}_k\right)$$, the estimation of a column of $$\boldsymbol{\Omega}$$ is an independent problem, i.e., it does not depend on the other columns. Indeed, we wish to solve $$m$$ optimization problems

$$\min_{\boldsymbol{\Omega}_i} L\left(\boldsymbol{\Omega}_i\right) = \min_{\boldsymbol{\Omega}_i} \sum_s \left(g_{\boldsymbol{\Omega}_i}\left(\boldsymbol{x}_s\right)-{\boldsymbol{y}_s}_i\right)^2 = \min_{\boldsymbol{\Omega}_i} \sum_s \left({\boldsymbol{\Omega}_i}^T \boldsymbol{x}_s - {\boldsymbol{y}_s}_i\right)^2\,.$$

The independence property implies that these are equivalent to

$$\min_{\boldsymbol{\Omega}} H\left(\boldsymbol{\Omega}\right) := \min_{\boldsymbol{\Omega}} \sum_i L\left(\boldsymbol{\Omega}_i\right) = \min_{\boldsymbol{\Omega}} \sum_{i, s} \left(g_{\boldsymbol{\Omega}_i}\left(\boldsymbol{x}_s\right)-{\boldsymbol{y}_s}_i\right)^2 = \min_{\boldsymbol{\Omega}} \sum_{i, s} \left({\boldsymbol{\Omega}_i}^T \boldsymbol{x}_s - {\boldsymbol{y}_s}_i\right)^2\,.$$

If we let $$\boldsymbol{X}$$ and $$\boldsymbol{Y}$$ be the input and output sample matrices, respectively, that is:

$$\boldsymbol{X} :=
\begin{pmatrix}
{\boldsymbol{x}_1}^T\\
{\boldsymbol{x}_2}^T\\
\vdots\\
{\boldsymbol{x}_k}^T
\end{pmatrix},\quad
\boldsymbol{Y} :=
\begin{pmatrix}
{\boldsymbol{y}_1}^T\\
{\boldsymbol{y}_2}^T\\
\vdots\\
{\boldsymbol{y}_k}^T
\end{pmatrix},$$

this minimization problem can be written as

$$\min_{\boldsymbol{\Omega}} { {\| \boldsymbol{X} \boldsymbol{\Omega} - \boldsymbol{Y} \|}_F }^2\,,$$

where $${\| \cdot \|}_F$$ is the *Frobenius norm*, defined for matrices as the square root of the sum of the squares of its entries. Moreover, we know the gradient of each independent subproblem:

$$\nabla_{\boldsymbol{\Omega}_i} L\left(\boldsymbol{\Omega}_i\right) = 2 \boldsymbol{X}^T \left( \boldsymbol{X} \boldsymbol{\Omega}_i - {\boldsymbol{Y}_i}\right)\,,$$

where $$\boldsymbol{Y}_i$$ is the $$i$$-th column of $$\boldsymbol{Y}$$, and we can arrange the full gradient in an $$\left(n, m\right)$$ matrix by stacking these $$m$$ gradients in columns:

$$\nabla_{\boldsymbol{\Omega}} H\left(\boldsymbol{\Omega}\right) := \begin{pmatrix} \nabla_{\boldsymbol{\Omega}_1} L\left(\boldsymbol{\Omega}_1\right) & \cdots & \nabla_{\boldsymbol{\Omega}_m} L\left(\boldsymbol{\Omega}_m\right) \end{pmatrix} = 2 \boldsymbol{X}^T \left(\boldsymbol{X} \boldsymbol{\Omega} - \boldsymbol{Y}\right)\,.$$

This particular arrangement is convenient because the $$\left(i, j\right)$$-th entry of the gradient corresponds to the partial derivative of $$H$$ with respect to the $$\left(i, j\right)$$-th entry of $$\boldsymbol{\Omega}$$, namely,

$$\frac{\partial H}{\partial \boldsymbol{\Omega}_{ij}}\left(\boldsymbol{\Omega}\right) = {\nabla_{\boldsymbol{\Omega}} H\left(\boldsymbol{\Omega}\right)}_{ij}\,.$$

This is useful for learning rules based on gradient descent because one can update the parameters in a natural way:

$$\boldsymbol{\Omega}^{(i+1)} \leftarrow \boldsymbol{\Omega}^{(i)} - \alpha \nabla_{\boldsymbol{\Omega}} H\left(\boldsymbol{\Omega}^{(i)}\right)\,.$$

Nevertheless, we can determine the solution analytically, as before:

$$\hat{\boldsymbol{\Omega}} := {\left(\boldsymbol{X}^T\boldsymbol{X}\right)}^+\boldsymbol{X}^T\boldsymbol{Y}\,.$$

Note that this expression is the same as the one found in the solution to the problem discussed in the previous post, which can be regarded as a particular case, when $$m = 1$$.
