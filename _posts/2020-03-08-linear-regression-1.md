---
layout: post
mathjax: true
title: Linear regression 1
subtitle: A first look
#cover-img: /assets/img/path.jpg
#thumbnail-img: /assets/img/linear-regression.png
#share-img: /assets/img/path.jpg
tags: [linear regression, machine learning, linear algebra]
share-title: true
---

This is one of the simplest examples of a machine learning approach to solve a particular problem. Assume that you are trying to model an unknown process $$f\colon \mathbb{R}^n \to \mathbb{R}$$ and that you have collected a bunch of input/output samples $$\left(\boldsymbol{x}_1, y_1\right), \ldots, \left(\boldsymbol{x}_k, y_k\right)$$ from that process, that is, $$y_i = f\left(\boldsymbol{x}_i\right)$$ for all $$i$$. More generally, it could also be assumed that the data may be corrupted by some form of noise. For example, in the presence of additive noise, $$y_i = f\left(\boldsymbol{x}_i\right) + n_i$$ for all $$i$$, where $$n_i$$ is a random (does not depend on $$f$$) unknown scalar. Under the further assumption that the unknown process $$f$$ can be modeled as a linear map

$$\begin{align*}
g_{\boldsymbol{\omega}}\colon \mathbb{R}^n & \to \mathbb{R}\\
\boldsymbol{x} & \mapsto \boldsymbol{\omega}^T \boldsymbol{x}\,,
\end{align*}$$

linear regression—by *least squares*—aims to find $$\boldsymbol{\omega}$$ such that the sum of the squared residuals $$L$$ is minimized, i.e., solve the optimization problem

$$\min_{\boldsymbol{\omega}} L\left(\boldsymbol{\omega}\right) := \min_{\boldsymbol{\omega}} \sum_i \left(g_{\boldsymbol{\omega}}\left(\boldsymbol{x}_i\right)-y_i\right)^2 = \min_{\boldsymbol{\omega}} \sum_i \left(\boldsymbol{\omega}^T \boldsymbol{x}_i - y_i\right)^2\,.$$

In words, we are trying to estimate the parameters $$\boldsymbol{\omega}$$ (and hence the function $$g_{\boldsymbol{\omega}}$$) such that the output $$g_{\boldsymbol{\omega}}\left(\boldsymbol{x}_i\right)$$ of the estimated function when we feed it $$\boldsymbol{x}_i$$ is as close as possible (in the squared difference sense) to the associated observed output $$y_i$$, across all available samples. In an upcoming post we'll see a principled justification as to why we minimize, generally, the sum of the squared residuals and not any other quantity (e.g., the sum of the absolute value of the residuals). If we define matrices $$\boldsymbol{X}$$ and $$\boldsymbol{Y}$$ as the matrices that contain the $$k$$ input and output samples as rows, respectively, that is:

$$\boldsymbol{X} :=
\begin{pmatrix}
{\boldsymbol{x}_1}^T\\
{\boldsymbol{x}_2}^T\\
\vdots\\
​{\boldsymbol{x}_k}^T
\end{pmatrix},\quad
\boldsymbol{Y} :=
\begin{pmatrix}
y_1\\
y_2\\
\vdots\\
y_k
\end{pmatrix},$$

then the minimization problem above can be written as

$$\min_{\boldsymbol{\omega}} {\| \boldsymbol{X} \boldsymbol{\omega} - \boldsymbol{Y} \|}^2 = \min_{\boldsymbol{\omega}} {\left(\boldsymbol{X} \boldsymbol{\omega} - \boldsymbol{Y}\right)}^T \left(\boldsymbol{X} \boldsymbol{\omega} - \boldsymbol{Y}\right)\,.$$

Such matrix $$\boldsymbol{X}$$, whose rows correspond to the observed explanatory variables, is called the *design matrix*. Since $$L$$ is convex the fact that $$\nabla_{\boldsymbol{\omega}} L\left(\boldsymbol{\omega}\right) = 0$$ is both a necessary and sufficient condition for $$\boldsymbol{\omega}$$ to be a (possibly non-unique) global minimum. It holds that

$$\frac{\partial L}{\partial \omega_j}\left(\boldsymbol{\omega}\right) = \boldsymbol{e_j}^T\boldsymbol{X}^T\left(\boldsymbol{X}\boldsymbol{\omega} - \boldsymbol{Y}\right) + \left(\boldsymbol{\omega}^T\boldsymbol{X}^T - \boldsymbol{Y}^T\right)\boldsymbol{X}\boldsymbol{e_j} = 2 \boldsymbol{e_j}^T\boldsymbol{X}^T\left(\boldsymbol{X}\boldsymbol{\omega} - \boldsymbol{Y}\right)\,,$$

where $$\boldsymbol{e_j}$$ is the $$j$$-th standard basis (column) vector, so

$$\nabla_{\boldsymbol{\omega}} L\left(\boldsymbol{\omega}\right) = 2 \boldsymbol{X}^T {\left(\boldsymbol{X} \boldsymbol{\omega} - \boldsymbol{Y}\right)}\,.$$

The minimum of $$L$$ is achieved whenever $$2 \boldsymbol{X}^T {\left(\boldsymbol{X} \boldsymbol{\omega} - \boldsymbol{Y}\right)} = 0$$ which is equivalent to

$$\boldsymbol{X}^T\boldsymbol{X}\boldsymbol{\omega} = \boldsymbol{X}^T\boldsymbol{Y}\,.$$

The $$n$$ equations on $$\omega_1, \ldots, \omega_n$$ defined by this matrix equation are known as the *normal equations*. If $$\boldsymbol{X}^T\boldsymbol{X}$$ is invertible the (unique) solution is given by

$$\boldsymbol{\omega} = {\left(\boldsymbol{X}^T\boldsymbol{X}\right)}^{-1}\boldsymbol{X}^T\boldsymbol{Y}\,.$$

Nevertheless, a solution always exists regardless of the rank of $$\boldsymbol{X}^T\boldsymbol{X}$$. That is, $$\boldsymbol{X}^T\boldsymbol{Y} \in \text{Im}\left(\boldsymbol{X}^T\boldsymbol{X}\right)$$, which is equivalent to say that the system is compatible for arbitrary matrices $$\boldsymbol{X}$$ and $$\boldsymbol{Y}$$. We'll prove this result in the [appendix](#appendix) of the post—it shouldn't be surprising though, since, intuitively, a vector $$\boldsymbol{\omega}$$ that minimizes an Euclidean distance should always exist, right? In particular, if $$\text{rank}\left(\boldsymbol{X}^T\boldsymbol{X}\right) < n$$ the system has infinitely many solutions. In this case, we say that the model is *non-identifiable*. In general, the solution can be written as

$$\boldsymbol{\omega} = \left(\boldsymbol{X}^T\boldsymbol{X}\right)^g\boldsymbol{X}^T\boldsymbol{Y}\,.$$

The notation $$\boldsymbol{A}^g$$ denotes a *generalized inverse* of the matrix $$\boldsymbol{A}$$, which is any matrix such that $$\boldsymbol{A} \boldsymbol{A}^g \boldsymbol{A} = \boldsymbol{A}$$. It can be shown that such a matrix always exists (even for non-square matrices), and it is easy to check that (a) provided that a solution to $$\boldsymbol{A}\boldsymbol{x} = \boldsymbol{b}$$ actually exists, $$\boldsymbol{A}^g\boldsymbol{b}$$ is a solution, and that (b) if $$\boldsymbol{A}$$ is indeed invertible, $$\boldsymbol{A}^g = \boldsymbol{A}^{-1}$$. The most commonly used generalized inverse is the so-called *Moore–Penrose pseudo-inverse*, denoted by $$\boldsymbol{A}^+$$. This matrix has some additional desirable properties—which, for instance, guarantee its uniqueness—but note that the single condition given above is enough to provide a solution. The vector

$$\hat{\boldsymbol{\omega}} := \left(\boldsymbol{X}^T\boldsymbol{X}\right)^+\boldsymbol{X}^T\boldsymbol{Y}$$

is the canonical solution to the least squares problem.

Note that we can further generalize the learning setting to accommodate nonlinear functions. Namely, given a (possibly nonlinear) function

$$\begin{align*}
\boldsymbol{h}\colon \mathbb{R}^n & \to \mathbb{R}^d\\
\boldsymbol{x} & \mapsto \boldsymbol{h}\left(\boldsymbol{x}\right) = \left(h_1\left(\boldsymbol{x}\right), \ldots, h_d\left(\boldsymbol{x}\right)\right)\,,
\end{align*}$$

find $$\boldsymbol{\omega} \in \mathbb{R}^d$$ that minimizes

$$L_{\boldsymbol{h}}\left(\boldsymbol{\omega}\right) := \sum_i \left(\left(g_{\boldsymbol{\omega}} \circ \boldsymbol{h}\right)\left(\boldsymbol{x}_i\right)-y_i\right)^2 = \sum_i \left(\boldsymbol{\omega}^T \boldsymbol{h}\left(\boldsymbol{x}_i\right) - y_i\right)^2\,.$$

Clearly, $$g_{\boldsymbol{\omega}} \circ \boldsymbol{h}$$ is still linear in $$\boldsymbol{\omega}$$ and we can replace $$\boldsymbol{x}_i$$ by $$\boldsymbol{h}\left(\boldsymbol{x}_i\right)$$ (and $$n$$ by $$d$$) in the discussion above. The scalar functions $$h_i$$ are called *basis functions* and allow us to broaden the class of functions where we pick a solution from.

## Let's code

We'll simulate a bunch of noisy data collected from a one-dimensional hidden process

$$f(x) = 100\sin(x) - 2x^2 + 200\cos^2(x) + 70x\,,$$

and we'll try to fit a fifth degree polynomial to it. That is, we'll augment our one-dimensional input samples with basis functions $$\{1, x^2, x^3, x^4, x^5\}$$. The code in the provided notebook below should be self-explanatory.

{% gist f4efb5e12bed97e03e553b5f0ae8484d example_1d.ipynb %}

Feel free to download the notebook and play with it.

## Appendix

​​We want to show that for arbitrary matrices $$\boldsymbol{X}$$ and $$\boldsymbol{Y}$$ there exists $$\boldsymbol{\omega}$$ such that $$\boldsymbol{X}^T\boldsymbol{X} \boldsymbol{\omega} = \boldsymbol{X}^T\boldsymbol{Y}$$, i.e., $$\boldsymbol{X}^T\boldsymbol{Y} \in \text{Im}\left(\boldsymbol{X}^T\boldsymbol{X}\right)$$. Note that, trivially, $$\boldsymbol{X}^T\boldsymbol{Y} \in \text{Im}\left(\boldsymbol{X}^T\right)$$. We'll show that $$\text{Im}\left(\boldsymbol{X}^T\right) = \text{Im}\left(\boldsymbol{X}^T\boldsymbol{X}\right)$$. Indeed, we have:

$$\text{Im}\left(\boldsymbol{X}^T\boldsymbol{X}\right) = \{\boldsymbol{X}^T\boldsymbol{X}\boldsymbol{x} \mid \boldsymbol{x} \in \mathbb{R}^n\} = \{\boldsymbol{X}^T\boldsymbol{y} \mid \boldsymbol{y} \in \text{Im}\left(\boldsymbol{X}\right)\}\,,$$

which means that there exists a surjective map

$$\begin{align*}
\boldsymbol{\phi}\colon \text{Im}\left(\boldsymbol{X}\right) & \to \text{Im}\left(\boldsymbol{X}^T\boldsymbol{X}\right)\\
\boldsymbol{y} & \mapsto \boldsymbol{X}^T\boldsymbol{y}\,.
\end{align*}$$

Since $$\text{Im}\left(\boldsymbol{X}\right) = \text{Ker}\left(\boldsymbol{X}^T\right)^\perp$$, it holds that $$\text{Im}\left(\boldsymbol{X}\right) \cap \text{Ker}\left(\boldsymbol{X}^T\right) = \boldsymbol{0}$$, which implies that $$\boldsymbol{\phi}$$ is also injective. As a consequence, $$\dim\text{Im}\left(\boldsymbol{X}^T\right) = \dim\text{Im}\left(\boldsymbol{X}\right) = \dim\text{Im}\left(\boldsymbol{X}^T\boldsymbol{X}\right)$$. Since $$\text{Im}\left(\boldsymbol{X}^T\boldsymbol{X}\right) \subseteq \text{Im}\left(\boldsymbol{X}^T\right)$$, we conclude that $$\text{Im}\left(\boldsymbol{X}^T\right) = \text{Im}\left(\boldsymbol{X}^T\boldsymbol{X}\right)$$.
