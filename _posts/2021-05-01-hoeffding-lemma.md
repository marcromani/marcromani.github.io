---
layout: post
mathjax: true
title: A short proof of Hoeffding's lemma
subtitle: Modulo a factor of 2 
#cover-img: /assets/img/path.jpg
#thumbnail-img: /assets/img/linear-regression.png
#share-img: /assets/img/path.jpg
tags: [machine learning, statistics, statistical learning theory]
share-title: true
last-updated: 2021-05-03
---

Hoeffding's lemma is an elementary yet pivotal result in empirical process theory that allows the construction of exponential measure concentration bounds such as Hoeffding's inequality. The standard proof of the lemma does not require sophisticated tools but, in my opinion, it's not very elegant either. In fact, the proof uses an ad hoc argument based on the minimization of a function via differentiation; see, for example, [Mohri, Mehryar, et al. *Foundations of Machine Learning*](https://mitpress.ublish.com/ereader/7093/?preview=#page/437) or [Wikipedia](https://www.wikiwand.com/en/Hoeffding's_lemma). An alternative proof appears in [Maxim Raginsky's notes](http://maxim.ece.illinois.edu/teaching/fall14/notes/concentration.pdf) but is somewhat convoluted. On the other hand, a very nice proof of a weaker version of the lemma can be found in [John Duchi's notes](http://cs229.stanford.edu/extra-notes/hoeffding.pdf).

The following proof of Hoeffding's lemma builds on Duchi's approach but, to my knowledge, was never presented before. It is similar to the latter in that a symmetrization technique is employed but does not invoke a Rademacher variable. Since it is not necessary to bound the moment-generating function of such variable—a key step in that proof—a tighter bound is obtained. In this case, we are off by a factor of 2 instead of 4.

We start with the usual assumptions that $$\mathbb{P}(a \leq X \leq b) = 1$$ and $$\mathbb{E}(X) = 0$$. Note that this implies[^1] $$\text{Var}(X) \leq \frac{\left(b-a\right)^2}{4}$$. Let $$X'$$ be an i.i.d. random variable, then

$$\mathbb{E}_X\!\left(e^{\lambda X}\right) = \mathbb{E}_X\!\left(e^{\lambda (X - \mathbb{E}_{X'}(X'))}\right) \leq \mathbb{E}_{X, X'}\!\left(e^{\lambda (X - X')}\right)\,.$$

The bound on the right is an application of Jensen's inequality and can also be derived directly[^2]. Let $$Y := X - X'$$, of course, $$\mathbb{P}(a-b \leq Y \leq b-a) = 1$$ and $$\mathbb{E}(Y) = 0$$. The key observation is that, in fact, since $$Y$$ is symmetric $$\mathbb{E}(Y^r) = 0$$ for every odd $$r$$. By Taylor's theorem, for every $$\lambda \in \mathbb{R}$$ and $$y \in [a-b, b-a]$$ there exists $$\varepsilon \in [a-b, b-a]$$ such that

$$e^{\lambda y} = 1 + \lambda y + \frac{\lambda^2}{2} y^2 + \frac{\lambda^3}{6} y^3 e^{\lambda \varepsilon} \leq 1 + \lambda y + \frac{\lambda^2}{2} y^2 + \frac{\lambda^3}{6} y^3 e^{2|\lambda| (b-a)}\,.$$

It follows that

$$\begin{align*}
\mathbb{E}\!\left(e^{\lambda Y}\right) &\leq 1 + \lambda \mathbb{E}(Y) + \frac{\lambda^2}{2} \mathbb{E}(Y^2) + \frac{\lambda^3}{6} \mathbb{E}(Y^3)e^{2|\lambda| (b-a)}\\
&= 1 + \frac{\lambda^2}{2}\text{Var}(Y)\\
&= 1 + \lambda^2 \text{Var}(X)\\
&\leq 1 + \frac{\lambda^2 (b-a)^2}{4}\\
&\leq \exp\left(\frac{\lambda^2 (b-a)^2}{4}\right)\,.
\end{align*}$$

This concludes the proof.

## Footnotes

<p></p>

[^1]: If $$\mathbb{P}(0 \leq Y \leq c) = 1$$ then $$\mathbb{E}(Y^2) \leq c\mathbb{E}(Y)$$, so $$\text{Var}(Y) = \mathbb{E}(Y^2) - \mathbb{E}(Y)^2 \leq \mathbb{E}(Y)(c - \mathbb{E}(Y))$$. Whatever the value of the expectation is, the function $$\gamma(c-\gamma)$$ has a maximum at $$\gamma = \frac{c}{2}$$, so $$\text{Var}(Y) \leq \frac{c^2}{4}$$. If $$\mathbb{P}(a \leq X \leq b) = 1$$ let $$Y := X - a$$, then $$\mathbb{P}(0 \leq Y \leq b-a) = 1$$ and $$\text{Var}(X) = \text{Var}(Y) \leq \frac{(b-a)^2}{4}$$.

[^2]: If $$\mathbb{E}(Z)$$ is finite then

    $$\mathbb{E}\!\left(e^Z\right) = \mathbb{E}\!\left(e^{(Z-\mathbb{E}(Z)+\mathbb{E}(Z))}\right) = e^{\mathbb{E}(Z)}\mathbb{E}\!\left(e^{(Z-\mathbb{E}(Z))}\right) \geq e^{\mathbb{E}(Z)}\mathbb{E}(1 + Z - \mathbb{E}(Z)) = e^{\mathbb{E}(Z)}\,.$$

    The inequality holds because $$e^t \geq 1+t$$ for every $$t \in \mathbb{R}$$.
