---
layout: post
mathjax: true
title: Confidence intervals
subtitle: A brief note
#cover-img: /assets/img/path.jpg
#thumbnail-img: /assets/img/linear-regression.png
#share-img: /assets/img/path.jpg
tags: [statistics]
# share-title: true
last-updated: 2021-04-29
---

The frequentist confidence interval is a fundamental tool in statistics that provides a way to make inferences about the values of parameters of interest whenever there is a (quantifiable) degree of uncertainty in the studied process. However, it is common to have misconceptions about how it should be interpreted. We'll try to elucidate this on a very simple setting provided by our good old Gaussian distribution. Say you have a bunch of data that you know is distributed as a Gaussian. Specifically, you have $$n$$ observations $$x_1, \ldots, x_n$$ drawn independently from a Gaussian distribution $$N(\mu, \sigma^2)$$, and you know $$\sigma$$ but you don't know $$\mu$$. How do you estimate $$\mu$$? Since $$\mu$$ is the expectation of this distribution a natural estimate is

$$\overline{x} := \frac{1}{n}\sum_{i = 1}^n x_i\,.$$

This estimate will be better the larger $$n$$ is. But why is that so? And in what sense would the estimate be better had we a larger sample? Note that having $$n$$ observations drawn independently from a random variable $$X \sim P$$ is the same as having one observation of the random vector $$\boldsymbol{X} := (X_1, \ldots, X_n)$$, where each component is independent of the others and distributed as $$X$$ (i.e., $$X_i \sim P$$). Indeed, the scalar $$x_i$$ can be seen as a realization of $$X_i$$, and the vector $$(x_1, \ldots, x_n)$$ as a realization of $$\boldsymbol{X}$$. We can define

$$\overline{X} := \frac{1}{n}\sum_{i = 1}^n X_i\,,$$

where the sum is not over fixed scalar values (the observations) but random variables, so that $$\overline{X}$$ is also a random variable. It has its own distribution and so on. In particular,

$$E(\overline{X}) = \frac{1}{n}\sum_{i = 1}^n E(X_i) = \frac{1}{n}\sum_{i = 1}^n \mu = \mu\,.$$

This already answers the two questions above: $$\overline{x}$$ can be regarded as an estimate of $$\mu$$ because, in fact, it is an observation of $$\overline{X}$$—that is, if you draw $$n$$ values from $$X$$ and you compute their mean the result is distributed according to the distribution of $$\overline{X}$$—and you can make $$\overline{x}$$ arbitrarily close to $$\mu$$—with probability $$1$$—provided that $$n$$ is large enough, by the strong law of large numbers that applies to $$\overline{X}$$. But we can say even more about $$\overline{X}$$, it is distributed as a Gaussian itself:

$$\overline{X} \sim N(\mu, \frac{\sigma^2}{n})\,.$$

It holds that

$$\overline{X} \sim N(\mu, \frac{\sigma^2}{n}) \iff \frac{\overline{X} - \mu}{\frac{\sigma}{\sqrt{n}}} \sim N(0, 1)\,,$$

and given $$\alpha \in (0, 1)$$ we can find scalars $$a, b$$ such that

$$P(a < \frac{\overline{X} - \mu}{\frac{\sigma}{\sqrt{n}}} < b) = \alpha\,.$$

Since $$N(0, 1)$$ is symmetric around the origin we can always pick $$b \geq 0$$, $$a = -b$$. If we write $$c_\alpha := b$$ (to emphasize the dependency on $$\alpha$$) it holds:

$$\begin{align*}
\alpha &= P(-c_\alpha < \frac{\overline{X} - \mu}{\frac{\sigma}{\sqrt{n}}} < c_\alpha)\\
&= P(\overline{X} - \frac{c_\alpha \sigma}{\sqrt{n}} < \mu < \overline{X} + \frac{c_\alpha \sigma}{\sqrt{n}})\\
&= P(\mu \in (\overline{X} - \frac{c_\alpha \sigma}{\sqrt{n}}, \overline{X} + \frac{c_\alpha \sigma}{\sqrt{n}}))\,.
\end{align*}$$

Note that the only thing that is random in the expression above is $$\overline{X}$$, and certainly not $$\mu$$ (which is unknown, but fixed). The random interval

$$\boldsymbol{I}_{\alpha} := (\overline{X} - \frac{c_\alpha \sigma}{\sqrt{n}}, \overline{X} + \frac{c_\alpha \sigma}{\sqrt{n}})$$

is a random vector called a confidence interval for $$\mu$$, at confidence level $$\alpha$$. The correct way to interpret such construction is the following: The probability that an observation of $$\boldsymbol{I}_{\alpha}$$ contains $$\mu$$ is $$\alpha$$. Equivalently, the probability that we draw $$n$$ values from $$X$$ and build a realization of $$\boldsymbol{I}_{\alpha}$$, namely,

$$(\overline{x} - \frac{c_\alpha \sigma}{\sqrt{n}}, \overline{x} + \frac{c_\alpha \sigma}{\sqrt{n}})\,,$$

that contains $$\mu$$ is $$\alpha$$. *A confidence interval says something about our procedure to make estimations using the data and nothing about the parameters to be estimated themselves* (or about what happens with a particular observation of the confidence interval). For instance, if after drawing $$n$$ values from $$X$$ we construct a realization of the $$\alpha$$-confidence interval it either contains $$\mu$$ or it does not, and we can't possibly know it. It is not true that the probability that $$\mu$$ is in that particular observed interval is $$\alpha$$. All we know is that our procedure provides, with probability $$\alpha$$, realizations of $$\boldsymbol{I}_\alpha$$ that capture the parameter. That's still very useful because we can fix $$\alpha$$ in advance so that when we make an experiment and end up with a particular observation of the random interval we are as confident as we need to be that it really contains $$\mu$$. Note that, as an abuse of language, we also call any particular realization of $$\boldsymbol{I}_\alpha$$ a confidence interval, but the distinction between the two should be clear by now. The length of the interval is

$$\frac{2 c_\alpha \sigma}{\sqrt{n}}\,,$$

so it gets smaller as $$n$$ gets larger, and it gets larger when $$\alpha$$ does, because $$(-c_\alpha, c_\alpha)$$ has to cover more area under $$N(0, 1)$$. For instance, if you want to estimate $$\mu$$ with at least $$d > 0$$ correct decimal digits you require that

$$\frac{c_\alpha \sigma}{\sqrt{n}} < 10^{-d}\,,$$

so you need

$$n > 100^{d} \, c_\alpha^2 \, \sigma^2$$

observations. As an exercise, I drew a sample of $$20$$ values from a Gaussian distribution and built a confidence interval at confidence level $$0.9$$ for $$\mu$$. I repeated this process $$50$$ times and made this figure:

![Confidence interval sampling](https://user-images.githubusercontent.com/14805305/103790058-04e2c980-5041-11eb-9fd9-b97c80767766.png){: .center width="550px"}

As you can see, in $$7$$ of the runs the obtained interval does not contain $$\mu$$. Any of them could be an actual experiment embedded in another process that relies on the constructed interval. Also, note that at this confidence level you would expect around $$5$$ of the $$50$$ runs not to provide an appropriate interval, so the results are consistent. You can explore and download the notebook used to make this figure [here](https://gist.github.com/marcromani/b9f3e70ce0e8b93822c0c032356ae114#file-confidence_intervals-ipynb).
