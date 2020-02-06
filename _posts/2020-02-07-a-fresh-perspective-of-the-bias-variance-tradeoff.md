---
layout: post
comments: true
title:  "A Fresh Perspective of the Bias-Variance Tradeoff"
excerpt: "I show how to talk about bias and variance of data (instead of model)"
date:   2020-02-07 01:00:00
mathjax: true
---

We all know that the bias-variance tradeoff is a fundamental tradeoff of machine learning models. In this post, I will offer a new and alternative definition for bias and variance: **Instead of talking about the bias and variance of a model, we can speak about the variance and bias of a training data set**. With this definition (made precise below), we can analyze in terms of bias and variance some standard methods used in machine learning, such as increasing the training set size or training set balancing. But first, I will start with the standard approach to bias and variance.

---

Suppose that we have training data $$x_\text{train}$$ with labels $$y_\text{train}$$, which are samples of the random vectors $$X_\text{train}$$ and $$Y_\text{train}$$  which have a joint distribution $$P_{\text{train}}$$ (note that the single vector $$x_\text{train}$$ represents the *entire* training set, which could be comprised, for example, from i.i.d. samples. Similarly, $$y_\text{train}$$ represents the labels of the *entire* training set, and $$P_{\text{train}}$$ is the *joint* distribution of all the examples and labels in the training set, which in the i.i.d. case is simply a multiplication of single-example distributions). We also have a machine learning model $$M$$. The machine learning model takes as input the entire training set $$x_\text{train}$$ and  $$y_\text{train}$$, and returns a function which can be used to predict the $$y$$ of any $$x$$ (here, $$x$$ and $$y$$ represent a *single* example).  Note that $$M$$ describes not only the functional form of the function (e.g. a decision tree, a neural network), but also the optimization process required for fitting the function. For example, $$M$$ can describe a decision tree, fitted with the [C4.5](https://en.wikipedia.org/wiki/C4.5_algorithm) algorithm. We can write the function that $$M$$ returns as $$M(x;x_{\text{train}},y_{\text{train}})$$. The prediction of this function $$M$$ for an input $$x$$ is $$\hat{y}=M(x;x_{\text{train}},y_{\text{train}})$$. The hat over the $$y$$ designates that this is an estimation of the true $$y$$.

The bias-variance formula is most easily written for regression problems with the mean-squared-error criterion, so from now on we will focus on this case. Given an example $$x$$ with a true label $$y$$, the squared error is given by:

$$\Delta(x,y,M,x_{\text{train}},y_{\text{train}})=\left( y-\hat{y} \right)^2=\left( y-M(x;x_{\text{train}},y_{\text{train}}) \right)^2$$

And the mean-squared-error is obtained by taking the mean *using the distribution* $$P_{\text{train}}$$:

$$\Delta(x,y,M,P_{\text{train}})=\underset{X_{\text{train}},Y_{\text{train}}\sim P_{\text{train}}}{\mathbb{E}}\left[ y-M(x;X_{\text{train}},Y_{\text{train}}) \right]^2=$$

$$=y^2-2y\ \mathbb{E}_{\text{train}}\left[M(x;X_{\text{train}},Y_{\text{train}})\right]+\mathbb{E}_{\text{train}}\left[M^2(x;X_{\text{train}},Y_{\text{train}})\right]=$$

$$=y^2-2y\ \mathbb{E}_{\text{train}}\left[M(x;X_{\text{train}},Y_{\text{train}})\right]+\left[\mathbb{E}_{\text{train}}\left[M(x;X_{\text{train}},Y_{\text{train}})\right]\right]^2-\left[\mathbb{E}_{\text{train}}\left[M(x;X_{\text{train}},Y_{\text{train}})\right]\right]^2+\mathbb{E}_{\text{train}}\left[M^2(x;X_{\text{train}},Y_{\text{train}})\right]=$$

$$=(y-\mathbb{E}_{\text{train}}\left[M(x;X_{\text{train}},Y_{\text{train}})\right])^2+\mathbb{E}_{\text{train}}\left[M^2(x;X_{\text{train}},Y_{\text{train}})\right]-\left[\mathbb{E}_{\text{train}}\left[M(x;X_{\text{train}},Y_{\text{train}})\right]\right]^2=$$

$$=\text{bias}^2(x,y,M,P_{\text{train}})+\text{variance}(x,M,P_{\text{train}})$$

where 

$$\text{bias}(x,y,M,P_{\text{train}})=y-\mathbb{E}_{\text{train}}\left[M(x;X_{\text{train}},Y_{\text{train}})\right]$$

$$\text{variance}(x,M,P_{\text{train}})=\mathbb{E}_{\text{train}}\left[M^2(x;X_{\text{train}},Y_{\text{train}})\right]-\left[\mathbb{E}_{\text{train}}\left[M(x;X_{\text{train}},Y_{\text{train}})\right]\right]^2$$

The expectation in these definition is over the training set distribution $$P_{\text{train}}$$ ($$\mathbb{E}_{\text{train}}$$ is a shorthand of $$\underset{X_{\text{train}},Y_{\text{train}}\sim P_{\text{train}}}{\mathbb{E}}$$).

Notice that in these definitions, the bias is a a function of $$x$$ and $$y$$ (as well as the model $$M$$ and the training data distribution $$P_{\text{train}}$$), and the variance is a function of $$x$$ (as well as $$M$$ and $$P_{\text{train}}$$) but not $$y$$. This makes sense, because the variance measures how different the predictions of $$M$$ for $$x$$ are for different realizations of the train set, irrespective of whether these predictions are correct. The bias, on the other hand, does depend on the true label $$y$$, because it measures the mean deviations of the predictions of $$M$$ for $$x$$ with respect the $$y$$.

Remember that for a reasonable model, high variance is usually related to overfitting: If the model is very flexible, it will be able to fit any realizations of the training set very well, and sometimes too well. This overfitting means that for $$x$$ not in the training set, $$M$$ will give nonsensical results, or random results (using the layman's version of "random"). Furthermore, for a different realization of the training set (that is, if you had collected other alternative training data from the same distribution), the overfitting could yield a totally different fitted model, which could make the predictions for the same $$x$$ to be quite different, hence the high variance.

On the other hand, for a reasonable model, low bias is usually related to high flexibility of the model. A flexible model is a model which has the capacity to approximate a large and diverse family of functions. On the other hand, a non-flexible model can represent only a small class of functions (e.g. a linear model can represent only linear functions). This is another way of saying the a non-flexible model has high bias. The bias here is the "presumptions" that the model makes about the relation between $$x$$  and $$y$$.

Although the the bias and variance depend on $$x$$, $$y$$, $$P_\text{train}$$ and $$M$$, we normally think about them as characterizing only a machine learning model $$M$$ (in fact, we could eliminate the dependence on $$P_\text{train}$$ by performing another expectation, over some meta-distribution over all "realistic" training sets. In other words, we can imagine $$P_\text{train}$$ itself being drawn from some distribution over distributions. I would do it here, but I'm afraid it will be be too confusing). I shall refer to this bias and variance by the names **model bias** and **model variance**. For example, we say that a deep decision tree has low model bias and high model variance, while a shallow tree has high model bias and low model variance. We can say, very (*very!*) roughly, that a main goal of a pure machine learning researcher is to find new models $$M$$ that have low model bias and model variance, for generic data. Or, if the researcher is oriented towards computer vision, her goal is to find models with low model bias and model variance for generic image data. For example, I imagine that the [people who invented the random forest model](https://en.wikipedia.org/wiki/Random_forest#History), did not want to solve a specific data problem instance, but rather to find an algorithm that would work for many different datasets (and they certainly succeeded).

Now comes the fresh perspective. Data scientist, or machine learning practitioners, usually aren't interested in developing an algorithm that can be used off-the-shelf for any dataset. Instead, they typically just choose one model among the popular machine learning models, such as a multilayer perceptron (MLP), random forest (RF), or gradient boosting (GDB). In fact, with much oversimplification, I suggest a new view: there is a set of machine learning models $$\{\text{MLP},\text{RF},\text{GDB}...\}$$ (this set is essentially a subset of [scikit-learn's classes](https://scikit-learn.org/stable/modules/classes.html), which is what practitioners normally have at their disposal). The model $$M$$ is, a-priori, drawn from some distribution over this set. (by "a-priori" I mean that this is the distribution without conditioning on the specific data set). We denote this distribution with $$P_\text{model}$$. Notice that in the expressions above for $$\Delta(x,y,M,P_{\text{train}})$$, if instead of taking the expectation over $$X_\text{train},Y_\text{train}$$ we would have taken the expectation over $$M$$, we would have obtained a very similar decomposition into bias and variance:

$$\Delta(x,y,x_{\text{train}},y_{\text{train}},P_\text{model})=\text{bias}^2(x,y,x_{\text{train}},y_{\text{train}},P_\text{model})+\text{variance}(x,x_{\text{train}},y_{\text{train}},P_\text{model})$$

where, just as before:

$$\text{bias}(x,y,x_{\text{train}},y_{\text{train}},P_\text{model})=y-\mathbb{E}_{\text{model}}\left[M(x;x_{\text{train}},y_{\text{train}})\right]$$

$$\text{variance}(x,x_{\text{train}},y_{\text{train}},P_\text{model})=\mathbb{E}_{\text{model}}\left[M^2(x;x_{\text{train}},y_{\text{train}})\right]-\left[\mathbb{E}_{\text{model}}\left[M(x;x_{\text{train}},y_{\text{train}})\right]\right]^2$$

Here, the expectation $$\mathbb{E}_{\text{model}}$$ is a shorthand of $$\underset{M\sim P_{\text{model}}}{\mathbb{E}}$$.  But what do these new expressions mean? Here is how I think about them:

As I said before, we usually think about the bias and variance of a model. But in fact, **we can also talk about the bias and variance of a training set**. I call these **data bias** and **data variance**. Moreover, practitioners typically put a lot of effort into choosing a good training set, which will yield a small generalization error, which means a small data bias and data variance.

In the model bias and model variance expressions, we kept $$M$$ free (not averaged over), since these expressions are from the point of view of a pure machine learning researcher, whose job is to find $$M$$. (in some sense, the researcher is trying to find $$\underset{M}{\arg\min}\ \left[\text{bias}^2(M)+\text{variance}(M)\right]$$ where the dependence on $$P_\text{train}$$ and $$x$$ and $$y$$ was removed by some expectation). And the data bias and data variance expressions are from the eyes of practitioners, whose task (among others) is to construct a good training set, such that when they will apply some off-the-shelf machine learning models, they will get good results.

The following point is quite subtle and could be a source of confusion, so please make sure that you agree with it: when we speak about data variance, we are referring to how the predictions change when the model $$M$$ itself changes, and the training set is kept fixed. Likewise, the data bias is the signed error of the prediction, averaged over different options for $$M$$ without changing the training set. (I am guessing that most treatments of bias and variance, even those that talk about how these are affected by the data - e.g. [this](https://scholar.google.co.il/scholar?cluster=4735590936771959751&hl=en&as_sdt=2005&sciodt=0,5) - talk about model bias and model variance.)

The issue that still needs to be addressed is: what are some concrete example of data bias and data variances of training sets? I will try to give a few crude answers.

If the training set consists of only one example, it will certainly have very high data bias and low data variance. Why? because for any realization of $$M$$ (I mean, any reasonable choice of $$M$$), the function $$M(x;x_{\text{train}},y_{\text{train}})$$ (seen as a function of $$x$$) will be a constant function (thus high bias), returning the same value of $$y_\text{train}$$ for any $$M$$ (thus low variance).

I'm not sure yet how the variance and bias depend on the training set size. When I will figure it out, I will post about it. I do believe, though, that as the number of training samples (sampled i.i.d) increases, the mean squared error should become smaller (when evaluated on a test set drawn from the same distribution).

Other scenarios that can be analyzed in terms of data variance and data bias: unbalanced data and the actions taken to fix it; stratified sampling; increasing the proportions of rare "hard" examples in the training set.

Here is one final example: [Malach and Shalev-Schwartz](https://arxiv.org/abs/1910.11923) have recently pointed out that it is hard to learn with neural networks some functions represented by boolean circuits. However, if the training set distribution is biased to have a non-uniform distribution of zeros and ones, the functions become learnable. The paper discusses only neural networks, and not on other models, so it's hard to say the implications for data bias and data variance. But the idea of biasing the training set can be analyzed in terms of our new framework too.