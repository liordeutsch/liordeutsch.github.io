---
layout: post
comments: true
title:  "About Variational Autoencoders"
excerpt: "I present the formulation of variational autoencoders."
date:   2021-09-08 08:00:00
mathjax: true
---

## About Variational Autoencoders

### Introduction

This is a review of variational autoencoders (VAEs). The emphasis is on the probabilistic formulation, and not so much on specific examples of implementations/applications. 

To follow the review, you should have basic knowledge of probability theory, as well as acquainted with the following concepts: Maximum likelihood, neural networks (NNs), stochastic gradient descent (SGD), backpropagation, bias of an estimator, kullback Leibler (KL) divergence. The review mentions the following terms as well, but knowing them is not essential and the corresponding sections can be skipped: Gaussian mixture model (GMM), expectation-maximization (EM), importance sampling, entropy, maximum-a posteriori (MAP) estimation, autoencoder, autoregressive model. 

In short, VAE is a procedure for fitting a latent variable model that allows easy sampling and posterior inference, in the case of complicated high dimensional distributions with continuous latent variables and a lot of data. In the rest of this review we will unpack what this all means.

### Background

#### Latent Variable Models

We have data $$x_1,..,x_n$$ drawn IID from some unknown distribution, which we will call the "true" distribution. We want to find a distribution that approximates the true distribution as much as possible. In simple, low dimensional cases we can do this by choosing a simple parametric family of distributions (e.g normal distribution), and doing maximum likelihood. 

If the data is very high dimensional and complicated, the true distribution might be very different from one of the simple known distributions (e.g normal). One way to model a complicated distribution is by using a latent variables model (LVM). In these models, we imagine the process of sampling a data point $$x$$ as a two-(or more) step process. First, a latent variable $$z$$ is sampled from some pdf $$p(z)$$, and then $$x$$ is sampled from $$p(x\vert z)$$. This description is also knows as a *generative model*, because it describes how to generate $$x$$ in multiple steps. The most well known example for a LVM is GMM: First $$z$$ is sampled from some finite set $$\{1,2,..,k\}$$ with $$p(z=i)=p_i$$, and then $$x$$ is sampled from a multivariate normal distribution whose mean and covariance are $$\mu_z,\Sigma_z$$.  This gives us a relatively flexible function approximator, because we can tune the parameters $$\{p_i,\mu_i,\Sigma_i\}_{i=1}^k$$  and get multi-modal distributions. Typically for GMM this is done using the EM algorithm. It's important to understand that this is just a model. Our dataset does not include samples of the latent variables $$z$$. Moreover, the true data generating process is not necessarily the two-step process defined here, and the latent variables don't necessarily exist in "reality". This is just a model.

There are a few reasons for why LVMs are good. First, they make sense generatively. For example, if $$x_1,...,x_n$$ are photographs of vehicles, then we can imagine the process of generating a photograph $$x$$ as consisting of first sampling a low-dimensional latent variable $$z$$ which includes information such as the vehicle's type, the lighting, the camera's angle, etc. And then, given $$z$$, the camera and physics sample an image from $$p(x\vert z)$$.

So we see that it's natural to speak about data as being sampled in a two-step processes. However, this only serves as a motivation for using a LVM, and it's important to understand that in the context of this article, we do not attach any sematic meaning to $$z$$, since we don't have any data about it.

Another reason for using LVMs is that it allows modeling complicated distributions using simple building blocks. We've seen this already in the context of GMM, where two simple distribution building blocks are combined to get a multimodal distribution.

Also, LVMs are useful because they may allow us to do posterior inference, in the sense that they allow us to examine or sample or maximize $$z$$ from $$p(z\vert x)$$. This is good because, when $$z$$ is from a smaller dimension than $$x$$, we can obtain a lower dimensional representation of $$x$$ by using $$z$$ instead. This is beneficial for compression or clustering and other tasks.

Lastly, LVMs often make sampling new instances of $$x$$ easy, because we can follow the generative process of sampling $$z$$ from $$p(z)$$ and then sampling $$x$$  from $$p(x\vert z)$$. 

#### Limitations of GMM+EM

We mentioned GMM as an example of an LVM, and we mentioned that EM is used to optimize the GMM model. In some cases, GMM is not good enough. For example, having a finite set for $$z$$ can be a limitation on the flexibility of the resulting distribution. For complicated, high-dimensional distributions, we may want $$z$$ to be a random vector that is itself from a continuous complicated distribution, and the dependence of $$x$$ on $$z$$ can be complicated as well . But these cases are not suitable for EM (at least, in its vanilla form). Here are some of the reasons:

* The E step of EM requires calculating $$p_{\theta_t}(z\vert x)$$, which, according to Bayes is $$\frac{p_{\theta_t}(x\vert z)p_{\theta_t}(z)}{\int p_{\theta_t}(x\vert z')p_{\theta_t}(z')\text{d}z'}$$.  The integral in the denominator is typically intractable. 
* Also, the E step requires calculating a mean $$\underset{z\sim p_{\theta_t}(z\vert x)}{\mathbb{E}}[\cdot\cdot\cdot]$$ which requires doing an integral which can be intractable.
* To fit a complicated distribution, we will need a lot of data. EM needs all the data in each of its iterations, and this can be inefficient when there is a lot of data. 

Other extensions of, and alternatives to GMM+EM exist. The variational autoencoder (VAE) is one of them, designed to allow flexible high dimensional probabilistic modeling - NNs are used to model the distributions - with a continuous latent space, that scales to large data sets.

#### Variational Inference

VAE is an algorithm which uses the larger framework of *variational inference* (VI). We will now give a very brief introduction to VI. In VI, there is a LVM that consists of $$p(z)$$ and  $$p(x\vert z)$$.  We refer to $$p(z)$$ as the *prior* (notice that this is not a prior over parameters, it is a prior over the values of the latent variable $$z$$). Notice that by specifying these two distributions, we don't have anymore freedom to also specify the *posterior* $$p(z\vert x)$$, because it is implied by the other two using Bayes theorem: $$p(z\vert x)=\frac{p(x\vert z)p(z)}{\int p(x\vert z')p(z') \text{d}z'}$$. However, calculating the posterior may be hard due to the integral in the denominator. Therefore, given a specific $$x=x_0$$, instead of calculating the its posterior $$p(z\vert x_0)$$, we will find a function $$q(z)$$ that approximates it. We call it the *approximate posterior* or the *variational distribution* (we write $$q(z)$$ instead of $$q(z\vert x_0)$$ because, in this basic setting, $$x_0$$ is considered fixed and we want to find the posterior function only for $$x_0$$. That is, we want to find a function of only $$z$$). We first take some parametric family of functions $$q_\varphi(z)$$ with parameters $$\varphi$$, and we search for a value $$\varphi^\ast$$ which will yield $$q_{\varphi^\ast(z)}\approx p(z\vert x_0)$$. To do this, we attempt to minimize the KL-divergence between $$q_\varphi(z)$$ and $$p(z\vert x_0)$$:

$$D_\text{KL}(q_\varphi\vert \vert p)=\underset{z\sim q_\varphi}{\mathbb{E}}\log\frac{q_\varphi(z)}{p(z\vert x_0)}$$. Using Bayes rule, we get:

$$D_\text{KL}(q_\varphi\vert \vert p)=\underset{z\sim q_\varphi}{\mathbb{E}}\log\frac{q_\varphi(z)}{p(x_0\vert z)p(z)/p(x_0)}=$$

$$=\underset{z\sim q_\varphi}{\mathbb{E}}[\log q_\varphi(z)-\log p(x_0\vert z)-\log p(z)+\log p(x_0)]$$.

The last term does not depend on $$z$$, so it can be taken out of the expectation, and we get:

$$D_\text{KL}(q_\varphi\vert \vert p)=\underset{z\sim q_\varphi}{\mathbb{E}}[\log q_\varphi(z)-\log p(x_0\vert z)-\log p(z)]+\log p(x_0)=$$

$$=-\mathcal{L}(\varphi;x_0)+\log p(x_0)$$

where we defined $$\mathcal{L}(\varphi;x_0)=\underset{z\sim q_\varphi}{\mathbb{E}}[\log p(x_0\vert z)+\log p(z)-\log q_\varphi(z)]$$. 

The last term in the expression for the KL divergence does not depend on $$\varphi$$. Therefore, finding $$\varphi^\ast$$, the minimizer of the KL divergence, is equivalent to finding the maximizer of $$\mathcal{L}(\varphi;x_0)$$. And finding this maximizer is something that can be done relatively easily if we choose the parametric family $$q_\varphi$$ wisely. We won't go into details here, except for the case of VAEs described below.

$$\mathcal{L{(\varphi;x_0)}}$$ is called the *evidence lower bound* (ELBO). The reason is the following: We see that $$\log p(x_0)=D_\text{KL}(q_\varphi\vert \vert p)+\mathcal{L}(\varphi;x_0)$$. The left-hand side does not depend on $$\varphi$$. In the right-hand side, the KL divergence is always nonnegative. It follows that the ELBO is a lower bound on $$\log p(x_0)$$, and the latter is sometimes called the *evidence*. We also see that if we find $$\varphi^\ast$$ such that $$D_\text{KL}(q_{\varphi^\ast}\vert \vert p)=0$$, then the approximate posterior becomes identical to the true posterior, and the ELBO becomes identical to $$\log p(x_0)$$.

(The ELBO is sometimes also known as the *free energy*, because it parallels with a similar quantity in statistical physics).

We can see that $$\mathcal{L}(\varphi;x_0)$$ can be written as the sum of $$\underset{z\sim q_\varphi}{\mathbb{E}}\log p(x_0,z)$$ and the entropy of $$z$$ under $$q_\varphi$$. We can therefore interpret the maximization of the ELBO as maximizing $$\underset{z\sim q_\varphi}{\mathbb{E}}\log p(x_0,z)$$ , with a regularization of the entropy. Without the entropy term, the optimal distribution $$q$$  would be a delta function around the MAP estimator $$\text{argmax}_z\ p(z\vert x_0)$$. The entropy term encourages the distribution to be more spread out instead of collapsing to a point.

#### Amortized Variational Inference

As presented thus far, $$q_\varphi(z)$$ is a function of $$z$$ and not of $$x$$, because the ELBO maximization process is done separately for each $$x$$, and therefore we never try to find an explicit functional dependence of the approximate posterior on $$x$$. An alternative to this is called *amortized VI*, where we try to find an explicit dependence on $$x$$. now the approximate posterior is a function  $$q_\varphi(z\vert x)$$ which takes as input both $$z$$ and $$x$$. Since the dependence on $$x$$ can be complicated, it is typically modelled using a NN. And now, instead of talking about the ELBO for a specific data point $$x_0$$, we talk about the ELBO averaged over the entire training set, and this is what we want to maximize: $$\mathcal{L}(\varphi)=\frac{1}{n}\sum\limits_{i=1}^{n}\mathcal{L}(\varphi;x_i)$$. An alternative definition is $$\mathcal{L}(\varphi)=\underset{x\sim p_\text{true}}{\mathbb{E}}\ \mathcal{L}(\varphi;x)$$   where the expectation is over the true distribution of $$x$$ (in the following, we will use the name ELBO for both $$\mathcal{L}(\varphi;x)$$ and $$\mathcal{L}(\varphi)$$). We can optimize this ELBO using SGD by sampling batches from the training set. Amortized VI can be much more efficient than the alternative, because it finds an approximate posterior that works for all $$x$$ values instead of finding a separate function for each $$x$$, but on the other hand the approximated posterior can be less accurate per $$x$$. 

#### Adding Generative Parameters

Notice that in this presentation of VI, we modeled the distribution of $$x$$ using  $$p(z)$$ and  $$p(x\vert z)$$. We did not attach any parameters $$\theta$$ to these because we assumed that we already have a tuned or true model. In practice, we will probably need to optimize the model as well. That is, we will look at a parametric family of models $$p_\theta(z), p_\theta(x\vert z)$$, and will try to find parameters  $$\theta^\ast$$ that give the largest likelihood of our data (Notice that we use $$\theta$$ to collectively refer to the parameters of both $$p_\theta(z)$$ and $$p_\theta(x\vert z)$$, even if these two families don't share any of their parameters). As usual, we will want to find parameters that maximize the likelihood of our data. For a single datapoint $$x$$, we saw that the (log) likelihood equals $$\log p_\theta(x)=D_\text{KL}(q_\varphi\vert \vert p_\theta)+\mathcal{L}(\theta,\varphi;x)$$, where the ELBO is defined as before $$\mathcal{L}(\theta, \varphi;x)=\underset{z\sim q_\varphi}{\mathbb{E}}[\log p_\theta(x\vert z)+\log p_\theta(z)-\log q_\varphi(z)]$$. From this we see that if we maximize $$\mathcal{L}(\theta, \varphi;x)$$ we get two effects: maximizing with respect to the parameters $$\varphi$$ makes the ELBO a better approximation of the log-likelihood $$\log p_\theta(x)$$, and maximizing with respect to $$\theta$$ increases the likelihood, thus making the LVM a better fitted model. We refer to $$\theta$$ as *generative parameters* and to $$\varphi$$ as *variational parameters*.

In the amortized version, we will want to maximize the following ELBO, with respect to both the generative and variational parameters:

$$\mathcal{L}(\theta, \varphi)=\underset{x\sim p_\text{true}, \ z\sim q_\varphi}{\mathbb{E}}[\log p_\theta(x\vert z)+\log p_\theta(z)-\log q_\varphi(z\vert x)]$$.

### Variational Autoencoders

#### Choice of Distribution Families for the LVM and Approximate Posterior

This ELBO $$\mathcal{L}(\theta, \varphi)$$ that we've just seen is exactly the objective function of VAEs. In the basic VAE setting, we have data $$x_1,..,x_n$$ drawn IID from the unknown true distribution $$p_\text{true}$$. The number of samples $$n$$ is typically large, and $$x_i$$ are typically high-dimensional, and/or come from a complicated distribution. Henceforth we will use the example of $$x_i$$ being images. The true distribution of images is modeled as a LVM, whose latent variable is sampled from a prior continuous distribution that belongs to a parametric family $$p_\theta(z)$$. The dimension of $$z$$ is usually much smaller than the dimension of $$x$$. In the simplest case this is taken to be a standard multivariate normal distribution $$\mathcal{N}(0,I)$$ without parameters, but more expressive versions exist: for example, in *hierarchical VAEs* $$z$$ itself is sampled in a multi-step process each of which is modeled as a NN. The probability of an image $$x$$ given the latent $$z$$ is also assumed to belong to a parametric family $$p_\theta(x\vert z)$$. This may be either a continuous or a discrete distribution, and it is implemented as a NN.  Typically, $$p_\theta(x\vert z)$$ is a multivariate normal distribution whose mean is a NN which takes $$z$$ as input, and the covariance is either taken as constant $$\sigma^2I$$ or as an additional output of the NN. This NN is called the *decoder*, for reasons that will become apparent later. In the discrete case, a popular choice is to make $$p_\theta(x\vert z)$$ a distribution over $$k$$ options, whose probabilities are the softmaxed output of a NN with input $$z$$. When $$x$$ is a sequence, $$p_\theta(x\vert z)$$ can be an autoregressive model (in these case, sometimes $$z$$ is itself modeled as a latent sequence).

For the approximate posterior $$q_\varphi(z\vert x)$$, VAEs typically use a normal distribution with an isotropic covariance whose mean and covariance are given by the output of a NN with parameters $$\varphi$$ and input $$x$$. This NN is called the *encoder*, for reasons that will become apparent later. Other, more expressive alternatives for the encoder exist, such as a using a normalizing flow NN or a hierarchical version which samples $$z$$ in a multistep process that depends on $$x$$.

To summarize, a typical case is: 
$$p_\theta(z)=\mathcal{N}(z;0,I)$$   ,
$$p_\theta(x\vert z)=\mathcal{N}(x;\mu_\theta(z),\sigma^2I)$$    ,

$$q_{\varphi}(z\vert x)=\mathcal{N}(z;\mu_\varphi(x),\sigma_\varphi^2(x)I)$$   ,
where $$\sigma$$ is a constant (hyperparameter), and the following are NNs: $$\mu_\theta(z)$$, $$\mu_\varphi(x)$$, $$\sigma_\varphi(x)$$. (The latter NN returns a scalar which equals the standard deviation along the diagonal of  the diagonal covariance matrix). (In case you were wondering, the notation $$\mathcal{N}(y;\mu,\Sigma)$$ is a shorthand for writing the PDF of a normal vector with mean vector $$\mu$$ and covariance matrix $$\Sigma$$, evaluated at point $$y$$).

#### Reparameterization Trick

The objective of VAEs is to maximize the ELBO $$\mathcal{L}(\theta, \varphi)=\underset{x\sim p_\text{true}, \ z\sim q_\varphi}{\mathbb{E}}[\log p_\theta(x\vert z)+\log p_\theta(z)-\log q_\varphi(z\vert x)]$$ .  This can be done using SGD with backpropagation (actually, gradient *ascent*, not descent). As a recap, let's review SGD:

Usually (not in the VAE context), we have a loss function $$L(\theta)=\underset{x\sim p}{\mathbb{E}}l(x;\theta)$$   which we want to minimize. With gradient descent, this means that we start with some random initialization of $$\theta$$, and then we perform steps of updating $$\theta$$ in the direction opposite to the gradient $$\nabla_\theta L(\theta)$$ . This gradient cannot be computed analytically because we don't have $$p$$, but we can form an estimation of this gradient. To do this, we notice that we can move the gradient to within the expectation operator: $$\nabla_\theta L(\theta)=\nabla_\theta\underset{x\sim p}{\mathbb{E}}l(x;\theta)=\underset{x\sim p}{\mathbb{E}}\nabla_\theta l(x;\theta)$$. This means that if we take a batch of $$x$$ values sampled from $$p$$, we can calculate the gradient of $$l$$ for each of them, take the average, and this will be an unbiassed estimate of $$\nabla_\theta L(\theta)$$. This estimation might still have an error (its variance), but due to the unbiasedness, over many SGD iterations the overall direction of convergence will be good. Therefore, in SGD we take, on each iteration, a new small batch of training points (small for computational efficiency), calculate the average of their gradient (using backpropagation), and update the parameters $$\theta$$ accordingly. 

Now let's look at VAEs again. How can we optimize the ELBO using SGD? Similarly to what was just explained, we will need to use batches of training data $$x$$, and for each $$x$$ to sample $$z$$ from $$q_\varphi(z\vert x)$$. But there is a problem. It is no longer the case that we can move the gradient to within the expectation operator:  $$\nabla_\varphi\mathcal{L}(\theta, \varphi)=\nabla_\varphi\underset{x\sim p_\text{true}, \ z\sim q_\varphi}{\mathbb{E}}[\cdot\cdot\cdot]\neq{\underset{x\sim p_\text{true}, \ z\sim q_\varphi}{\mathbb{E}}}\nabla_\varphi[\cdot\cdot\cdot]$$  . The reason is that $$\varphi$$ appears also in the distribution $$q_\varphi$$ over which the expectation is done. This means that if we tried to naively do SGD+backprop, we would get a biased gradient estimator which might converge to a bad point, or not converge at all.

The simple solution to this problem is to use the *reparameterization trick*. As we mentioned, in the typical case we take $$q_{\varphi}(z\vert x)=\mathcal{N}(z;\mu_\varphi(x),\sigma_\varphi^2(x)I)$$. In this case, the random vector $$z$$ can be sampled from the approximate posterior by first sampling $$\epsilon\sim\mathcal{N}(0,I)$$ with the same dimension as $$z$$, and then calculating $$z=\sigma_\varphi(x)\epsilon+\mu_\varphi(x)$$. By doing this, we can rewrite the ELBO as: 

$$\mathcal{L}(\theta, \varphi;x)=\underset{\epsilon\sim \mathcal{N}(0,I)}{\mathbb{E}}[\log p_\theta(x\vert z(\epsilon))+\log p_\theta(z(\epsilon))-\log q_\varphi(z(\epsilon)\vert x)]$$.  Here, we use the notation $$z(\epsilon)=\sigma_\varphi(x)\epsilon+\mu_\varphi(x)$$. And now, $$\varphi$$ does not appear anymore in the distribution under the $$\mathbb{E}$$, which means that we can proceed with SGD+backprop as usual.

#### Training the VAE Using SGD

Putting this all together, here is the training process of VAE (in the typical case presented here):

1. Choose a size (=number of dimensions) for the latent vector $$z$$.
2. Choose a NN architecture for the decoder $$\mu_\theta(\cdot)$$, whose input will be of the same size as the latent vector $$z$$ and output as the same size of the data samples $$x$$.
3. Choose a NN architecture for the encoder $$\mu_\varphi(\cdot)$$, $$\sigma_\varphi(\cdot)$$ (this is either two separate NNs or one NN with two types of outputs), whose input will be of the same size as the data samples $$x$$ and whose two outputs are each the same size of the latent vector $$z$$.
4. Initialize random values for $$\theta$$ and $$\varphi$$.
2. Repeat many times (the calculations should be done in an autograd framework, such as Tensorflow or Pytorch, that performs backpropagation through the entire computational graph):
   1. Take a random batch of training data $$x_1,x_2,...,x_m$$
   2. For each $$i=1,...,m$$ :
      1. Apply the encoder NN on $$x_i$$ to get $$\mu_\varphi(x_i),\sigma_\varphi(x_i)$$. 
      2. Using the normal distribution PDF formula, calculate $$q_\varphi(z_i\vert x_i)=\mathcal{N}(z_i;\mu_\varphi(x_i),\sigma_\varphi^2(x_i)I)$$
      3. Sample a value $$\epsilon_i$$ from $$\mathcal{N}(0,I)$$
      4. Calculate: $$z_i=\sigma_\varphi(x_i)\epsilon_i+\mu_\varphi(x_i)$$
      5. Using the normal distribution PDF formula, calculate $$p_\theta(z_i)=\mathcal{N}(z_i;0,I)$$.
      6. Apply the decoder NN on $$z_i$$ to get $$\mu_\theta(z_i)$$. 
      7. Using the normal distribution PDF formula, calculate $$p_\theta(x_i\vert z_i)=\mathcal{N}(x_i;\mu_\theta(z_i),\sigma^2I)$$
      8. Calculate the sum of logs of the elements calculated above: $$\log p_\theta(x_i\vert z_i)+\log p_\theta(z_i)-\log q_\varphi(z_i\vert x_i)$$
      9. Using backpropagation, calculate the gradient $$G_i$$ of the expression from the previous step with respect to $$\theta$$ and $$\varphi$$ (in particular, make sure to backpropagate through all of the above steps).
   3. Calculate the average of all gradients: $$G=\frac{1}{m}\sum\limits_{i=1}^{m}G_i$$ which serves as an approximation to the gradient of the ELBO, and use $$G$$ to update the parameters $$\theta$$ and $$\varphi$$ so as to maximize the ELBO ($$G$$ is the input to the optimizer, be it SGD or Adam or anything else).

#### Applications of VAE

A trained VAE, has at least two nice applications. First, we get an approximate posterior $$q_\varphi(z\vert x)$$ which should be a good approximation to $$p_\theta(z\vert x)$$. This allows us to take an image $$x$$ and see the values of $$z$$ that may have created this $$x$$. These values of $$z$$ can serve as a low dimensional representation of $$x$$, which can be useful for upstream tasks (such as clustering or classification). 

Another nice application is that we can easily generate new samples $$x$$ that do not exist in our dataset. To do this, we simply sample $$z$$ from the prior $$p_\theta(z)$$ and then sample $$x$$ from $$p_\theta(x\vert z)$$. These two sampling steps are easy to do because these are normal distributions. For example, this allows generating realistic looking images. Alternatively, given an image $$x$$, we can sample $$z$$ from $$q_\varphi(z\vert x)$$, and then sample another image $$x'$$ from $$p_\theta(x'\vert z)$$. This can give us a new image which is similar to the original image. 

#### Some Additional Information

##### Relation to Autoencoder

The terms "encoder" and "decoder" used above come from the view that the approximate posterior takes an image $$x$$ an returns a (distribution whose mean is a) low dimensional vector $$z$$, which can be seen as an encoded version of $$x$$. And, given $$z$$ , we can "decode" it back to $$x$$ by calculating (the mean of) $$p_\theta(x\vert z)$$. There is a similarity here to autoencoders, even though the formalism is completely different. The ELBO term $$\underset{z\sim q_{\varphi}(z\vert x)}{\mathbb{E}}\log p_\theta(x\vert z)$$ is often referred to as the *reconstruction term*, and it plays a roll similar to reconstruction in autoencoders: maximizing it with respect to $$\theta$$ means that the decoder is likely to reconstruct $$x$$ after obtaining a sample $$z$$ drawn from $$q_\varphi(z\vert x)$$.

##### Reconstruction and Rate terms

The ELBO can be written as the difference between the reconstruction term and a KL divergence: $$\mathcal{L}(\theta, \varphi;x)=\underset{z\sim q_\varphi}{\mathbb{E}}[\log p_\theta(x\vert z)]-D_{\text{KL}}(q_\varphi(z\vert x)\ \vert \vert \ p_\theta(z))$$.  The KL divergence term is often referred to as the *rate*,  because it has an information theoretic interpretation of the average number of additional bits required for encoding $$z$$ drawn from the approximate posterior, with a code optimized for samples from the prior. Maximizing the ELBO balances between maximizing reconstruction whilst minimizing the rate. Without the rate, the optimal $$q_\varphi$$ would put all of its probability mass on the single value of $$z$$ which gives highest likelihood to $$x$$, ignoring the prior entirely. The rate term is there to "remind" $$q_\varphi$$ that it should also be compatible with the prior. 

##### Estimating the Marginal Likelihood

The quantity $$p_\theta(x)$$ is often called the *marginal likelihood* of $$x$$. Notice that the VAE formalism does not give us a way to calculate this value for a given $$x$$. We can try to estimate this value using Monte Carlo: Since $$p_\theta(x)=\underset{z\sim p_\theta(z)}{\mathbb{E}}p_\theta(x\vert z)$$ , we can sample a large batch of $$z$$ values from the prior, calculate $$p_\theta(x\vert z)$$  for each, and average the result. This will give an unbiassed estimator of the marginal likelihood. However, this estimator will typically have a very large variance, and therefore a large error. Instead, we can use importance sampling, where we change the sampling distribution. A good choice for the sampling distribution is in fact $$q_\varphi(z\vert x)$$. It can be seen that $$p_\theta(x)=\underset{z\sim q_\varphi(z\vert x)}{\mathbb{E}}p_\theta(x\vert z)\frac{p_\theta(z)}{q_\varphi(z\vert x)}$$ . So, if we sample $$z$$ from the approximate posterior, the value of $$p_\theta(x\vert z)\frac{p_\theta(z)}{q_\varphi(z\vert x)}$$ is an unbiassed estimator of the marginal likelihood. To see that this estimator has a smaller variance, notice that in the case where the approximation $$q_\varphi(z\vert x)\approx p_\theta(z\vert x)$$  is very good, then $$p_\theta(x\vert z)\frac{p_\theta(z)}{q_\varphi(z\vert x)}\approx p_\theta(x\vert z)\frac{p_\theta(z)}{p_\theta(z\vert x)}=p_\theta(x)$$ by using Bayes rule. The latter does not depend on $$z$$ and therefore has zero variance.

In practice, estimating the marginal likelihood of $$x$$ can only be done accurately when the dimension of $$z$$ is very small.

##### Another Derivation of the ELBO

In the previous section we mentioned that $$p_\theta(x)=\underset{z\sim q_\varphi(z\vert x)}{\mathbb{E}}p_\theta(x\vert z)\frac{p_\theta(z)}{q_\varphi(z\vert x)}$$. Applying $$\log$$ to both sides, we get: $$\log p_\theta(x)=\log \underset{z\sim q_\varphi(z\vert x)}{\mathbb{E}}p_\theta(x\vert z)\frac{p_\theta(z)}{q_\varphi(z\vert x)}$$. Noticing that $$\log$$ is a convex function, we can use Jensen's inequality and move the $$\log$$ into the expectation:  $$\log p_\theta(x)\geq \underset{z\sim q_\varphi(z\vert x)}{\mathbb{E}}\log \left[p_\theta(x\vert z)\frac{p_\theta(z)}{q_\varphi(z\vert x)}\right]$$. Notice that the right hand side is exactly the ELBO. This shows that indeed the ELBO is a lower bound on the evidence. Also, we remind you that Jensen's inequality becomes an equality only when the random variable inside the expectation is deterministic. Here, this will happen only when the approximate posterior exactly equals the posterior (This can be shown using Bayes). So, we can see that maximizing the ELBO with respect to $$\varphi$$ gives a better lower bound on the evidence, and maximizing with respect to $$\theta$$ gives a higher likelihood to the data.    

#####  Hierarchical VAE

We presented the VAE as consisting of a two step process: first a latent variable is sampled, and then the image is sampled. Sometimes, more steps are required for modeling complicated distributions. This is what hierarchical VAE does. The latent vector $$z$$ is broken into different hierarchical groups $$z^1,z^2,...,z^k$$ , and the prior is written as $$p_\theta(z)=p_\theta(z^1)p(z^2\vert z^1)...p(z^k\vert z^{<k})$$ . This describes a process where first hierarchy $$z^1$$ is sampled, then $$z^2$$, and so on. The distribution $$p_\theta(x\vert z)$$ is taken (in the image case) such that $$z^1$$ controls lowest resolution features of the image, and $$z^k$$ controls highest resolution features, and in general the resolution becomes higher for the higher hierarchies. The approximate posterior takes the form $$q_\varphi (z\vert x)=q_\varphi (z^1\vert x)q_\varphi (z^2\vert z^1,x)...q_\varphi (z^k\vert z^{<k},x)$$. All of the components of the LVM and approximate posterior are typically taken as diagonal normal distributions, with mean and standard deviation that are the output of a NN. The ELBO  in this case takes the form: 
$$\mathcal{L}(\theta, \varphi; x)=\underset{z\sim q_\varphi}{\mathbb{E}}[\log p_\theta(x\vert z)]-D_{\text{KL}}(q_\varphi(z^1\vert x)\ \vert \vert \ p_\theta(z^1))+\\-\sum\limits_{i=2}^k \underset{z^{<i}\sim q_\varphi(z^{<i}\vert x)}{\mathbb{E}}D_{\text{KL}}(q_\varphi(z^i\vert z^{<i},x)\ \vert \vert \ p_\theta(z^i\vert z^{<i}))$$

This ELBO too can be optimized using SGD+backprop+reparameterization trick. 

##### Posterior Collapse

In some cases, the decoder is so powerful (=flexible=expressive), that the VAE optimization runs into a problem: The decoder learns to ignore $$z$$, and the encoder learns to ignore $$x$$. Looking at the ELBO, $$\mathcal{L}(\theta, \varphi;x)=\underset{z\sim q_\varphi}{\mathbb{E}}[\log p_\theta(x\vert z)]-D_{\text{KL}}(q_\varphi(z\vert x)\ \vert \vert \ p_\theta(z))$$ , we see that this means that the approximate posterior "collapses" to the prior, and $$p_\theta(x\vert z)$$ becomes effectively a marginal likelihood $$p_\theta(x)$$. This posterior collapse is characteristic in cases where $$x$$ is a sequence and the decoder is autoregressive. In an autoregressive model, the next element of $$x$$ is conditioned on all previous elements as well as on $$z$$. The conditioning on the previous elements might make the conditioning on $$z$$ negligible. 

The consequence of posterior collapse is that the LVM modeling is destroyed - the latent variable $$z$$ carries no information about $$x$$. There are various methods to mitigate posterior collapse, we won't go into details here.
