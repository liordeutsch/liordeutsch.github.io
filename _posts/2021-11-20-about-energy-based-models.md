---
layout: post
comments: true
title:  "About Energy Based Models"
excerpt: "I present the formulation of deep energy based models."
date:   2021-11-20 08:00:00
mathjax: true
---

This is a summary of the formulation of deep energy based models (EBMs). The emphasis is on the formulation, and not on specific examples of implementations/applications. This is not an exhaustive overview of all types of EBMs. 

This review assumes basic knowledge of probability theory, as well as basic knowledge of the following concepts: Maximum likelihood, neural networks (NNs), stochastic gradient descent (SGD), bias of an estimator. The review mentions the following terms as well, but prior knowledge of them is not essential:  Langevin MCMC. 

---

We have data which was sampled from some unknown complicated PDF $$p(x)$$  (for example, images). 

We want to have a NN that allows us to approximate $$p(x)$$ for any $$x$$.  The NN defines a (flexible) parametric family of distributions $$q(x;\theta)$$, and our goal is to find parameters $$\theta^\ast$$ such that $$q(x;\theta^\ast)\approx p(x)$$.

It is hard to design a NN $$q(x;\theta)$$ with the desired properties of being both a legal probability density distribution (i.e nonnegative and integrates to $$1$$), and a flexible function approximator (that allows approximating complicated realistic high dimensional distributions).

As it turns out, it is sometimes easier to have a model that is trained to return a value approximately *proportional* to $$p(x)$$ (where the constant of proportion does not depend on $$x$$). 

More precisely, we can take a NN $$E(x;\theta)$$ whose scalar output does not have any restriction (it can take any sign; it does not need to integrate to $$1$$). And we can view $$e^{-E(x;\theta)}$$ as being proportional to a PDF $$q(x;\theta)$$, with proportionality factor $$Z_\theta$$:    $$q(x;\theta)= \frac{e^{-E(x;\theta)}}{Z_\theta}$$

We never model or calculate $$Z_\theta$$ directly; it depends on $$\theta$$ only in the sense that the numerator depends on $$\theta$$  and therefore the normalizing constant in the denominator also has a dependency on $$\theta$$ so that the integral of the PDF gives $$1$$ as required. To obtain the value of $$Z_\theta$$, we would need to integrate over all $$x$$ values: $$Z_\theta=\int\limits_x e^{-E(x;\theta)}\text{d}x$$. 

We wish to find $$\theta^\ast$$ such that $$q(x;\theta^\ast)\approx p(x)$$.  We can do this with doing maximum likelihood on $$q(x;\theta)$$  (of course, if $$q(x;\theta)$$ is *too* flexible, then it might just memorize the training set which would yield a very high likelihood but a bad approximation of $$p(x)$$; in practice, this is usually not a problem for very large datasets trained with SGD...).

The function $$E(x;\theta)$$ is called the *energy function*. (This terms comes from an analogy to statistical mechanics, where (roughly) the probability density of a physical state $$x$$ of a system at equilibrium with a given temperature $$T$$ is proportional to $$e^{-E(x)/kT}$$, where $$E(x)$$ is the energy of the state, and $$k$$ is a known constant; this is also known as the *Gibbs distribution* of the energy function $$E(x)$$).   The lower the energy of , $$x$$ the more probable it is under $$q(x;\theta)$$.  (in some texts, an opposite convention is used, where high energy corresponds to higher probability density, and then we omit the minus sign in the exponent).

Given $$E(x;\theta^\ast)$$ for a given $$x$$, we cannot calculate/approximate $$p(x)$$ easily, we can only approximate a value that is proportional to $$p(x)$$:    $$e^{-E(x,\theta^\ast)}$$.   For many applications, this is good enough. For example, it allows calculating the ratio of probability densities of $$x_1$$ and $$x_2$$;  it allows sampling from the PDF using MCMC; it allows applying thresholds on $$E(x;\theta)$$  for decision making (e.g., accept/reject $$x$$ only if it has enough energy).

We said that we want to do maximum likelihood. But how is it done? Here:

Define $$L(\theta)=\underset{x\sim p}{\mathbb{E}}\log q(x;\theta)$$ . This is the function we would like to maximize. Then:

$$L(\theta)=\underset{x\sim p}{\mathbb{E}}[-E(x;\theta)-\log Z_\theta]=-\underset{x\sim p}{\mathbb{E}}[E(x;\theta)]-\log Z_\theta$$

Therefore:

$$(\star)$$    $$\nabla_\theta L(\theta)=-\underset{x\sim p}{\mathbb{E}}[\nabla_\theta E(x;\theta)]-\frac{\nabla_\theta Z_\theta}{Z_\theta}$$

We saw that $$Z_\theta=\int\limits_x e^{-E(x;\theta)}\text{d}x$$ , therefore $$\frac{\nabla_\theta Z_\theta}{Z_\theta}=-\frac{\int\limits_x e^{-E(x;\theta)}\ \nabla_\theta E(x;\theta)\text{d}x}{Z_\theta}=-\int\limits_x \frac{e^{-E(x;\theta)}}{Z_\theta}\ \nabla_\theta E(x;\theta)\text{d}x=-\int\limits_x q(x;\theta)\nabla_\theta E(x;\theta)$$ 
$$=-\underset{x\sim q}{\mathbb{E}}[\nabla_\theta E(x;\theta)]$$

Putting this back in $$(\star)$$ we get:

$$\nabla_\theta L(\theta)=\underset{x\sim q}{\mathbb{E}}[\nabla_\theta E(x;\theta)]-\underset{x\sim p}{\mathbb{E}}[\nabla_\theta E(x;\theta)]$$

It follows that, the one-sample unbiased gradient estimator is: $$\nabla_\theta E(x^-;\theta)-\nabla_\theta  E(x^+;\theta)$$ , where $$x^-$$ is a sample from $$q(x;\theta)$$ and  $$x^+$$ is a sample from $$p(x)$$. Thus, to do SGD, we can sample a batch from $$q(x;\theta)$$ and a batch from $$p(x)$$ and apply backpropagation to update the parameters $$\theta$$. 

Sampling a batch from $$p(x)$$ is easy -  we simply sample from our training data. How do we sample from $$q(x;\theta)$$? (Notice that this distribution changes after each update step of SGD). The answer is MCMC. There are multiple variants of MCMC, but the one most suitable for here is Langevin dynamics MCMC. This particular variant is very suitable for the case where we have an energy function which is differentiable (with respect to $$x$$). This is exactly our situation. According to Langevin MCMC, to sample from $$q(x;\theta)$$, we do the following process: 

1. Initialize a random value $$x$$   (for example, from a standard normal distribution).
2. Repeat many times:     update $$x$$ using gradient descent, and add noise:   $$x\leftarrow x-\nabla_x E(x;\theta)+z$$ ,  where $$z\sim\mathcal{N}(0,\sigma^2I)$$  (the noise std $$\sigma$$ is a hyperparameter...)

If step 2 is repeated enough times, it can be shown that the resulting $$x$$ is a fair sample from $$q(x;\theta)$$. 

So, sampling a batch from $$q(x;\theta)$$ requires doing many MCMC steps for each batch element. In cases where this takes too much time, a few techniques were invented: 

1. *Contrastive divergence* - where instead of initializing $$x$$ from a normal distribution, it is initialized from a training point, and this should reduce the mixing time of the MCMC process (i.e the number of iterations required).
2. *Persistent chains* - where $$x$$ is not reinitialized between consecutive SGD steps, which again might mean that less steps are required.

To summarize, to train an EBM:

1. Define a NN $$E(x;\theta)$$ whose output is a scalar. 
2. Perform SGD to do maximum likelihood by repeating the following steps:
   1. Sample a batch of points $$x_1^+,...,x_n^+$$ from the training data $$p(x)$$ , and calculate their average energy gradient:  $$G_+=\frac{1}{n}\sum\limits_{i=1}^n\nabla_\theta  E(x_i^+;\theta)$$
   2. Use Langevin MCMC (maybe with contrastive divergence or persistent chains) to sample a batch of points  $$x_1^-,...,x_m^-$$ from $$q(x;\theta)= \frac{e^{-E(x;\theta)}}{Z_\theta}$$, and calculate their average energy gradient:  $$G_-=\frac{1}{m}\sum\limits_{i=1}^m\nabla_\theta  E(x_i^-;\theta)$$
   3. Update the parameters of $$E(x;\theta)$$ using the following gradient approximation:  $$G_--G_+$$  .
   

(the same Langevin dynamics described above can be used to generate samples from the trained model).

(there are other formulations for EBMs as well. For example, restricted Boltzmann machines, Markov random fields....)
