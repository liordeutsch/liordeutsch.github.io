---
layout: post
comments: true
title:  "A New Generative Model?"
excerpt: "Initial thoughts about a new formalism for generative models."
date:   2019-10-25 11:00:00
mathjax: true
---

This post will describe initial thoughts about a new formalism for generative models. Hopefully, I will be able to further develop these ideas in future posts. I'd be happy to hear the readers' critical thoughts about this.

---

### The goal

We are given samples $$x_1,x_2,...,x_n \in \mathbb{R}^d$$ from a random vector $$X$$ whose distribution $$p_X$$ is unknown.

We sample from a latent random vector $$Z$$ over a simple known distribution $$p_Z$$ (e.g. an isotropic Gaussian) and feed this into a generator neural network $$G_{\theta}(z) \in \mathbb{R}^d$$. The random vector $$G_{\theta}(Z)$$ has distribution $$p_{\theta}$$.

We would like to optimize $$\theta$$ so that $$p_{\theta}$$ is similar to $$p_{X}$$.



### Formalizing the goal as equalities of the probabilities of all subsets

Ideally, we would want that $$P(X \in \omega) = P(G_{\theta}(Z) \in \omega)$$ for all subsets $$\omega\subseteq \mathbb{R}^d$$  (more precisely, $$\omega$$ should be an event in the $$\sigma$$-algebra of the probability space).

The relation $$\forall \omega, P(X \in \omega) = P(G_{\theta}(Z) \in \omega)$$ imposes infinitely many constraint equations - one for each $$\omega$$. We can relax it, though, into a single equation: Instead of $$\forall \omega$$, let's take some probability distribution $$p_{\Omega}(\omega)$$ over subsets.  $$\Omega$$ is a "random  variable" whose values are subsets of $$\mathbb{R}^d$$, and $$p_{\Omega}(\omega)$$  is a function $$2^{\mathbb{R}^d} \rightarrow \mathbb{R}$$ that associates a probabilit density for each subset. We now demand that the equality holds under expectation:

$$\underset{\omega \sim p_{\Omega}}{\mathbb{E}} \ P(X \in \omega) = \underset{\omega \sim p_{\Omega}}{\mathbb{E}} \  P(G_{\theta}(Z) \in \omega)$$

Notice that we can also write this as:

$$\underset{\omega \sim p_{\Omega}}{\mathbb{E}} \ \underset{x \sim p_X}{\mathbb{E}} \ \mathbf{1}_{x \in \omega} = \underset{\omega \sim p_{\Omega}}{\mathbb{E}} \ \underset{z \sim p_Z}{\mathbb{E}} \  \mathbf{1}_{G_{\theta}(z) \in \omega}$$

where $$\mathbf{1}_{c}$$ is the indicator function, which equals $$1$$ if condition $$c$$ holds and $$0$$ otherwise.

Notice that this is not a a loss function. Rather, it is a equation/condition that we would want to hold after optimizing the loss function. And we would like this to hold for some smart choice of $$p_{\Omega}(\omega)$$.

(To show that we need to choose a *smart* $$p_{\Omega}(\omega)$$, here is an example of a choice for $$p_{\Omega}(\omega)$$  which is *not* smart : make it a delta function at $$\mathbb{R}^d$$, that is, make $$\Omega$$ always equal to the whole sample space, with probability $$1$$. And then the equation above would hold trivially for all $$\theta$$.)



### Representing subsets of $$\mathbb{R}^d$$ and a distribution over them using neural networks

We want to be able to practically represent a subset $$\omega$$ and a "random variable" $$\Omega$$ whose values are subsets. 

We can represent $$\omega$$ using a neural network. Take a neural network $$g_{\varphi}(x)$$ with weights $$\varphi$$ that takes as input a vector $$x \in \mathbb{R}^d$$ and returns a scalar number. We can say that $$g_{\varphi}(x)$$ is a representation of the subset $$\omega_{\varphi}=\{ x : g_{\varphi}(x) > 0 \}$$. This shows that a neural network represents a subset of $$\mathbb{R}^d$$. We can also claim that the correspondence works the other way: every subset can be represented by some neural network $$g_{\varphi}(x)$$ (for this claim, we need to assume that we are dealing with large capacity neural networks $$g_{\varphi}$$, and that we are interested in regular enough subsets $$\omega$$, and we need to use the universal approximator theorem).

To represent $$\Omega$$, we can use [hypernetworks](https://arxiv.org/abs/1801.01952). A hypernetwork is a neural network which takes as input some random vector, and outputs weights for another neural network (with a known and fixed architecture). We denote the hypernetwork with $$H_{\psi}(u)$$, where $$\psi$$ are its weights, and $$u$$ is its input. We feed $$H_\psi$$ with a random vector $$U$$ drawn from a simple distribution, and obtain $$\Phi=H_\psi(U)$$. The hypernetwork induces some distribution over weight vectors. In our case, the random vector $$\Phi$$ can be seen as a representation of $$\Omega$$. We can sample from $$\Phi$$ to obtain a weight vector $$\varphi$$,  plug it into $$g_\varphi (x)$$ to get a representation of a subset $$\omega_{\varphi}$$. 



### The objective equation

In the equation above, we can replace $$\mathbf{1}_{x \in \omega}$$ with $$\mathbf{1}_{g_\varphi (x) > 0}$$. To make things nicer, we can add a sigmoid to the output of $$g_{\varphi} $$ ,  and we can use soft (fuzzy) association and replace $$\mathbf{1}_{g_\varphi (x) > 0}$$ simply with $$g_{\varphi}(x) $$.

We also need to replace $$\omega \sim p_{\Omega}$$ with $$\varphi \sim p_{\Phi}$$, where $$p_{\Phi} (\varphi)$$ is a distribution over weights $$\varphi$$, induced by the hypernetwork.

With these replacements, the equation above turns into:

$$\underset{\varphi \sim p_{\Phi}}{\mathbb{E}} \ \underset{x \sim p_X}{\mathbb{E}} \ g_{\varphi}(x) = \underset{\varphi \sim p_{\Phi}}{\mathbb{E}} \ \underset{z \sim p_Z}{\mathbb{E}} \  g_{\varphi}(G_{\theta}(z))$$

Reorganizing it becomes:

$$\underset{\varphi \sim p_{\Phi}}{\mathbb{E}} \left[ \ \underset{x \sim p_X}{\mathbb{E}}  \ g_{\varphi}(x)-\underset{z \sim p_Z}{\mathbb{E}}g_{\varphi}(G_{\theta}(z)) \right]=0$$

The hypernetwork $$H_\psi$$ does not explicitly appear in this equation, so I add it in:
 
$$\underset{u \sim p_{U}}{\mathbb{E}} \left[ \ \underset{x \sim p_X}{\mathbb{E}}  \ g_{H_{\psi}(u)}(x)-\underset{z \sim p_Z}{\mathbb{E}}g_{H_{\psi}(u)}(G_{\theta}(z)) \right]=0$$

I refer to either of the last two equations as the **objective equation**. (I was careful not to call them an **objective function**, since this is not a function we want to minimize, but rather an equation that we want to be satisfied).



### What's next?

I'm not sure yet how to proceed from here. Here are some notes:

* In the objective equation, $$\theta$$ is what we ultimately want to optimize, but it should be accompanied by a concurrent optimization of $$\psi$$, such that $$\psi$$ yields a "good" distribution over subsets, in some sense. I think that such a distribution would generate subsets that can discriminate between $$p_{\theta}$$ and $$p_{X}$$.
  Solving the objective equation would probably be a [GAN](http://papers.nips.cc/paper/5423-generative-adversarial-nets)-like process: you perform a step to update $$\theta$$, followed by a step to update $$\psi$$, and repeat. The $$\psi$$ update step needs to make it so that the subsets generated by $$H_\psi$$ can optimally help us reach the generative model's training goal.

* We need to find a way to turn the objective equation into a loss function. I'm not sure how to do so, but notice that if we simply take the left hand side as a loss $$L(\theta)$$, we get the [Wasserstein GAN](https://arxiv.org/abs/1701.07875) loss for an appropriate choice of $$p_{\Phi}$$.
* Another idea is that we don't use a loss function, but instead try to directly solve the equation using something like Newton-Raphson method. However, this doesn't apply here since the number of equations here (which is one) is different than the number of parameters (the dimension of $$\theta$$). This is a symptom of a larger problem: Since we reduced the number of equations to being only 1, then in general the manifold of the solutions to the equation is very large, and not all of these solutions are necessarily good solutions in terms of our final goal. I hope that we can find a way to choose  $$p_{\Phi}$$ which ensures that there is only a single, good solution (or that all solutions are good). 
* Here are some open issues:
  * How to further develop this formulation?
  * Can this formulation offer anything essentially different than the GAN formulation?
  * This formulation seems to suggest that instead of using a single discriminator (as in GAN), we can use a distribution or ensemble of discriminators. How do we choose this distribution? What is the optimality criterion that it should hold? Will the distribution simply turn out to be a delta function, which means that only one discriminator is used?
  * Which loss functions does this formulation incorporate ? Can it include Wasserstein? Jensen-Shannon? and what distribution $$p_{\Phi}$$ should we choose in each case?

