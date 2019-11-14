---
layout: post
comments: true
title:  "Learning Quantum Dynamics with Feynman's Path Integrals"
excerpt: "Through the lens of a deep learning practitioner, the elegant path integral formulation of quantum mechanics looks like a an expectation over trajectories of a reinforcement learning agent."
date:   2019-11-08 04:00:00
mathjax: true
---

In this post, I will share some thought about applying ideas from quantum mechanics to machine learning.

---

## Background on the path integral formulation of quantum mechanics

One of the most remarkable and beautiful aspects of the mathematics of quantum mechanics is the formulation due to Richard Feynman and Paul Dirac, called the *[path intergral formulation](https://en.wikipedia.org/wiki/Path_integral_formulation)* (PIF). In its simplest form, the PIF is a formula that allows us to calculate (in principle and loosely speaking) the probability that a particle at position $$x_1$$ will move to position $$x_2$$ within a time span of $$T$$. Here it is:

$$P(x_1 \underset{T}\longrightarrow x_2)=\vert A \vert ^2$$ ,

where $$A$$ is a complex number, called the *amplitude*:

$$A=\displaystyle{\int\limits_{x(0)=x_1}^{x(T)=x_2} \mathcal{D}x\ \exp\left( i \displaystyle{\int\limits_{0}^{T}dt\ L(x(t),\dot{x}(t)) } \right) }$$

This formula might seem intimidating at first look, but it is not actually very complicated. Let me explain what it says:

First, note that the amplitude $$A$$ is a complex number (only due to the imaginary $$i$$ appearing in the $$\exp$$), but the probability $$P(x_1 \underset{T}\longrightarrow x_2)$$  is a real number because we take the squared modulus $$\vert A \vert^2$$. The outer $$\mathcal{D}x$$ integral in $$A$$ is the *path integral*. This might be an unfamiliar object  - it is an integral over all trajectories $$x(t)$$ such that $$x(0)=x_1$$ and $$x(T)=x_2$$. (I'll elaborate more on what this means in a moment). The outer integra*nd* is simply the exponent of another integral, which is the familiar and simple integral of the kind we met in our first calculus course. The inner integrand is some real function $$L$$ of two variables, which is called the *Lagrangian*. The Lagrangian is evaluated at the location $$x(t)$$ and the velocity $$\dot{x}(t)$$  (the dot represents the derivative with respect to time).

To make the formula for the amplitude $$A$$ clearer, I will write it in pseudo-Python-code, approximating the integrals as sums:

```python
def get_amplitude(L,x_1,x_2,T):
    """
    L:the Lagrangian. x_1, x_2:the initial and final positions. T:the time.
    """
    A = 0 # the outer integral will be accumulated here
    dt = 0.0001 # time resolution for inner integral
    for x in [x1,x2,x3,...,xn]: #x1,x2,... are functions (=trajectories) that satisfy x(0)=x_1, x(T)=x_2  (don't confuse between x_1 and x1...)
        s = 0 # the inner integral will be accumulated here
        for t in arange(0,T,dt): # arange as in Numpy. we run over a grid of time steps
            velocity = (x(t+dt)-x(t))/dt # the velocity of the trajectory x at time t
            s += i*L(x(t),velocity)*dt # inner integrand, L is the Lagrangian, i is the imaginary unit
        A += exp(s)*Dx # outer integrand. Dx is some factor that represents the "spacing" between the functions x1,x2,...
        return A
```

In other words, just like the regular integration $$\int dt$$ is approximated as a sum, where the sum is over a discrete set of numbers $$t$$,  the path integral $$\int\mathcal{D}x$$ can be approximated as a sum, where the sum is over a discrete set of functions $$x(t)$$.

The Lagrangian function $$L$$ is where all the specific details of the physical system are encoded. Two different systems will have different Lagrangian functions. For example, an electron moving in an electric field will have a Lagrangian different than an electron moving in a magnetic field. However, no matter what the system is, you can plug in the Lagrangian function into the PIF formula for the amplitude $$A$$ to obtain the probability $$P(x_1 \underset{T}\longrightarrow x_2)$$.

Before we continue, you might want to pause for a moment to ponder about how strange the path integral is. It asserts that to calculate a seemingly simple thing, such as the probability of a particle moving between two corners of my bedroom, we need to take into account all possible trajectories between these two locations, including "crazy" trajectories such as a trajectory where the particle moves in a spiral from Earth to Pluto back and forth five times faster than the speed of light. And the contribution of the outer integrand is never zero (since it is a complex number of the form $$e^{is}$$, for real $$s$$, which means that its modulus is always $$1$$).

## Two ideas for applications of the PIF with machine learning

### Idea #1: using the PIF to learn quantum dynamics

Suppose that we have a dataset of the form $$\{(x_1^1,x_2^1,T^1),(x_1^2,x_2^2,T^2),...\}$$ where each tuple consists of measurements of the initial and final locations of a particle, and the time it took it to travel. We don't have the entire trajectory, only initial and final locations. And suppose that we would like to learn the physical law governing the particle. In other words, we would like to learn to Lagrangian function $$L(x,v)$$. We can choose $$L_{\theta}(x,v)$$ to be a neural network with trainable parameters $$\theta$$. The objective function is to make the probabilities of the examples in the training dataset high (The probabilities as defined above in $$P(x_1 \underset{T}\longrightarrow x_2)$$ ). To perform the optimization, we approximate the path integral $$\int\mathcal{D}x$$ with a sum over trajectories.  That is, we sample random trajectories $$x$$ such that $$x(0)=x_1$$ and $$x(T)=x_2$$ (Monte Carlo). We also approximate the integral $$\int dt$$ as a sum (this is a 1-dimensional integral, so we can either sum over a fixed grid or do Monte Carlo again).

#### How to sample trajectories?

An important question here is - which distribution do we sample the trajectories from? I don't have a good answer to this yet, although we can take inspiration from what physicist do, which is the [stationary phase approximation](https://en.wikipedia.org/wiki/Stationary_phase_approximation). The idea is that the most important contributions to the path integral come from trajectories $$x$$ for which the inner integral is approximately constant (as a function of $$x$$).  The reason (roughly speaking) is that the exponent is of the form $$e^{is}$$, which is an oscillatory function ($$e^{is}=\cos s+i\sin s$$) and therefore contributions to the path integral $$\int \mathcal{Dx}$$ from different $$s$$ are summed up to zero on average. The exception to this is when there are many occurrences of the same $$s$$, and then the sum is coherently summed to a non-zero number.

#### Why not learn a model for $$P$$ instead of $$L$$?

Are there any advantages to learning a model for the Lagrangian $$L(x,v)$$ instead of directly learning a model for the probabilities $$P(x_1 \underset{T}\longrightarrow x_2)=\vert A \vert ^2$$?  Here are some:

*  $$L$$ is probably a much simpler function than $$P$$. First of all because, as opposed to $$P$$, it doesn't depend on the time $$T$$. But much more than this, by using the PIF we have incorporated domain knowledge into the prediction problem, which means that $$L$$ can be a smaller neural network than $$P$$. Also, $$L$$ represents a much simpler object than $$P$$: it describes the local and time-invariant behavior of the physical system.
* From a physical point of view, the Lagrangian is probably the most fundamental description we can have about a physical system. In fact, a typical workflow in theoretical physics is to ask which symmetries we think our system has, and then to write down an analytical expression for a Lagrangian that obeys these symmetries, and derive from this Lagrangian the probabilities for measurements. It turns out that this recipe sometimes gives amazing results in terms of the prediction accuracy.
* More practically, given a Lagrangian, we can quite easily adjust it to a new scenario. For example, if we trained the Lagrangian on data obtained from a system without an electric field, it could be easy to obtain a new Lagrangian with an electric field added (simply by adding the appropriate term to the Lagrangian)

#### What about a non-quantum system?

In fact, for classical physics (that is, non-quantum physics) there is a simpler version for the PIF. As opposed to quantum mechanics, classical physics is deterministic, and a particles moves on well defined trajectories in space even when they are not observed (actually, I would say that also quantum physics is deterministic, but it feels random due to the "many worlds", but never mind this). Therefore, the classical mechanics version is a formula for finding this trajectory. It basically says that if a particles moved from $$x_1$$ to $$x_2$$ in time $$T$$, the trajectory that the particle took was the the trajectory that minimizes the integral $$\displaystyle{\int\limits_{0}^{T}dt\ L(x(t),\dot{x}(t)) }$$ among all trajectories: 

$$x=\underset{x'}{\arg\min}\displaystyle{\int\limits_{0}^{T}dt\ L(x'(t),\dot{x}'(t)) }$$

This is called the *principle of least action*. 

We could think about how to use this for a learning objective, similarly to what we did with PIF.

(By the way, notice this hand-waving fact: if $$x$$ is the $$\arg\min$$ , then in some sense we can say that the derivative with respect to $$x$$ is zero (recall that $$x$$ is not a number but a trajectory, so this derivative needs an appropriate definition), which means that the integral is almost constant for small changes of $$x$$, which is exactly what the stationary phase approximation seeks for.

#### What are concrete applications?

One idea for an application is protein folding prediction. As you may know, proteins are important building blocks in all living creatures, and they take part in most functions of the body and cells. The function of a protein molecule is largely determined by its 3d structure (that is, the location of each of the atoms forming the molecule), and to some extent the 3d structure of all proteins of the same kind are exactly the same, no matter what the external conditions are. This is called [*Anfinsen's dogma* ](https://en.wikipedia.org/wiki/Anfinsen%27s_dogma). You can find large and open datasets of 3d structures proteins of proteins online. Using Anfinsen's dogma, we can assume that no matter what the initial state of the protein was, the final state will be the same. Therefore, we can form a dataset of tuples $$(x_1,x_2,T)$$  where $$x_1$$ is a randomly sampled initial configuration of the protein, $$x_2$$ is the known final configuration, and $$T$$ is a random time (which should be long enough to let the protein fold into its final stable state). From this dataset we can train a model to find the Lagrangian. (although, I'm not sure whether this problem should be treated as a quantum problem, and moreover, we would need to extend the PIF to the multi-particle version, since the protein is built from many atoms). 

#### What about learning the Hamiltonian instead of the Lagrangian ?

In physics, the Hamiltonian is an object which is the infinitesimal generator of the dynamics. Very roughly speaking, this means that if a particle is in location $$x_1$$ and has momentum $$p_1$$, then by applying the Hamiltonian $$H$$ (which is a function of $$x_1$$ and $$p_1$$), and multiplying by a small number $$dt$$, we get the displacement of the particle after time $$dt$$. 

Can we learn $$H$$ instead of $$L$$? Sure. This requires a learning scheme which is a bit different. Instead of fixing $$x_1$$ and $$x_2$$ and sampling trajectories between them, we fix only $$x_1$$, and we generate a trajectory by applying $$H$$ multiple times that represent a time lapse of $$T$$. The result is some final location $$\hat{x}_2$$. We then take as a loss function the distance between $$\hat{x}_2$$ and the true $$x_2$$, and use this loss to update the learned Hamiltonian.

In fact, I have taken this approach exactly in a project I did of learning a protein folding predictor. I plan to have a blog post about this.

### Idea #2: Resemblance of the PIF to the reinforcement learning objective

Here is another version of the PIF. Now, instead of calculating the amplitude $$A$$, we calculate something else, whose meaning is quite subtle: It is the derivative of $$A$$ with respect to the magnitude of a "kick" that the particle receives at time $$t$$, where $$0\leq t \leq T$$. The derivative is evaluated at the zero magnitude of the kick. The kick can be of different sorts, for example: a small change in momentum or a small change in position at time $$t$$. In general, let's call the operator that "generates" the kick $$Q$$. By the way, it turns out that a change in position along a specific direction is generated by the momentum operator in that direction, and a change in momentum in some direction is generated by the position operator in that direction. Then, I denote the thing that we want to calculate $$\langle x_2,T \vert Q,t\vert x_1,0\rangle$$. It turns out that the formula for this is:

$$\langle x_2,T \vert Q,t\vert x_1,0\rangle=\displaystyle{\int\limits_{x(0)=x_1}^{x(T)=x_2} \mathcal{D}x\ Q(t)\exp\left( i \displaystyle{\int\limits_{0}^{T}dt\ L(x(t),\dot{x}(t)) } \right) }$$

So, for example, if $$Q$$ is the position operator along some axis, then in the path integral we added a factor $$Q(t)$$ which is the position of the particle at time $$t$$ along the axis, on the trajectory $$x$$. (so in fact, in this case, $$Q(t)$$ is just one of the components of $$x(t)$$, corresponding to the axis).

More generally, if there are several kicks at different times $$t_1,t_2,...,t_k$$, such that $$0\leq t_1\leq...\leq t_k \leq T$$, then the formula is:

$$\langle x_2,T \vert Q,t_k,...,t_1\vert x_1,0\rangle=\displaystyle{\int\limits_{x(0)=x_1}^{x(T)=x_2} \mathcal{D}x\ Q(t_k)...Q(t_1)\exp\left( i \displaystyle{\int\limits_{0}^{T}dt\ L(x(t),\dot{x}(t)) } \right) }$$

There is one more step I want to take, which I haven't actually seen done elsewhere, and I'm not going to be careful about doing it. I want to get to the limit of a continuous kick at all times. So instead of $$Q(t)$$ I write $$\exp\left(\log  Q(t) \right)$$, (I am not worrying for now about negative values of $$Q(t)$$). And the the multiplication of all $$Q(t)$$ values becomes a single exponent of the sum of all $$\log Q(t)$$ values. I denote $$q(t)=\log Q(t)$$. In the continuous kick limit, this sum can be written as an integral. So, :

$$\langle x_2,T \vert Q,\text{continuous}\vert x_1,0\rangle=\displaystyle{\int\limits_{x(0)=x_1}^{x(T)=x_2} \mathcal{D}x\ \exp\left( \displaystyle{\int\limits_{0}^{T}dt\ q(t) } \right)\exp\left( i \displaystyle{\int\limits_{0}^{T}dt\ L(x(t),\dot{x}(t)) } \right) }$$ 

Ok. What does this have to do with reinforcement learning? Well, notice that the basic objective in RL is :

$$J(\pi)=\displaystyle{\int \mathcal{D} x\ R(x)P(x\vert\pi)}$$

where $$\pi$$ is the agent's policy which is to be optimized, the integral $$\int\mathcal{D}x$$ is an integral over all possible trajectories of the agent, $$P(x\vert\pi)$$ is the probability of the trajectory $$x$$ given the policy $$\pi$$ (and given the environment dynamics), and $$R(x)$$ is the accumulated rewards of the trajectory. 

In this analogy, the trajectory of an agent corresponds to the trajectory of the particle. The probability of a trajectory of the agent corresponds to the imaginary factor $$\exp{(i\int dt\ L)}$$ for the particle. The policy $$\pi$$ corresponds to the parts of $$L$$ which are controllable by an experimenter or engineer. The environment of the agent corresponds to the parts of $$L$$ which are not controllable. The reward accumulated along the trajectory correspond to the factor $$\exp(\int dt \ q)$$. 

But do $$J(\pi)$$ and $$\langle x_2,T \vert Q,\text{continuous}\vert x_1,0\rangle$$ correspond nicely? The former is something we would like to optimize, but is that true also with the latter? I don't actually see how this would be true,  although these PIF formulas are in use by physicist to calculate probabilities of various events (these formulas are the basis for Feynman diagrams, which is a method for calculating probabilities of events in particle physics and other subfields). So maybe there is a way to take this analogy a step further and think of a way to implement RL algorithms in this physical context.