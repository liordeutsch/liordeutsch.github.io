---
layout: post
comments: true
title:  "Nostalgia Post"
excerpt: "Following the first post of the blog, I mention some other things that I did as a teenager, and I'm somewhat proud of."
date:   2019-11-08 17:00:00
mathjax: true
---

In the [first post of this blog]({% post_url 2019-10-14-my-failed-attempt-to-crack-rsa %}), I talked about an idea I had when I was 17. Apparently this put me in a nostalgia spree, and I think that my way out of it will be to post other things that I did as a teenager. So here they are.

---

## Method for finding roots of polynomials
As a teenager, I knew about the [famous theorem by Galois and Abel and Ruffini](https://en.wikipedia.org/wiki/Abel%E2%80%93Ruffini_theorem), that there is no general solution in radicals for a polynomial equation of degree five or above. However, in retrospective, I did not understand what this means. The thing that is obvious to me now, but wasn't obvious back then, is that it is quite easy to find the roots of any polynomial using numerical iterative methods. These methods will get you as close as you want to the roots of the polynomial, but they won't give you a closed expression for the roots.

Anyway, at some point I thought that I had found a way to solve any polynomial equation. Now I understand that this is simply an approximate iterative method, like many others that exist, and that it does not contradict in any way the famous theorem (and it is not actually novel or particularly useful). But back then I was quite excited about this result.

The idea is simple, and was inspired from the Fibonacci Sequence, which is defined by the recursive formula $$a_n=a_{n-1}+a_{n-2}$$ (with $$a_1=a_2=1$$). I knew that the ratio between consecutive Fibonacci numbers has as limit the golden ratio number $$\varphi$$: 
$$\underset{n\rightarrow\infty}{\lim}\frac{a_{n+1}}{a_{n}}=\varphi$$. This number is a root of the polynomial $$x^2-x-1$$.

Looking at the proof of this statement, I realized that it can be extended to an algorithm of general root finding. Suppose that you want to find a solution for the equation  
 $$x^m-\beta_{m-1}x^{m-1}-\beta_{m-2}x^{m-2}-...-\beta_{1}x^{1}-\beta_{0}x^{0}=0$$
 
Then, define a Fibonacci-like sequence as follows:  
$$a_n=\beta_{m-1}a_{n-1}+\beta_{m-2}a_{n-2}+...+\beta_{0}a_{n-m}$$  
and give it some arbitrary initial values (for example, $$a_1=a_2=...=a_{m}=1$$). Then, assuming that the limit $$\underset{n\rightarrow\infty}{\lim}\frac{a_{n+1}}{a_{n}}$$ exists and equals a number $$x_0$$, it turns out that this number is a root of the polynomial, as you can see:  

$$x_0=\underset{n\rightarrow\infty}{\lim}\frac{a_{n+1}}{a_{n}} = $$  
$$ = \underset{n\rightarrow\infty}{\lim}\frac{\beta_{m-1}a_{n}+\beta_{m-2}a_{n-1}+...+\beta_{0}a_{n-m+1}}{a_{n}} $$  
$$ = \underset{n\rightarrow\infty}{\lim} \left( \beta_{m-1} + \beta_{m-1}\frac{a_{n-1}}{a_n} + \beta_{m-2}\frac{a_{n-2}}{a_n} + ... + \beta_0\frac{a_{n-m+1}}{a_n} \right)$$  
Applying the limit to each term separately, and using the fact that  $$\underset{n\rightarrow\infty}{\lim}\frac{a_{n}}{a_{n-k}}=\underset{n\rightarrow\infty}{\lim}\frac{a_{n}}{a_{n-1}}\cdot\frac{a_{n-1}}{a_{n-2}}\cdot ... \frac{a_{n-k+1}}{a_{n-k}} = x_0^k$$, we get  
$$x_0 = \beta_{m-1} + \beta_{m-1}x_0^{-1} + \beta_{m-2}x_0^{-2} + ... + \beta_0x_0^{-(m-1)}$$  
Multiplying both sides by $$x_0^{(m-1)}$$ and rearranging, we get  
$$x_0^m-\beta_{m-1}x_0^{m-1}-\beta_{m-2}x_0^{m-2}-...-\beta_{1}x_0^{1}-\beta_{0}x_0^{0}=0$$  
Which shows that $$x_0$$ is the root of the polynomial. 

Therefore, the algorithm is to calculate as many temrs of the sequence $$a_n$$ as possible, and then to take the ratio between the two last terms. To find another root, we can simply divide the polynomial by $$(x-x_0)$$ and repeat the process.

At some point later in life I found that Euler has mentioned this algorithm in one of his papers, but I cannot find the reference for this now.

---

## An interesting sequence with prime numbers
As an adolescent, I had a weird hobby of searching for patterns in number sequences. I would just define some arbitrary sequence, calculate its terms in Microsoft Excel (why not in a programming language? I now ask myself the same question), and see what happens. It wasn't serious math, rather it was more like recreational experimental math.

There is one sequence I found that I remember in particular, since it involves prime numbers and has a nice limit which I can't explain even in retrospect. In fact, at some point I have even submitted it to the [Online Encyclopedia of Integer Sequences](https://oeis.org/A135026). The sequence is defined as follows:  
Start with the number $$2$$, which is the smallest prime number. Now, on each step, subtract the next prime number, but only if the result is non-negative. Otherwise, add the next prime number. These are the first terms of this sequence: $$2,5,0,7,18,5,22,3,26,55,...$$. 

Most terms in this sequence are either larger than their two neighbors, or smaller than their two neighbors. However, every once in a while there is a term which is larger than its neighbor on the left and smaller than its neighbor on the right. I now focus only on the subsequence formed by the terms. Its first elements are $$7,26,81,270,841,2480,...$$

The interesting thing about this subsequence is that, if you look at the ratio between consecutive terms, it seems to converge to the number $$3$$. This is just an empirical finding that awaits a proof (or a disproof). I don't know how to prove this (and I haven't actually tried much), but I suspect that invoking the prime number theorem can help.

Another thing I remember noticing is that the limit of $$3$$ has a universal character, in the sense that if I repeat the construction above but use other known sequences instead of prime numbers, then $$3$$ appears as the limit many times.

---

## A variation on the quantum Mach-Zehnder experiment
What got me into science as a teenager was obsessive reading for popular science books, a lot of them were about quantum mechanics. There is one particular experiment that was mentioned in many of the books, as it demonstrates quantum "weirdness". This is the [Mach-Zehnder interferometer](https://en.wikipedia.org/wiki/Mach%E2%80%93Zehnder_interferometer) experiment.

The setting is as follows: We have a source of light, which is so dim that it emits a single photon at a time. The photon is then passed through a beam splitter. This is a device that splits a beam of light into two equally-intensive beams that go in two different directions. Also, if the light beam is coherent, the beam splitter causes a phase difference between the two beams (that is, between the two electromagnetic waves). Now, since in our case the incoming beam consists of just a single photon, we need to treat this system using quantum mechanics. In an idealized experiment, the beam splitter will cause the photon to go into a quantum state which is a superposition of going along both paths, with equal probabilities (that is, equally sized amplitudes), and with a phase shift between them. Importantly, the photon does not take a particular path, but rather "takes both path simultaneously" (loosely speaking). 

Along the two paths, we place mirrors and another beam splitter that guide both paths towards two light detectors. The experiment is set up such that, if we used a source of coherent light which was not very dim, the two beams would destructively interfere at one of the detectors and constructively interfere at the second detector. In other words, only one of the two detectors will detect light. The crux of the single photon version of the experiment, is that in it too, only one of the detectors detects light. This is "weird" because, thinking of a photon as a point particle, it's not clear how it can destructively interfere with itself. However, the formulation of quantum mechanics does allow this to happen. Moreover, if we try to squeeze out information (by doing a measurement) about which path the photon did take, we will always discover that the photon took one path or the other, and the interference at the detectors will be gone. The bewildering thing is that the photon behaves as if it is "taking both paths simultaneously" only when we don't try to observe which path it took.

Anyway, my na&iuml;ve idea back then, was to do an indirect measurement of which path the photon took, by making one path longer than the other. Thus, by noting the time from the emission of the photon until the time it was detected, we can tell which path the photon took (the longer path will take a longer time). I thought that, as this is an indirect measurement, the photon would not be affected by it, and therefore we would still see the constructive and destructive interference at the detectors, yet we would know which path the photon took, which is in contradiction to the bewildering statement above.

The truth is that today, even after studying real physics (and not just popular science books...), it's not very clear to me what the outcome of such a thought experiment would be. My guess is that if we make a time difference between the two paths too big, then the interference will be canceled due to no time overlap between the two trajectories arriving at the detector. If the time difference is small enough to keep the overlap, then we would face issues with the time-energy uncertainty principle.

---

## Snake and Tetris
In high school, the programming language we were taught we Pascal. If your reaction to this is "you probably went to high school during the 80's", you are wrong. It was around 2004-2006. 

Anyway, during high school I coded in Pascal, using character graphics, the games Snake and Tetris. Unfortunately I don't have the source code anymore, but the executable still exist so at least I can make GIFs :). 	

<img src="/assets/snake-000001.gif" style="width:30%">
<img src="/assets/tetris-000001.gif" style="width:30%">