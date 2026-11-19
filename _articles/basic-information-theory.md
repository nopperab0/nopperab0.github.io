---
title: "basic Information theory"
category: "Machine Learning"
order: 2
description: "Entropy, mutual information, cross-entropy, KL divergence, and a few uses in machine learning."
---

Start with a discrete random variable $$X$$ and its distribution $$p(x)$$. We use natural logs for nats and base-two logs for bits, with $$0\ln0=0$$.

* TOC
{:toc}

## Entropy

The information in an outcome is $$-\ln p(x)$$, so independent outcomes contribute additively. Entropy is its average.

$$
H(X)=-\sum_x p(x)\ln p(x),\qquad
h_b(u)=-u\ln u-(1-u)\ln(1-u).
$$

A fair bit has entropy $$h_b(1/2)=\ln2$$ nats. Using $$p(x,y)=p(x)p(y\mid x)$$ gives

$$
\begin{aligned}
H(X,Y)&=-\sum_{x,y}p(x,y)\ln p(x,y),\\
H(Y\mid X)&=-\sum_{x,y}p(x,y)\ln p(y\mid x),\\
H(X,Y)&=H(X)+H(Y\mid X),\qquad
H(X_{1:T})=\sum_{t=1}^T H(X_t\mid X_{<t}).
\end{aligned}
$$

For a density $$p(x)$$, differential entropy replaces the sum with an integral. For a stationary sequence, entropy rate measures average entropy per symbol over a long block.

$$
h(X)=-\int p(x)\ln p(x)\,dx,\qquad
\bar H=\lim_{T\to\infty}\frac{H(X_{1:T})}{T}.
$$

## Mutual information

Mutual information is the reduction in uncertainty about $$Y$$ after observing $$X$$.

$$
\begin{aligned}
I(X;Y)&=H(Y)-H(Y\mid X)\\
      &=H(X)+H(Y)-H(X,Y)\\
      &=\sum_{x,y}p(x,y)\ln\frac{p(x,y)}{p(x)p(y)}.
\end{aligned}
$$

Mutual information is zero exactly when $$X$$ and $$Y$$ are independent. For a Markov chain $$X\to Y\to Z$$, conditioning on $$Y$$ gives $$I(X;Z\mid Y)=0$$.

$$
\begin{aligned}
I(X;Y\mid Z)&=H(Y\mid Z)-H(Y\mid X,Z),\\
I(X;Y)&=I(X;Y,Z)=I(X;Z)+I(X;Y\mid Z)\ge I(X;Z).
\end{aligned}
$$

## Coding and capacity

In bits, Shannon code lengths $$\ell(x)=\lceil-\log_2p(x)\rceil$$ give an average length within one bit of the prefix-coding lower bound.

$$
H_2(X)\le\mathbb E[\ell(X)]<H_2(X)+1.
$$

For a binary symmetric channel that flips a bit with probability $$\epsilon$$, the output entropy is at most $$\ln2$$ and uniform inputs attain it. [Shannon, A Mathematical Theory of Communication](https://web.mit.edu/6.441/spring05/reading/shannon1948.pdf)

$$
C=\max_{p_X}I(X;Y)=\max_{p_X}H(Y)-h_b(\epsilon)
=\ln2-h_b(\epsilon).
$$

## Cross-entropy and KL divergence

If data follow $$p$$ and the model assigns probabilities $$q$$, the average negative log probability under the model is the cross-entropy.

$$
\begin{aligned}
H(p,q)&=-\sum_x p(x)\ln q(x),\\
D_{\rm KL}(p\|q)&=\sum_x p(x)\ln\frac{p(x)}{q(x)}
                 =H(p,q)-H(p).
\end{aligned}
$$

When $$q$$ is positive wherever $$p$$ is positive, Jensen's inequality gives nonnegativity. Otherwise the divergence is infinite.

$$
D_{\rm KL}(p\|q)=\mathbb E_p[-\ln(q/p)]
\ge-\ln\sum_{x:p(x)>0}q(x)\ge0.
$$

Equality holds exactly when $$p=q$$. Mutual information is $$D_{\rm KL}(p_{XY}\|p_Xp_Y)$$.

### Classification and language models

With a one-hot label, cross-entropy is the negative log probability of the observed class. Averaging over examples gives the negative log-likelihood per example.

$$
\begin{aligned}
\mathcal L(\theta)&=-\frac1n\sum_i\ln q_\theta(y_i\mid x_i),\\
q_j&=\frac{e^{z_j}}{\sum_k e^{z_k}},\qquad
H(t,q)=\ln\sum_k e^{z_k}-\sum_k t_k z_k,\\
\partial_{z_j}H(t,q)&=q_j-t_j .
\end{aligned}
$$

For autoregressive sequences, perplexity is the exponential of the mean token loss.

$$
\mathrm{PPL}=\exp\!\left(-\frac1T\sum_{t=1}^T\ln q(x_t\mid x_{<t})\right).
$$

## Natural gradient

Suppose we want a small update measured by how much the model distribution changes. For a smooth model, the second-order approximation of KL gives

$$
\begin{aligned}
s&=\nabla_\theta\ln p_\theta(X),\qquad F=\mathbb E_{p_\theta}[ss^\top],\\
D_{\rm KL}(p_\theta\|p_{\theta+\Delta})
&\approx\tfrac12\Delta^\top F\Delta.
\end{aligned}
$$

For loss gradient $$g=\nabla_\theta L$$ and invertible $$F$$, minimize the linearized loss plus the KL penalty. [Amari, Natural Gradient Works Efficiently in Learning](https://doi.org/10.1162/089976698300017746)

$$
\min_\Delta\left(g^\top\Delta+\frac{1}{2\eta}\Delta^\top F\Delta\right)
\ \Rightarrow\ g+\frac{F\Delta}{\eta}=0
\ \Rightarrow\ \Delta=-\eta F^{-1}g.
$$

For a policy, take the score $$s=\nabla_\theta\ln\pi_\theta(a\mid x)$$ and average over visited states and sampled actions. Maximizing expected reward $$J$$ then uses $$F^{-1}\nabla J$$ as the natural policy-gradient direction.

## KL-regularized policies

For a fixed input, reward $$r(a)$$ and reference policy $$\pi_0$$, define $$\pi^*$$ by reweighting the reference by exponentiated reward.

$$
\begin{aligned}
Z&=\sum_a\pi_0(a)e^{r(a)/\beta},\qquad
\pi^*(a)=\pi_0(a)e^{r(a)/\beta}/Z,\\
\mathbb E_\pi[r]-\beta D_{\rm KL}(\pi\|\pi_0)
&=\beta\ln Z-\beta D_{\rm KL}(\pi\|\pi^*).
\end{aligned}
$$

For $$\beta>0$$, nonnegativity of KL makes $$\pi^*$$ optimal.

## On-policy distillation

Fix a prompt and let the student $$p_s$$ generate tokens. One distillation loss compares student and teacher $$p_t$$ using reverse KL at the prefixes the student visits. [Agarwal et al., On-Policy Distillation](https://arxiv.org/abs/2306.13649)

$$
\begin{aligned}
D_{\rm KL}(p_s(Y_{1:T})\|p_t(Y_{1:T}))
&=\mathbb E_{Y\sim p_s}\sum_{i=1}^T
   \ln\frac{p_s(Y_i\mid Y_{<i})}{p_t(Y_i\mid Y_{<i})}\\
&=\sum_{i=1}^T\mathbb E_{Y_{<i}\sim p_s}
   D_{\rm KL}\big(p_s(\cdot\mid Y_{<i})\|p_t(\cdot\mid Y_{<i})\big).
\end{aligned}
$$
