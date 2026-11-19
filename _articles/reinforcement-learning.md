---
title: "Reinforcement Learning"
category: "Machine Learning"
order: 3
description: "Bellman targets, policy gradients, and what modern RL objectives constrain."
---

In a Markov decision process, the policy $$\pi_\theta(a\mid s)$$ chooses an action and the environment returns a reward and a new state. We train the policy to maximize expected discounted return.

$$
G_t=\sum_{k=t}^{T-1}\gamma^{k-t}r_k,\qquad
J(\theta)=\mathbb E_{\tau\sim\pi_\theta}[G_0].
$$

* TOC
{:toc}

The examples use PyTorch and take rollout data as input. In language-model batches, the axes are prompts, responses per prompt, and tokens. The Boolean mask selects response tokens. Methods named `loss» return objectives to minimize. [Download the Python source]({{ '/assets/reinforcement-learning/rl_objectives.py' | relative_url }})

~~~python
import math

import torch
from torch.nn import functional as F


def masked_mean(values, mask, dim=None):
    total = values.masked_fill(~mask, 0).sum(dim=dim)
    return total / mask.sum(dim=dim).clamp_min(1)


def log_ratio(logp, old_logp, mask):
    return (logp - old_logp.detach()).masked_fill(~mask, 0)


def clipped_objective(log_w, advantage, lower, upper):
    lo = torch.as_tensor(lower, dtype=log_w.dtype, device=log_w.device).log()
    hi = torch.as_tensor(upper, dtype=log_w.dtype, device=log_w.device).log()
    bounded = torch.where(advantage > 0, log_w.minimum(hi), log_w.maximum(lo))
    bounded = bounded.masked_fill(advantage == 0, 0)
    return bounded.exp() * advantage
~~~

## Q-learning

The Bellman equation uses the best future action value as a target. Each sample moves one table entry toward that target by a fraction $$\alpha$$.

$$
\begin{aligned}
Q^*(s,a)&=\mathbb E[r+\gamma(1-d)\max_{a'} Q^*(s',a')\mid s,a],\\
q_{\rm target}&=r+\gamma(1-d)\max_{a'}Q(s',a'),\\
Q(s,a)&\leftarrow Q(s,a)+\alpha\big(q_{\rm target}-Q(s,a)\big).
\end{aligned}
$$

Here $$d=1$$ marks a terminal transition. Monte Carlo uses the observed return $$G_t$$ as its target. A temporal-difference target bootstraps from the next state.

~~~python
class QLearning:
    def __init__(self, states, actions, lr=0.1, gamma=0.99):
        self.q = torch.zeros(states, actions)
        self.lr, self.gamma = lr, gamma

    def act(self, state, epsilon=0.1):
        if torch.rand(()) < epsilon:
            return int(torch.randint(self.q.shape[1], ()))
        return int(self.q[state].argmax())

    def update(self, state, action, reward, next_state, terminal):
        target = reward
        if not terminal:
            target = reward + self.gamma * self.q[next_state].max()
        self.q[state, action] += self.lr * (target - self.q[state, action])
~~~

## REINFORCE

Differentiate the probability of a sampled trajectory to obtain a policy update. Reward-to-go credits each action using rewards that follow it.

$$
\begin{aligned}
p_\theta(\tau)
 &=p(s_0)\prod_{t=0}^{T-1}\pi_\theta(a_t\mid s_t)
       P(s_{t+1},r_t\mid s_t,a_t),\\
\nabla_\theta J
 &=\mathbb E_{\tau\sim\pi_\theta}[G_0\nabla_\theta\ln p_\theta(\tau)]\\
 &=\mathbb E_{\tau\sim\pi_\theta}
 \left[\sum_{t=0}^{T-1}\gamma^tG_t\,\nabla_\theta\ln\pi_\theta(a_t\mid s_t)\right].
\end{aligned}
$$

The gradient of the environment term is zero. Conditional on the history through $$s_t$$, the score has mean zero, so terms from rewards before $$t$$ cancel.

~~~python
class REINFORCE:
    def loss(self, logp, rewards, gamma=0.99):
        rewards = rewards.detach()
        returns = torch.zeros_like(rewards)
        future = rewards.new_zeros(())
        for t in range(len(rewards) - 1, -1, -1):
            future = rewards[t] + gamma * future
            returns[t] = future
        discount = gamma ** torch.arange(len(rewards), dtype=rewards.dtype, device=rewards.device)
        return -(logp * discount * returns).sum()
~~~

## Actor-critic and GAE

A state-value baseline centers the policy update to reduce variance. Generalized Advantage Estimation (GAE) uses $$\lambda$$ to control how much later prediction errors contribute.

$$
\begin{aligned}
Q^\pi(s,a)&=\mathbb E_\pi[G_t\mid s_t=s,a_t=a],\\
V^\pi(s)&=\mathbb E_\pi[G_t\mid s_t=s],\quad A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s),\\
\mathbb E_{a\sim\pi_\theta}
 [b(s)\nabla_\theta\ln\pi_\theta(a\mid s)]&=b(s)\nabla_\theta\sum_a\pi_\theta(a\mid s)=0,\\
\delta_t&=r_t+\gamma(1-d_t)V(s_{t+1})-V(s_t),\\
\hat A_t&=\delta_t+\gamma\lambda(1-d_t)\hat A_{t+1},\qquad \hat A_T=0,\\
\hat G_t^\lambda&=V(s_t)+\hat A_t .
\end{aligned}
$$

The actor uses $$\hat A_t$$ as a fixed weight. The critic fits the fixed target $$\hat G_t^\lambda$$ by squared error.

~~~python
class ActorCritic:
    def advantage(self, reward, value, next_value, terminal, gamma=0.99, lam=0.95):
        bootstrap = next_value.detach().masked_fill(terminal, 0)
        delta = reward.detach() + gamma * bootstrap - value.detach()
        advantage = torch.zeros_like(delta)
        future = torch.zeros_like(delta[..., 0])
        for t in range(delta.shape[-1] - 1, -1, -1):
            future = delta[..., t] + gamma * lam * (~terminal[..., t]) * future
            advantage[..., t] = future
        return advantage

    def loss(self, logp, reward, value, next_value, terminal, gamma=0.99, lam=0.95):
        advantage = self.advantage(reward, value, next_value, terminal, gamma, lam)
        target = value.detach() + advantage
        actor = -(advantage * logp).mean()
        critic = 0.5 * (value - target).square().mean()
        return actor + critic
~~~

## TRPO

Let $$\mu$$ be the policy that collected the samples. Its state distribution is held fixed in the surrogate, and each action is reweighted by its probability ratio.

$$
w_t=\frac{\pi_\theta(a_t\mid s_t)}{\mu(a_t\mid s_t)},\qquad
J_{\rm surr}(\theta)=\mathbb E_{s_t,a_t\sim\mu}[w_t\hat A_t].
$$

TRPO bounds the mean KL change from $$\mu$$ to the new policy. A local linear objective with a quadratic KL approximation gives the candidate step.

$$
\begin{aligned}
\overline D(\theta)&=\mathbb E_{s_t\sim\mu}
 D_{\rm KL}\big(\mu(\cdot\mid s_t)\|\pi_\theta(\cdot\mid s_t)\big)
 \le\delta,\\
\max_\Delta\,g^\top\Delta&\quad\text{subject to}\quad
          \tfrac12\Delta^\top F\Delta\le\delta,\\
g&=\nabla J_{\rm surr}(\theta_{\rm old}),\quad
F=\nabla^2\overline D(\theta_{\rm old}),\\
g=\nu F\Delta&\quad\Longrightarrow\quad
\Delta^*=\sqrt{\frac{2\delta}{g^\top F^{-1}g}}F^{-1}g .
\end{aligned}
$$

The example uses a dense Hessian and checks the actual surrogate and KL during backtracking.

~~~python
class TRPO:
    def step(self, params, surrogate, mean_kl, limit=0.01, damping=1e-5):
        old = params.detach()
        gradient = torch.func.grad(surrogate)(old)
        fisher = torch.func.hessian(mean_kl)(old)
        eye = torch.eye(old.numel(), dtype=old.dtype, device=old.device)
        direction = torch.linalg.solve(fisher + damping * eye, gradient)
        curvature = direction @ fisher @ direction
        if curvature <= 0:
            return old
        delta = direction * (2 * limit / curvature).sqrt()
        base = surrogate(old)
        for i in range(10):
            candidate = old + (0.5 ** i) * delta
            if mean_kl(candidate) <= limit and surrogate(candidate) > base:
                return candidate
        return old
~~~

## PPO

PPO clips the gain in the surrogate for each sample. A positive advantage saturates above the upper bound, and a negative advantage saturates below the lower one.

$$
\begin{aligned}
C_{\ell,u}(w,A)&=\min\big(wA,\operatorname{clip}(w,\ell,u)A\big)\\
&=\begin{cases}
A\min(w,u),&A\ge0,\\
A\max(w,\ell),&A<0 ,
\end{cases}\\
J_{\rm PPO}&=\mathbb E_\mu[C_{1-\epsilon,1+\epsilon}(w_t,\hat A_t)] .
\end{aligned}
$$

![PPO surrogate for positive and negative advantages]({{ '/assets/reinforcement-learning/ppo-clipped-surrogate.png' | relative_url }}){: width="650"}

The flat branch gives zero gradient for that sample. Shared parameters can still move its probability when other samples update the policy.

~~~python
class PPO:
    def loss(self, logp, old_logp, advantage, mask, eps=0.2):
        log_w = log_ratio(logp, old_logp, mask)
        objective = clipped_objective(log_w, advantage.detach(), 1 - eps, 1 + eps)
        return -masked_mean(objective, mask)
~~~

## Rewards for language models

A state is a prompt and its generated prefix, and an action is the next token. A verifier can score a complete response by exact match. A reference-policy penalty can also shape the token rewards before GAE.

$$
\begin{aligned}
R(x,y)&=\mathbf1_{\{\operatorname{answer}(y)=y^\star\}},\\
r_t&=\mathbf1_{\{t=T\}}R(x,y)
 -\beta\ln\frac{\mu(y_t\mid x,y_{<t})}{\pi_{\rm ref}(y_t\mid x,y_{<t})},\\
\sum_{t=1}^T \ln\frac{\mu(y_t\mid x,y_{<t})}
                       {\pi_{\rm ref}(y_t\mid x,y_{<t})}
 &= \ln\frac{\mu(y\mid x)}{\pi_{\rm ref}(y\mid x)} .
\end{aligned}
$$

The rollout policy $$\mu$$ provides samples and sets the denominator of the PPO ratio. The reference policy $$\pi_{\rm ref}$$ sets the target of the KL penalty.

## GRPO

For each prompt, compare $$G$$ response rewards within their group. Each token in response $$i$$ gets the same advantage $$A_i$$, and each response gets the same total weight.

$$
\begin{aligned}
\bar R&=\frac1G\sum_i R_i,\qquad
A_i=\frac{R_i-\bar R}{\sqrt{G^{-1}\sum_j(R_j-\bar R)^2}+\varepsilon},\\
J_{\rm GRPO}
 &=\mathbb E_\mu\left[\frac1G\sum_i\frac1{T_i}
  \sum_{t=1}^{T_i}
    \big(C_{1-\epsilon,1+\epsilon}(w_{it},A_i)-\beta k_{it}\big)\right],\\
u_{it}&=\ln\frac{\pi_{\rm ref}(y_{it}\mid s_{it})}{\pi_\theta(y_{it}\mid s_{it})},
\qquad k_{it}=e^{u_{it}}-u_{it}-1.
\end{aligned}
$$

The optional penalty $$k_{it}$$ estimates $$D_{\rm KL}(\pi_\theta\|\pi_{\rm ref})$$ in expectation under the current policy at a fixed prefix.

~~~python
class GRPO:
    def advantage(self, rewards):
        rewards = rewards.detach()
        centered = rewards - rewards.mean(dim=-1, keepdim=True)
        return centered / (rewards.std(dim=-1, keepdim=True, correction=0) + 1e-6)

    def loss(self, logp, old_logp, rewards, mask, eps=0.2, ref_logp=None, beta=0.0):
        advantage = self.advantage(rewards).unsqueeze(-1)
        log_w = log_ratio(logp, old_logp, mask)
        objective = clipped_objective(log_w, advantage, 1 - eps, 1 + eps)
        if ref_logp is not None and beta:
            u = -(logp - ref_logp.detach()).masked_fill(~mask, 0)
            objective = objective - beta * (u.expm1() - u)
        return -masked_mean(objective, mask, dim=-1).mean()
~~~

## Dr. GRPO

Center the rewards and use a fixed token budget $$L$$ as the divisor. This keeps the gradient scale independent of the group's reward standard deviation and removes the per-response factor $$1/T_i$$.

$$
A_i=R_i-\bar R,\qquad
J_{\rm Dr.GRPO}=
\mathbb E_\mu\left[\frac{1}{GL}\sum_{i,t}
 C_{1-\epsilon,1+\epsilon}(w_{it},A_i)\right].
$$

~~~python
class DrGRPO:
    def loss(self, logp, old_logp, rewards, mask, budget, eps=0.2):
        centered = (rewards - rewards.mean(dim=-1, keepdim=True)).detach()
        log_w = log_ratio(logp, old_logp, mask)
        objective = clipped_objective(log_w, centered.unsqueeze(-1), 1 - eps, 1 + eps)
        return -objective.masked_fill(~mask, 0).sum(dim=-1).mean() / budget
~~~

## DAPO

Raise the upper clipping threshold to give positive-advantage tokens more room to grow. Use one denominator for all response tokens in a prompt and keep groups containing both correct and incorrect answers.

$$
\begin{aligned}
0&<\sum_i\mathbf1_{\{\operatorname{correct}(y_i)\}}<G,\qquad
\epsilon_{\rm high}>\epsilon_{\rm low},\\
J_{\rm DAPO}&=\mathbb E_{\rm kept}
 \left[\frac{\sum_{i,t}C_{1-\epsilon_{\rm low},\,1+\epsilon_{\rm high}}(w_{it},A_i)}
 {\sum_i T_i}\right].
\end{aligned}
$$

A soft length penalty changes the response reward before computing the group advantage. With generation cap $$L_{\max}$$ and buffer $$B>0$$,

$$
R_i^{\rm shaped}=R_i-
 \operatorname{clip}\!\left(\frac{T_i-(L_{\max}-B)}{B},\,0,\,1\right).
$$

![AIME accuracy and generation entropy with and without Clip-Higher]({{ '/assets/reinforcement-learning/dapo-clip-higher.png' | relative_url }}){: width="740"}

~~~python
class DAPO(GRPO):
    def shaped_reward(self, reward, length, cap, buffer):
        return reward - ((length - (cap - buffer)) / buffer).clamp(0, 1)

    def loss(self, logp, old_logp, rewards, correct, mask, eps_low=0.20, eps_high=0.28):
        keep = correct.any(dim=-1) & ~correct.all(dim=-1)
        valid = mask & keep[:, None, None]
        advantage = self.advantage(rewards).unsqueeze(-1)
        log_w = log_ratio(logp, old_logp, valid)
        objective = clipped_objective(log_w, advantage, 1 - eps_low, 1 + eps_high)
        per_prompt = masked_mean(objective, valid, dim=(-2, -1))
        return -masked_mean(per_prompt, keep)
~~~

## GSPO

Use the geometric mean of a response's token ratios. The clip acts once per response, at the same level as the reward.

$$
\begin{aligned}
s_i&=\exp\!\left(\frac1{T_i}\sum_t\ln w_{it}\right)
   =\left(\frac{\pi_\theta(y_i\mid x)}{\mu(y_i\mid x)}\right)^{1/T_i},\\
J_{\rm GSPO}&=\mathbb E_\mu\left[\frac1G\sum_i
                   C_{1-\epsilon,1+\epsilon}(s_i,A_i)\right].
\end{aligned}
$$

~~~python
class GSPO(GRPO):
    def loss(self, logp, old_logp, rewards, mask, eps=0.002):
        log_w = log_ratio(logp, old_logp, mask)
        log_sequence_ratio = masked_mean(log_w, mask, dim=-1)
        advantage = self.advantage(rewards)
        objective = clipped_objective(log_sequence_ratio, advantage, 1 - eps, 1 + eps)
        return -objective.mean()
~~~

## CISPO

Clip the importance weight and hold it fixed during differentiation. A token at the cap can still contribute through its log-probability gradient.

$$
\begin{aligned}
\bar w_{it}&=\operatorname{stopgrad}\!\big(\operatorname{clip}(w_{it},\ell,u)\big),\\
J_{\rm CISPO}&=\mathbb E_\mu\left[
 \frac{\sum_{i,t}\bar w_{it} A_i\ln\pi_\theta(y_{it}\mid s_{it})}{\sum_i T_i}
\right].
\end{aligned}
$$

~~~python
class CISPO(GRPO):
    def loss(self, logp, old_logp, rewards, mask, lower=0.0, upper=2.0):
        log_w = log_ratio(logp, old_logp, mask).detach()
        weight = log_w.clamp_max(math.log(upper)).exp().clamp_min(lower)
        advantage = self.advantage(rewards).unsqueeze(-1)
        objective = weight * advantage * logp.masked_fill(~mask, 0)
        return -masked_mean(objective, mask, dim=(-2, -1)).mean()
~~~

## SAPO

Use a sigmoid to shrink the slope smoothly as the token ratio moves away from one. The temperature can differ for positive and negative advantages.

$$
\begin{aligned}
v&=\sigma\big(\tau(w-1)\big),\qquad f_\tau(w)=4v/\tau,\\
f_\tau'(w)&=4v(1-v),\qquad f_\tau'(1)=1,\\
J_{\rm SAPO}&=\mathbb E_\mu\left[\frac1G\sum_i\frac1{T_i}
                  \sum_t A_i f_{\tau_i}(w_{it})\right].
\end{aligned}
$$

~~~python
class SAPO(GRPO):
    def loss(self, logp, old_logp, rewards, mask, tau_pos=1.0, tau_neg=1.05):
        advantage = self.advantage(rewards).unsqueeze(-1)
        tau = torch.where(advantage > 0, tau_pos, tau_neg)
        ratio = log_ratio(logp, old_logp, mask).clamp_max(30).exp()
        gate = 4 * torch.sigmoid(tau * (ratio - 1)) / tau
        return -masked_mean(gate * advantage, mask, dim=-1).mean()
~~~

## DCPO

Let $$q$$ and $$p=wq$$ be the old and new probabilities of a sampled token. DCPO bases its ratio bounds on the probability-weighted change $$\lvert w-1\rvert p$$, so rare tokens can have a wider ratio range.

$$
\begin{aligned}
|w-1|p&\le\epsilon
   \quad\Longrightarrow\quad q|w^2-w|\le\epsilon,\\
\ell(q)&=\tfrac12+\tfrac12\sqrt{\max(1-4\epsilon_{\rm low}/q,0)},\\
u(q)&=\tfrac12+\tfrac12\sqrt{1+4\epsilon_{\rm high}/q}.
\end{aligned}
$$

The code takes precomputed advantages, caps the ratio, and uses these adaptive bounds in the clipped objective.

~~~python
class DCPO:
    def loss(self, logp, old_logp, advantage, mask, eps_low=0.16, eps_high=0.20, cap=10.0):
        q = old_logp.detach().masked_fill(~mask, 0).exp()
        q = q.clamp_min(torch.finfo(q.dtype).tiny)
        lower = 0.5 + 0.5 * (1 - 4 * eps_low / q).clamp_min(0).sqrt()
        upper = (0.5 + 0.5 * (1 + 4 * eps_high / q).sqrt()).clamp_max(cap)
        log_w = log_ratio(logp, old_logp, mask).clamp_max(math.log(cap))
        objective = clipped_objective(log_w, advantage.detach().unsqueeze(-1), lower, upper)
        return -masked_mean(objective, mask, dim=-1).sum(dim=-1).mean()
~~~

## VESPO

The full sequence ratio can vary sharply when rollouts are stale. A smooth weight suppresses both tails and leaves the weight equal to one at $$W_i=1$$.

$$
\begin{aligned}
W_i &= \frac{\pi_\theta(y_i\mid x)}{\mu(y_i\mid x)}
      =\exp\left(\sum_t\ln w_{it}\right),\\
\phi(W_i)&=W_i^{c_1}e^{c_2(1-W_i)},\qquad c_1,c_2>0,\\
\nabla J_{\rm VESPO}&=\mathbb E_\mu\left[\frac1G\sum_i
  \phi(W_i)(R_i-\bar R)\nabla_\theta\ln\pi_\theta(y_i\mid x)\right].
\end{aligned}
$$

The code picks the parameters by advantage sign and detaches the weight to implement this score gradient.

~~~python
class VESPO:
    def loss(self, logp, behavior_logp, rewards, mask):
        advantage = (rewards - rewards.mean(dim=-1, keepdim=True)).detach()
        c1 = torch.where(advantage > 0, 2.0, 3.0)
        c2 = torch.where(advantage > 0, 3.0, 2.0)
        log_w = log_ratio(logp, behavior_logp, mask).sum(dim=-1).detach()
        weight = (c1 * log_w + c2 * (1 - log_w.exp())).exp()
        sequence_logp = logp.masked_fill(~mask, 0).sum(dim=-1)
        return -(weight * advantage * sequence_logp).mean()
~~~

## DPO

DPO fits a fixed preferred and rejected pair by comparing the policy's log-odds relative to a reference. The reward normalizer cancels within the pair.

$$
\begin{aligned}
\pi^*(y\mid x)&=\pi_{\rm ref}(y\mid x)e^{R(x,y)/\beta}/Z(x),\\
R(x,y)&=\beta\ln\frac{\pi^*(y\mid x)}{\pi_{\rm ref}(y\mid x)}+\beta\ln Z(x),\\
\Pr(y^+\succ y^-\mid x)&=\sigma(R(x,y^+)-R(x,y^-)),\\
\ell_\theta(y)&=\ln\frac{\pi_\theta(y\mid x)}{\pi_{\rm ref}(y\mid x)},\\
L_{\rm DPO}&=-\mathbb E\ln\sigma\big(\beta[\ell_\theta(y^+)-\ell_\theta(y^-)]\big).
\end{aligned}
$$

~~~python
class DPO:
    def loss(self, chosen_logp, rejected_logp, ref_chosen, ref_rejected, beta=0.1):
        difference = chosen_logp - rejected_logp
        ref_difference = (ref_chosen - ref_rejected).detach()
        return -F.logsigmoid(beta * (difference - ref_difference)).mean()
~~~
