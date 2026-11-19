import math
import random

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
        time = torch.arange(logp.shape[-1], dtype=logp.dtype, device=logp.device)
        actor = -(gamma ** time * advantage * logp).sum(dim=-1).mean()
        critic = 0.5 * (value - target).square().mean()
        return actor + critic


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


class DynaQ(QLearning):
    def __init__(self, states, actions, lr=0.1, gamma=0.99, planning_steps=5):
        super().__init__(states, actions, lr, gamma)
        self.model = {}
        self.planning_steps = planning_steps

    def update(self, state, action, reward, next_state, terminal):
        transition = (state, action, reward, next_state, terminal)
        super().update(*transition)
        self.model[state, action] = transition
        for simulated in random.choices(list(self.model.values()), k=self.planning_steps):
            super().update(*simulated)


class NaturalPolicyGradient:
    def direction(self, gradient, fisher, damping=1e-5):
        eye = torch.eye(gradient.numel(), dtype=gradient.dtype, device=gradient.device)
        return torch.linalg.solve(fisher + damping * eye, gradient)


class TRPO(NaturalPolicyGradient):
    def step(self, params, surrogate, mean_kl, limit=0.01, damping=1e-5):
        old = params.detach()
        gradient = torch.func.grad(surrogate)(old)
        fisher = torch.func.hessian(mean_kl)(old)
        direction = self.direction(gradient, fisher, damping)
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


class PPO:
    def loss(self, logp, old_logp, advantage, mask, eps=0.2):
        log_w = log_ratio(logp, old_logp, mask)
        objective = clipped_objective(log_w, advantage.detach(), 1 - eps, 1 + eps)
        return -masked_mean(objective, mask)


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


class DrGRPO:
    def loss(self, logp, old_logp, rewards, mask, budget, eps=0.2):
        centered = (rewards - rewards.mean(dim=-1, keepdim=True)).detach()
        log_w = log_ratio(logp, old_logp, mask)
        objective = clipped_objective(log_w, centered.unsqueeze(-1), 1 - eps, 1 + eps)
        return -objective.masked_fill(~mask, 0).sum(dim=-1).mean() / budget


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


class GSPO(GRPO):
    def loss(self, logp, old_logp, rewards, mask, eps=0.002):
        log_w = log_ratio(logp, old_logp, mask)
        log_sequence_ratio = masked_mean(log_w, mask, dim=-1)
        advantage = self.advantage(rewards)
        objective = clipped_objective(log_sequence_ratio, advantage, 1 - eps, 1 + eps)
        return -objective.mean()


class CISPO(GRPO):
    def loss(self, logp, old_logp, rewards, mask, lower=0.0, upper=2.0):
        log_w = log_ratio(logp, old_logp, mask).detach()
        weight = log_w.clamp_max(math.log(upper)).exp().clamp_min(lower)
        advantage = self.advantage(rewards).unsqueeze(-1)
        objective = weight * advantage * logp.masked_fill(~mask, 0)
        return -masked_mean(objective, mask, dim=(-2, -1)).mean()


class SAPO(GRPO):
    def loss(self, logp, old_logp, rewards, mask, tau_pos=1.0, tau_neg=1.05):
        advantage = self.advantage(rewards).unsqueeze(-1)
        tau = torch.where(advantage > 0, tau_pos, tau_neg)
        ratio = log_ratio(logp, old_logp, mask).clamp_max(30).exp()
        gate = 4 * torch.sigmoid(tau * (ratio - 1)) / tau
        return -masked_mean(gate * advantage, mask, dim=-1).mean()


class DCPO:
    def advantage(self, rewards, cumulative_mean, cumulative_std, visit):
        rewards = rewards.detach()
        current = rewards - rewards.mean(dim=-1, keepdim=True)
        current = current / rewards.std(dim=-1, keepdim=True, correction=0).clamp_min(1e-6)
        pooled = (rewards - cumulative_mean) / cumulative_std.clamp_min(1e-6)
        current_weight = 1 / visit
        smooth_current = (1 - current_weight) * current + current_weight * pooled
        smooth_pooled = current_weight * current + (1 - current_weight) * pooled
        return torch.where(smooth_current.abs() < smooth_pooled.abs(), smooth_current, smooth_pooled)

    def loss(self, logp, old_logp, advantage, mask, eps_low=0.16, eps_high=0.20, cap=10.0):
        q = old_logp.detach().masked_fill(~mask, 0).exp()
        q = q.clamp_min(torch.finfo(q.dtype).tiny)
        lower = 0.5 + 0.5 * (1 - 4 * eps_low / q).clamp_min(0).sqrt()
        upper = (0.5 + 0.5 * (1 + 4 * eps_high / q).sqrt()).clamp_max(cap)
        log_w = log_ratio(logp, old_logp, mask).clamp_max(math.log(cap))
        objective = clipped_objective(log_w, advantage.detach().unsqueeze(-1), lower, upper)
        return -masked_mean(objective, mask, dim=-1).sum(dim=-1).mean()


class VESPO:
    def loss(self, logp, behavior_logp, rewards, mask):
        advantage = (rewards - rewards.mean(dim=-1, keepdim=True)).detach()
        c1 = torch.where(advantage > 0, 2.0, 3.0)
        c2 = torch.where(advantage > 0, 3.0, 2.0)
        log_w = log_ratio(logp, behavior_logp, mask).sum(dim=-1).detach()
        weight = (c1 * log_w + c2 * (1 - log_w.exp())).exp()
        sequence_logp = logp.masked_fill(~mask, 0).sum(dim=-1)
        return -(weight * advantage * sequence_logp).mean()


class DPO:
    def loss(self, chosen_logp, rejected_logp, ref_chosen, ref_rejected, beta=0.1):
        difference = chosen_logp - rejected_logp
        ref_difference = (ref_chosen - ref_rejected).detach()
        return -F.logsigmoid(beta * (difference - ref_difference)).mean()
