from typing import Callable, Iterable, Tuple
import math

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary
                state = self.state[p]

                # Access hyperparameters from the `group` dictionary
                alpha = group["lr"]         # twitch-able
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]

                # Complete the implementation of AdamW here, reading and saving
                # your state in the `state` dictionary above.
                # The hyperparameters can be read from the `group` dictionary
                # (they are lr, betas, eps, weight_decay, as saved in the constructor).
                #
                # 1- Update first and second moments of the gradients
                # 2- Apply bias correction
                #    (using the "efficient version" given in https://arxiv.org/abs/1412.6980;
                #     also given in the pseudo-code in the project description).
                # 3- Update parameters (p.data).
                # 4- After that main gradient-based update, update again using weight decay
                #    (incorporating the learning rate again).

                ### TODO
                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["first_moment_vec"] = torch.zeros_like(p, memory_format = torch.preserve_format)
                    state["second_moment_vec"] = torch.zeros_like(p, memory_format = torch.preserve_format)

                first_moment_vec = state["first_moment_vec"]
                second_moment_vec = state["second_moment_vec"]
                state["step"] += 1
                step = state["step"]

                # Update first and second moments
                first_moment_vec.mul_(beta1).add_(grad, alpha=1-beta1)
                second_moment_vec.mul_(beta2).addcmul_(grad, grad, value=1-beta2)

                # Efficient version: instead of computing bias-correct moments, update alpha (step size) and theta (parameter p) directly
                alpha_t = alpha * (math.sqrt(1 - (beta2 ** step)) / (1 - (beta1 ** step)))
                denom = second_moment_vec.sqrt().add_(eps)
                p.data.addcdiv_(first_moment_vec, denom, value=-alpha_t)

                # Update p again using weight decay
                p.data.mul_(1 - alpha * weight_decay)

        return loss