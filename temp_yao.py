import torch 
import math 

# Adam-like optimizer are pretty simple
# - Missing :: Randomized Parameter Groups
# - Included:: Adaptive low-rank optimization & Geometric normalization

@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    Specifically it views the shorter dimension as 'column' and ensure orthogonality of column vector space, before transposing it back. 
    
    We opt to use a quintic iteration whose coefficients are selected to maximize the slope at zero. 
    For the purpose of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X


@torch.compile
def svd_lowrank(g, q, niter=2): 
    param_dtype = g.dtype
    compute_dtype = torch.float32 
    U, S, V = torch.svd_lowrank(g.to(compute_dtype), q=q, niter=niter)
    if param_dtype != compute_dtype: 
        return U.to(param_dtype), S.to(param_dtype), V.to(param_dtype)
    return U, S, V 


def calculate_rank(loss, max_loss=None, max_rank: int = 8, min_rank: int = 1):
    # update max loss and calculate rank
    if not max_loss: 
        max_loss = loss.item() 
    ratio = loss.item() / max_loss
    new_rank = max(min_rank, int(ratio * max_rank))
    return new_rank


def to_shape(t): 
    l = list(t.shape)
    return ', '.join(map(str, l))
    

class YAO(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        wd=0.1,
        momentum=0.95,
        nesterov=True,
        ns_steps=3,
        adamw_betas=(0.95, 0.95),
        adamw_eps=1e-8,
        local_steps=10,  # Global step runs every `local_steps` iterations
    ):
        defaults = dict(lr=lr, wd=wd, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, adamw_betas=adamw_betas, adamw_eps=adamw_eps)
        super().__init__(params, defaults)
        
        # Initialize max_loss and step counter
        self.max_loss = None
        self.local_steps = local_steps
        self.global_step_counter = 0  # Tracks steps since last global update

        # Mark which params use low-rank updates
        for p in self.param_groups[0]["params"]:
            self.state[p]["use_arg"] = (p.ndim == 2)  # Only for 2D params

    def step(self, current_loss=None, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # --- Global Step: Adjust Rank & Project Momentum ---
        if self.global_step_counter % self.local_steps == 0:
            print(":: Global Step ::")
            self._global_step(current_loss)

        # --- Local Step: Normal Optimization (Fixed Rank) ---
        self._local_step()

        self.global_step_counter += 1
        return loss

    def _global_step(self, current_loss):
        """Adjust rank and project momentum buffers."""
        if current_loss is None:
            return

        # Update max_loss and compute new rank
        if self.max_loss is None:
            self.max_loss = current_loss
        else:
            self.max_loss = max(self.max_loss, current_loss)

        for group in self.param_groups:
            for p in group["params"]:
                if not self.state[p]["use_arg"]:
                    continue  # Skip non-low-rank params

                state = self.state[p]
                if "moment1_u" not in state:
                    continue  # Not initialized yet

                # Adaptive rank for each parameter
                new_rank = calculate_rank(current_loss, self.max_loss, max_rank=min(p.shape))
                
                # Get current rank and buffers
                current_rank = state["moment1_u"].shape[1]
                if new_rank == current_rank:
                    continue  # No change needed

                # Project momentum buffers to new rank
                state["moment1_u"] = self._adjust_rank(state["moment1_u"], new_rank)
                state["moment1_v"] = self._adjust_rank(state["moment1_v"], new_rank)
                state["moment1_s"] = self._adjust_rank(state["moment1_s"], new_rank)
                state["moment2_s"] = self._adjust_rank(state["moment2_s"], new_rank)

    def _local_step(self):
        """Standard optimization step with fixed rank."""
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["wd"]

            # --- Low-Rank Params ---
            lowrank_params = [p for p in group["params"] if self.state[p]["use_arg"]]
            for p in lowrank_params:
                g = p.grad
                if g is None:
                    continue

                state = self.state[p]
                if "step" not in state:
                    # Initialize on first step
                    rank = min(g.shape)  # Default rank (adjust if needed)
                    state["step"] = 0
                    state["moment1_u"] = torch.zeros(g.shape[0], rank)
                    state["moment1_v"] = torch.zeros(g.shape[1], rank)
                    state["moment1_s"] = torch.zeros(rank)
                    state["moment2_s"] = torch.zeros(rank)

                # Low-rank SVD approximation | this guy does not seem to support bfloat16 input type
                U, S, V = svd_lowrank(g, q=state["moment1_u"].shape[1], niter=2)
                
                # Update momentum buffers
                state["step"] += 1

                state["moment1_u"].lerp_(U, 1 - beta1)
                state["moment1_v"].lerp_(V, 1 - beta1)
                state["moment1_s"].lerp_(S, 1 - beta1)
                state["moment2_s"].lerp_(S.norm()**2, 1 - beta2)

                # Newton-Schulz orthogonalization
                U = zeropower_via_newtonschulz5(state["moment1_u"], group["ns_steps"])
                V = zeropower_via_newtonschulz5(state["moment1_v"], group["ns_steps"])
                _mid = (state["moment1_s"] / (eps + state["moment2_s"].sqrt())).unsqueeze(-1).to(torch.bfloat16)
                g = U @ (_mid * V.T)
                
                # Weight decay and update
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                scale = bias_correction1 / bias_correction2 ** 0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)

            # --- Full-Rank Params ---
            fullrank_params = [p for p in group["params"] if not self.state[p]["use_arg"]]
            for p in fullrank_params:
                g = p.grad
                if g is None:
                    continue

                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)

                # Standard AdamW-like update
                state["step"] += 1
                state["moment1"].lerp_(g, 1 - beta1)
                state["moment2"].lerp_(g.norm()**2, 1 - beta2)
                g = state["moment1"] / (eps + state["moment2"].sqrt())

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                scale = bias_correction1 / bias_correction2 ** 0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)


    def _adjust_rank(self, tensor, new_rank):
        """Pad or truncate a tensor along its last dimension to match new_rank."""
        if tensor.dim() == 1:  # For singular values (1D)
            if new_rank > tensor.shape[0]:
                return torch.cat([tensor, torch.zeros(new_rank - tensor.shape[0])])
            else:
                return tensor[:new_rank]
        else:  # For U/V matrices (2D)
            if new_rank > tensor.shape[1]:
                return torch.cat([tensor, torch.zeros(*tensor.shape[:-1], new_rank - tensor.shape[1])], dim=-1)
            else:
                return tensor[:, :new_rank]