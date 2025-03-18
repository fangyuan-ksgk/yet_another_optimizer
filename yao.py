import torch 
import math 

# Adam-like optimizer are pretty simple

# Traditional Optimizer assumes 'gradient' and 'parameter' are fixed
# - Our idea combines 'wrapping low-rank adaptor' and call .backward() with optimization gadegts together 
# - in terms of code this is not just a custom optimizer, there needs to be another functionality happening before the .backward() functional .... 

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


def calculate_rank(loss, max_loss=None, max_rank: int = 8, min_rank: int = 1):
    # update max loss and calculate rank
    if not max_loss: 
        max_loss = loss.item() 
    ratio = loss.item() / max_loss
    return max(min_rank, int(ratio * max_rank)), max_loss


class ARG(torch.optim.Optimizer): 
    
    # Adaptive Rank Geometric Optimizer (ver.6)
    # - combines DoRA & Muon with SVD
    # - fixed grouping
    # - can't ensure inverse proportional magnitude & direction change
        
    def __init__(
        self,
        lr=1e-3,
        wd=0.1,
        argm_params=None,
        momentum=0.95,
        nesterov=True,
        ns_steps=3,
        adamw_betas=(0.95, 0.95),
        adamw_eps=1e-8,
    ):

        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        params = list(argm_params)
        super().__init__(params, defaults)
        
        for p in argm_params:
            # Use Muon for every parameter in muon_params which is >= 2D and doesn't look like an embedding or head layer
            assert p.ndim == 2, p.ndim
            if p.ndim == 2: 
                self.state[p]["use_arg"] = True
            else: 
                self.state[p]["use_arg"] = False 
                
        # initialize max_loss
        self.max_loss = None
        self.ns_steps = ns_steps
        
    def step(self, current_loss, closure=None): 
        
        loss = None 
        if closure is not None: 
            with torch.enable_grad(): 
                loss = closure() 
                
        # calculate rank and update max loss
        rank, self.max_loss = calculate_rank(current_loss, self.max_loss)
            
        for group in self.param_groups: 
            
            ##############################
            # AdaRank Geometric Momentum #
            ##############################
            
            params = [p for p in group["params"] if self.state[p]["use_arg"]]
            lr = group["lr"]
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["wd"]
            
            for p in params: 
                g = p.grad 
                if g is None: # gradient is supposed to be calculated already
                    continue 
                state = self.state[p]
                if "step" not in state: 
                    state["step"] = 0 
                    state["moment1_u"] = torch.zeros(g.shape[0], rank)
                    state["moment1_v"] = torch.zeros(g.shape[1], rank)
                    state["moment1_s"] = torch.zeros(rank)
                    state["moment2_s"] = torch.zeros(rank)
                
                U, S, V = torch.svd_lowrank(g, max_loss=None, q=rank, niter=2)
                
                state["step"] += 1 
                step = state["step"]
                
                # first moment (U, S, V)
                buf1 = state["moment1_u"]
                buf2 = state["moment1_v"]
                buf3 = state["moment1_s"]
                buf1.lerp_(U, 1 - beta1)
                buf2.lerp_(V, 1 - beta1)
                buf3.lerp_(S, 1 - beta1)
                
                # second moment (S) | Geometric Momentum
                buf4 = state["moment2_s"]
                buf4.lerp_(S.norm()**2, 1 - beta2)
                
                # NS orthogonalization to regularize U, V
                U = zeropower_via_newtonschulz5(buf1, self.ns_steps)
                V = zeropower_via_newtonschulz5(buf2, self.ns_steps)
                g = U @ (buf3 / (eps + buf4.sqrt()) * V) # u, s, v scaled with momentum
                
                # weight decay
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale) # add scaled gradient step
                
                
            ########################
            #  Geometric Momentum  # 
            ########################
            
            params = [p for p in group["params"] if not self.state[p]["use_arg"]]
            lr = group["lr"]
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["wd"]
            
            for p in params: 
                g = p.grad 
                if g is None: 
                    continue 
                state = self.state[p]
                if "step" not in state: 
                    state["step"] = 0 
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1 
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.norm()**2, 1 - beta2)
                
                g = buf1 / (eps + buf2.sqrt())
                
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale) # add scaled gradient step