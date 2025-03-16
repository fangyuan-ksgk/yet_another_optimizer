import torch 
import math 

# Adam-like optimizer are pretty simple

# Traditional Optimizer assumes 'gradient' and 'parameter' are fixed
# - Our idea combines 'wrapping low-rank adaptor' and call .backward() with optimization gadegts together 
# - in terms of code this is not just a custom optimizer, there needs to be another functionality happening before the .backward() functional .... 


class Argm(torch.optim.Optimizer): 
    
    def __init__(
        self,
        lr=1e-3,
        wd=0.1,
        argm_params=None,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
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
                self.state[p]["adjust_rank"] = True
            else: 
                self.state[p]["adjust_rank"] = False 
                
    def step(self, closure=None): 
        
        loss = None 
        if closure is not None: 
            with torch.enable_grad(): 
                loss = closure() 
            
        for group in self.param_groups: 
            
            ######################
            # Geometric Momentum #
            ######################
            
            params = [p for p in group["params"] if not self.state[p]["adjust_rank"]]
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
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1 
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                squared_norm = torch.ones_like(g) * (g.norm()**2) # Change 1. Squared norm of gradient vector
                buf2.lerp_(squared_norm, 1 - beta2)
                
                g = buf1 / (eps + buf2.sqrt())
                
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale) # add scaled gradient step
                
                
            ########################################
            #    Adaptive Rank Geometric Momentum  # 
            ########################################
            
            # Question: How do we apply lora-updates within optimizer? 
            #   - we already have the gradient right? so the lora addition needs to be out-side the optimizer? 
            #   - it should probably be wrapped around the model, before the .backward() gets called ... 
            #   - both the weight decomposition and rank adaption should be done before the .backward() functional ... 
            
            
            default_rank = 8
            
            params = [p for p in group["params"] if self.state[p]["adjust_rank"]]
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
                squared_norm = torch.ones_like(g) * (g.norm()**2)
                buf2.lerp_(squared_norm, 1 - beta2)
                
                g = buf1 / (eps + buf2.sqrt())
                
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                
            
            
            