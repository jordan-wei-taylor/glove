import numpy as np

def get_optimiser(name):
    valid = dict(adam = Adam, adagrad = AdaGrad, adamax = AdaMax, sgd = Sgd)
    name  = name.lower()
    if name in valid:
        return valid[name]
    raise Exception(f"Invalid optimiser '{name}'")

class Optimiser():
    
    def __init__(self, kwargs, exceptions = ['__class__', 'self']):
        self.max_norm      = None
        self.gradient_clip = None
        self.decay         = None
        self.t             = 1
        for k, v in kwargs.items():
            if k in exceptions: continue
            if k == 'kwargs':
                for k, val in v.items():
                    setattr(self, k, val)
            else:
                setattr(self, k, v)
        
    def _clip(self, gradient):
        if self.gradient_clip:
            return np.clip(gradient, -self.gradient_clip, self.gradient_clip)
        return gradient
    
    def _norm(self, gradient):
        if self.max_norm:
            norm = np.linalg.norm(gradient)
            if norm > self.max_norm:
                ratio = self.max_norm / norm
                return ratio * gradient
        return gradient
    
    @property
    def _eta(self):
        if self.decay:
            return self.eta / (1 + self.decay * self.t)
        return self.eta
        
    def call(self, gradient):
        self.t   += 1
        gradient *= self._eta
        for func in [self._norm, self._clip]:
            gradient = func(gradient)
        return gradient
    
class Adam(Optimiser):
    
    def __init__(self, eta = 1e-3, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, **kwargs):     
        super().__init__(locals())
        self.m     = 0
        self.v     = 0
        
    def __call__(self, gradient):
        self.m  = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v  = self.beta2 * self.v + (1 - self.beta2) * gradient ** 2
        m_hat   = self.m / (1 - self.beta1 ** self.t)
        v_hat   = self.v / (1 - self.beta2 ** self.t)
        ret     = m_hat / (np.sqrt(v_hat) + self.eps)
        return super().call(ret)
    
class AdaGrad(Optimiser):
    
    def __init__(self, eta = 1e-3, eps = 1e-8, **kwargs):
        super().__init__(locals())
        self.gradsq = 0
        
    def __call__(self, gradient):
        self.gradsq += np.square(gradient)
        ret          = gradient / np.sqrt(self.gradsq + self.eps)
        return super().call(ret)
    
class AdaMax(Optimiser):
    
    def __init__(self, eta = 1e-3, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, **kwargs):
        super().__init__(locals())
        self.m     = 0
        self.v     = 0
        
    def __call__(self, gradient):
        self.m  = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v  = np.maximum(self.beta2 * self.v, np.fabs(gradient))
        m_hat   = self.m / (1 - self.beta1 ** self.t)
        ret     = m_hat / np.sqrt(self.v + self.eps)
        return super().call(ret)
    
class Sgd(Optimiser):
    
    def __init__(self, eta = 1e-3, **kwargs):
        super().__init__(locals())
        
    def __call__(self, gradient):
        return super().call(gradient)
    
