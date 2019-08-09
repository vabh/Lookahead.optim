import torch
import torch.optim as optim

class LookAhead(optim.Optimizer):
    def __init__(self, params, alpha, k, base_optim, reset_momentum=False):
        self.alpha = alpha
        self.A = base_optim
        self.k = k
        self.reset_momentum = reset_momentum
        self.inner_step = 0

        self._sync_params(params)

        defaults = {}
        super(LookAhead, self).__init__(self.phi, defaults)

    def _sync_params(self, params):
        with torch.no_grad():
            self.theta = list(params)
            self.phi = []
            for p in self.theta:
                self.phi.append(p.clone())

    def step(self, closure=None):
        if self.inner_step == self.k:
            self.inner_step = 0

            for pg1, pg2 in zip(self.param_groups, self.A.param_groups):
                for p1, p2 in zip(pg1['params'], pg2['params']):
                    p1.data = p1.data + self.alpha * (p2.data - p1.data)
                    p2.data = p1.data.clone()
                    if self.reset_momentum:
                        if 'momentum' in pg2['momentum'] and pg2['momentum'] != 0:
                            if 'momentum_buffer' in self.A.state[p2]:
                                self.A.state[p2].pop('momentum_buffer')
        else:
            self.inner_step += 1
            self.A.step()

    def zero_grad(self):
        self.A.zero_grad()
