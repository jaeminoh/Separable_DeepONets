from functools import partial

from jax import random, jvp, jit, vmap
import jax.numpy as np
from optax import adam, cosine_decay_schedule
from jaxopt import OptaxSolver
from tqdm import trange
import matplotlib.pyplot as plt

from nn import MLP


"""
Physics-Informed Deep Operator Networks.
Initial condition: ɑsin(x)
PDE: u_t + β u_x = 0, with varying β
Boundary condition: periodic on x
Approximating solution operator:
(ɑ,β) -> u
"""

def to_column(lst):
    return [_lst.reshape(-1,1) for _lst in lst]

class sno:
    T = np.array([0, 1])
    X = np.array([0, 2*np.pi])
    a = np.array([1, 2])
    b = np.array([1, 5])

    def __init__(self):
        N_tr = 100
        N_te = 64
        layers_b = [1, 100,100,100, 200]
        self.init_b, self.apply_b = MLP(layers_b, np.sin, w0=8.)
        layers_t = [1, 100,100, 200]
        self.init_t, self.apply_t = MLP(layers_t, np.sin, w0=10.)
        self.domain_te = (np.linspace(*self.T, N_te),
                          np.linspace(*self.X, N_te),
                          np.linspace(*self.a, N_te),
                          np.linspace(*self.b, N_te))
        self.domain_tr = [np.linspace(*self.T, N_tr),
                          np.linspace(*self.X, N_tr),
                          np.linspace(*self.a, N_tr),
                          np.linspace(*self.b, N_tr)]
        self.u0 = vmap(vmap(self._u0,
                            (0,None), 0),
                            (None,0), 1)


    def _u0(self, x, a):
        return a*np.sin(x)

    def u(self, params, t,x, a,b):
        params_b, params_t = params
        in_t = to_column([t,x])
        out_t = [self.apply_t(params_t[i], _in) for i, _in in enumerate(in_t)]
        in_b = to_column([a,b])
        out_b = [self.apply_b(params_b[i], _in) for i, _in in enumerate(in_b)]
        return np.einsum("az,bz,cz,dz->abcd", *out_t, *out_b) #(Nt, Nx, Na, Nb)
    
    @partial(jit, static_argnums=(0, 6))
    def pde(self, params, t,x, a,b, is_sampling=True):
        _, u_t = jvp(lambda t: self.u(params, t,x, a,b), (t,), (np.ones(t.shape),))
        _, u_x = jvp(lambda x: self.u(params, t,x, a,b), (x,), (np.ones(x.shape),))
        pde = (u_t + b*u_x)**2
        if is_sampling:
            return pde.mean((1,2,3)), pde.mean((0,2,3)), pde.mean((0,1,3)), pde.mean((1,2,3))
        return pde

    def loss_ic(self, params, t,x, a,b):
        init_data = self.u0(x, a)[...,None] # (Nx, Na, 1)
        init_pred = self.u(params, np.array([0]),x, a,b)[0,...] # (Nx, Na, Nb)
        loss_ic = np.mean( (init_pred - init_data)**2 )
        return loss_ic

    def loss_bc(self, params, t,x, a,b):
        u = self.u(params, t,self.X, a,b)
        loss_bc = np.mean( (u[:,1,...] - u[:,0,...])**2 )
        return loss_bc

    @partial(jit, static_argnums=(0,))
    def loss(self, params, t,x, a,b):
        loss = (self.pde(params, t,x, a,b, is_sampling=False).mean()
                + 1e3*self.loss_ic(params, t,x, a,b)
                + self.loss_bc(params, t,x, a,b))
        return loss
    
    def train(self, optimizer, nIter, seed):
        train_key, init_key = random.split(seed)
        key_b, key_t = random.split(init_key)
        keys_b = random.split(key_b)
        params_b = [self.init_b(_key) for _key in keys_b]
        keys_t = random.split(key_t)
        params_t = [self.init_t(_key) for _key in keys_t]
        params = [params_b, params_t]

        @jit
        def step(params, state, *args, **kwargs):
            params, state = optimizer.update(params, state, *args, **kwargs)
            return params, state
        
        @jit
        def sampling(key, domain, loss, minval, maxval):
            """R3 sampling (ICML 2023)"""
            N = domain.size
            mask = (loss < loss.mean()).astype(int)
            sample = random.uniform(key, (N,), minval=minval, maxval=maxval)
            update = (1-mask)*domain + mask*sample
            return update
        
        state = optimizer.init_state(params)
        loss_log = []
        for epoch in (pbar:=trange(nIter)):
            params, state = step(params, state, *self.domain_tr)
            if epoch % 100 == 0:
                loss = self.loss(params, *self.domain_te)
                loss_log.append(loss)
                pde_loss_t, pde_loss_x, pde_loss_a, pde_loss_b = self.pde(params, *self.domain_tr)
                train_key, sample_key = random.split(train_key)
                self.domain_tr[0] = sampling(sample_key, self.domain_tr[0], pde_loss_t, *self.T)
                self.domain_tr[1] = sampling(sample_key, self.domain_tr[1], pde_loss_x, *self.X)
                self.domain_tr[2] = sampling(sample_key, self.domain_tr[2], pde_loss_a, *self.a)
                self.domain_tr[3] = sampling(sample_key, self.domain_tr[3], pde_loss_b, *self.b)
                pbar.set_postfix({"loss":loss})
        return params, loss_log

model = sno()
seed = random.PRNGKey(0)

nIter = 1*10**5
lr = cosine_decay_schedule(1e-4, nIter)
optimizer = OptaxSolver(fun=model.loss, opt=adam(lr))

opt_params, loss_log = model.train(optimizer, nIter, seed)
def truth(t,x, a,b):
    return a*np.sin(x - b*t)

truth = vmap(vmap(vmap(vmap(truth,
                            (0,None,None,None), 0),
                            (None,0,None,None), 1),
                            (None,None,0,None), 2),
                            (None,None,None,0), 3)

u = model.u(opt_params, *model.domain_te)
truth = truth(*model.domain_te)

def relative_l2(pred, true):
    return np.linalg.norm(pred-true)/np.linalg.norm(true)

l2_err = vmap(vmap(relative_l2,
                   (2,2), 0),
                   (3,3), 1)(u, truth)

fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2, figsize=(14,12))
extent = [*model.T, *model.X]
ax0.imshow(u[...,1,1].T, origin='lower', extent=extent, aspect='auto', cmap='jet')
ax0.set_title(f'{model.domain_te[2][1]:.3e}, {model.domain_te[3][1]:.3e}')
ax1.imshow(u[...,-2,-2].T, origin='lower', extent=extent, aspect='auto', cmap='jet')
ax1.set_title(f'{model.domain_te[2][-2]:.3e}, {model.domain_te[3][-2]:.3e}')
ax2.semilogy(loss_log)
ax2.set_title('pinn loss')
im = ax3.imshow(l2_err.T, origin='lower', extent=[*model.a, *model.b], aspect='auto', cmap='jet')
ax3.set_xlabel("alpha")
ax3.set_ylabel("beta")
ax3.set_title('relative l2')
fig.colorbar(im)

plt.tight_layout()
plt.savefig('figure')