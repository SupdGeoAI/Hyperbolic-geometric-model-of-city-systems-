import torch
import pickle
import torch as th
import numpy as np
from abc import abstractmethod
from torch.nn import Embedding
import torch.nn.functional as F
from torch.autograd import Function
from torch.optim.optimizer import Optimizer, required

class RiemannianSGD(Optimizer):
    def __init__(self,params,lr=required,rgrad=required,expm=required):
        defaults = {'lr':lr,'rgrad':rgrad,'expm':expm}
        super(RiemannianSGD,self).__init__(params,defaults)
    
    def step(self,lr=None,counts=None,**kwargs):
        loss = None
        for group in self.param_groups:
            for p in group['params']:
                lr = lr or group['lr']
                rgrad = group['rgrad']
                expm = group['expm']

                if p.grad is None:
                    continue
                d_p = p.grad.data
                if d_p.is_sparse:
                    d_p = d_p.coalesce()
                d_p = rgrad(p.data,d_p)
                d_p.mul_(-lr)
                expm(p.data,d_p)
        return loss

class Manifold(object):
    def allocate_lt(self,N,dim,sparse):
        return Embedding(N,dim,sparse=sparse)

    def normalize(self,u):
        return u
    
    @abstractmethod
    def distance(self, u, v):
        return NotImplementedError
    
    def init_weights(self,w,scale=1e-4):
        w.weight.data.uniform_(-scale,scale)
    
    @abstractmethod
    def expm(self, p, d_p, lr=None, out=None):
        raise NotImplementedError
    
    @abstractmethod
    def logm(self, x, y):
        raise NotImplementedError
    
    @abstractmethod
    def ptransp(self, x, y, v, ix=None, out=None):
        raise NotImplementedError
    
    def norm(self, u, **kwargs):
        if isinstance(u, Embedding):
            u = u.weight
        return u.pow(2).sum(dim=-1).sqrt()
    
class EuclideanManifold(Manifold):
    __slots__ = ["max_norm"]
    def __init__(self,max_norm = None, K = None ,**kwargs):
        self.max_norm = max_norm
        self.K = K
        if K is not None:
            self.inner_radius = 2 * self.K / (1 + np.sqrt(1+4*self.K*self.K))
        
    def normalize(self, u):
        d = u.size(-1)
        if self.max_norm:
            u.view(-1,d).renorm_(2,0,self.max_norm)
        return u
    
    def distance(self,u,v):
        return (u - v).pow(2).sum(dim=-1)
    
    def rgrad(self, p, d_p):
        return d_p
    
    def expm(self, p, d_p, normalize=False, lr=None, out=None):
        if lr is not None:
            d_p.mul_(-lr)
        if out is None:
            out = p
        out.add_(d_p)
        if normalize:
            self.normalize(out)
        return out
    
    def logm(self, p, d_p, out=None):
        return p - d_p
    
    def ptransp(self, p, x, y, v):
        ix, v_ = v._indices().squeeze(), v._values()
        return p.index_copy_(0, ix, v_)

class PoincareManifold(EuclideanManifold):
    def __init__(self,eps=1e-5,K=None,**kwargs):
        self.eps = eps
        super(PoincareManifold,self).__init__(max_norm=1-eps)
        self.K = K
        if K is not None:
            self.inner_radius = 2 * K / (1 + np.sqrt(1 + 4 * K * self.K))
    
    def distance(self,u,v):
        return Distance.apply(u,v,self.eps)
    
    def regrad(self,p,d_p):
        if d_p.is_sparse:
            p_sqnorm = th.sum(
                p[d_p._indices()[0].squeeze()] ** 2, dim=1,
                keepdim=True
            ).expand_as(d_p._values())
            n_vals = d_p._values() * ((1 - p_sqnorm) ** 2) / 4
            n_vals.renorm_(2, 0, 5)
            d_p = th.sparse.DoubleTensor(d_p._indices(), n_vals, d_p.size())
        else:
            p_sqnorm = th.sum(p ** 2, dim=-1, keepdim=True)
            d_p = d_p * ((1 - p_sqnorm) ** 2 / 4).expand_as(d_p)
        return d_p

class Distance(Function):
    @staticmethod
    def forward(ctx, u, v, eps):
        squnorm = th.clamp(th.sum(u * u, dim=-1), 0, 1 - eps)
        sqvnorm = th.clamp(th.sum(v * v, dim=-1), 0, 1 - eps)
        sqdist = th.sum(th.pow(u - v, 2), dim=-1)
        ctx.eps = eps
        ctx.save_for_backward(u, v, squnorm, sqvnorm, sqdist)
        x = sqdist / ((1 - squnorm) * (1 - sqvnorm)) * 2 + 1
        z = th.sqrt(th.pow(x, 2) - 1)
        return th.log(x + z)

    @staticmethod
    def backward(ctx, g):
        u, v, squnorm, sqvnorm, sqdist = ctx.saved_tensors
        g = g.unsqueeze(-1)
        gu = Distance.grad(u, v, squnorm, sqvnorm, sqdist, ctx.eps)
        gv = Distance.grad(v, u, sqvnorm, squnorm, sqdist, ctx.eps)
        return g.expand_as(gu) * gu, g.expand_as(gv) * gv, None
    
    @staticmethod
    def grad(x, v, sqnormx, sqnormv, sqdist, eps):
        alpha = (1 - sqnormx)
        beta = (1 - sqnormv)
        z = 1 + 2 * sqdist / (alpha * beta)
        a = ((sqnormv - 2 * th.sum(x * v, dim=-1) + 1) / th.pow(alpha, 2))\
            .unsqueeze(-1).expand_as(x)
        a = a * x - v / alpha.unsqueeze(-1).expand_as(v)
        z = th.sqrt(th.pow(z, 2) - 1)
        z = th.clamp(z * beta, min=eps).unsqueeze(-1)
        return 4 * a / z.expand_as(x)

class EnergyFunction(torch.nn.Module):
    def __init__(self,manifold,dim,size,sparse=False,**kwargs):
        super().__init__()
        self.manifold = manifold
        self.lt = manifold.allocate_lt(size,dim,sparse)
        self.nobjects = size
        self.manifold.init_weights(self.lt)
    
    def forward(self,inputs):
        e = self.lt(inputs)
        with torch.no_grad():
            e = self.manifold.normalize(e)
        o = e.narrow(1,1,e.size(1)-1)
        s = e.narrow(1,0,1).expand_as(o)
        return self.energy(s,o).squeeze(-1)
    
    def optim_params(self):
        return [{
            'params': self.lt.parameters(),
            'rgrad': self.manifold.rgrad,
            'expm': self.manifold.expm,
            'logm': self.manifold.logm,
            'ptransp': self.manifold.ptransp,
        }]
    
    def loss_function(self, inp, target, **kwargs):
        raise NotImplementedError

class DistanceEnergyFunction(EnergyFunction):
    def energy(self,s,o):
        return self.manifold.distance(s,o)
    
    def loss(self,inp,target,**kwargs):
        return F.cross_entropy(inp.neg(),target)

    def save_embedding(self,epoch,id2word,version):
        embedding = self.lt.weight.cpu().data.numpy()
        emb_dict = {}
        for wid, w in id2word.items():
            emb_dict[w] = embedding[wid]
        save_path = '../log/v_'+str(version)+'/embedding_epoch_'+str(epoch+1)
        with open(save_path,'wb') as f:
            pickle.dump(emb_dict,f)