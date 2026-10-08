"""Anchored scale-family memory models and desired-retention schedulers.

Pure NumPy re-implementation of the retention functions and state-update rules
of the spaced-repetition algorithms compared in the public SRS benchmark
(open-spaced-repetition/srs-benchmark).  No PyTorch dependency.

The central object of the paper is the *anchored scale family*

    R(t | S; psi) = g(t / S; psi),        g(1; psi) = rho0  for all psi,

i.e. a retention function that depends on elapsed time only through the ratio
x = t / S, and whose graph is pinned to the anchor point (x, R) = (1, rho0)
regardless of the shape parameter psi.
"""

from __future__ import annotations
import numpy as np

__all__ = [
    "fsrs_forgetting_curve", "fsrs_inverse_curve", "interval_multiplier",
    "power_g", "exp_g", "FAMILIES", "FSRS6", "S_MIN", "S_MAX",
]

S_MIN = 0.001
S_MAX = 36500.0
RHO0_FSRS = 0.9
RHO0_HLR = 0.5

FSRS6_INIT_W = np.array([
    0.212, 1.2931, 2.3065, 8.2956, 6.4133, 0.8334, 3.0194, 0.001,
    1.8722, 0.1666, 0.796, 1.4835, 0.0614, 0.2629, 1.6483, 0.6014,
    1.8729, 0.5425, 0.0912, 0.0658, 0.1542,
], dtype=float)

FSRS6_BOUNDS = np.array([
    (S_MIN,100.0),(S_MIN,100.0),(S_MIN,100.0),(S_MIN,100.0),
    (1.0,10.0),(0.001,4.0),(0.001,4.0),(0.001,0.75),
    (0.0,4.5),(0.0,0.8),(0.001,3.5),(0.001,5.0),
    (0.001,0.25),(0.001,0.9),(0.0,4.0),(0.0,1.0),
    (1.0,6.0),(0.0,2.0),(0.0,2.0),(0.0,0.8),(0.1,0.8),
], dtype=float)

def power_g(x, psi, rho0=RHO0_FSRS):
    x=np.asarray(x,dtype=float); psi=np.asarray(psi,dtype=float)
    f=np.power(rho0,-1.0/psi)-1.0
    return np.power(1.0+f*x,-psi)

def power_g_inv(rho, psi, rho0=RHO0_FSRS):
    rho=np.asarray(rho,dtype=float); psi=np.asarray(psi,dtype=float)
    f=np.power(rho0,-1.0/psi)-1.0
    return (np.power(rho,-1.0/psi)-1.0)/f

def exp_g(x, rho0=RHO0_FSRS):
    return np.power(rho0,np.asarray(x,dtype=float))

def exp_g_inv(rho, rho0=RHO0_FSRS):
    return np.log(np.asarray(rho,dtype=float))/np.log(rho0)

def fsrs_forgetting_curve(t,s,psi):
    return power_g(np.asarray(t,dtype=float)/np.asarray(s,dtype=float),psi)

def fsrs_inverse_curve(rho,s,psi):
    return np.asarray(s,dtype=float)*power_g_inv(rho,psi)

def interval_multiplier(theta,psi,family="power"):
    if family=="power": return power_g_inv(theta,psi)
    if family=="exp": return exp_g_inv(theta,RHO0_FSRS)
    if family=="exp_hlr": return exp_g_inv(theta,RHO0_HLR)
    raise ValueError(f"unknown family {family!r}")

FAMILIES={
    "FSRS-6":dict(g=power_g,rho0=RHO0_FSRS,shape_free=True,shape="psi = w[20]"),
    "FSRS-5":dict(g=lambda x,psi=0.5:power_g(x,0.5),rho0=RHO0_FSRS,shape_free=False,shape="psi = 0.5 (fixed)"),
    "FSRS-4.5":dict(g=lambda x,psi=0.5:power_g(x,0.5),rho0=RHO0_FSRS,shape_free=False,shape="psi = 0.5 (fixed)"),
    "HLR":dict(g=lambda x,psi=None:exp_g(x,RHO0_HLR),rho0=RHO0_HLR,shape_free=False,shape="exponential (fixed)"),
    "SM2-trainable":dict(g=lambda x,psi=None:exp_g(x,RHO0_FSRS),rho0=RHO0_FSRS,shape_free=False,shape="exponential (fixed)"),
    "Anki":dict(g=lambda x,psi=None:exp_g(x,RHO0_FSRS),rho0=RHO0_FSRS,shape_free=False,shape="exponential (fixed)"),
}

class FSRS6:
    n_params=21
    def __init__(self,w=None):
        self.w=np.asarray(FSRS6_INIT_W if w is None else w,dtype=float).copy()
        if self.w.shape!=(self.n_params,): raise ValueError(f"expected 21 parameters, got {self.w.shape}")
    @property
    def psi(self): return self.w[20]
    def retention(self,t,s): return fsrs_forgetting_curve(t,s,self.w[20])
    def schedule(self,s,theta): return fsrs_inverse_curve(theta,s,self.w[20])
    def init_state(self,rating):
        rating=np.asarray(rating,dtype=int)
        s=self.w[np.clip(rating,1,4)-1]
        d=np.clip(self.w[4]-np.exp(self.w[5]*(rating-1))+1.0,1.0,10.0)
        return np.clip(s,S_MIN,S_MAX),d
    def _next_d(self,d,rating):
        delta_d=-self.w[6]*(rating-3.0)
        new_d=d+delta_d*(10.0-d)/9.0
        init_d4=self.w[4]-np.exp(self.w[5]*3.0)+1.0
        new_d=self.w[7]*init_d4+(1.0-self.w[7])*new_d
        return np.clip(new_d,1.0,10.0)
    def _s_success(self,s,d,r,rating):
        hard=np.where(rating==2,self.w[15],1.0)
        easy=np.where(rating==4,self.w[16],1.0)
        sinc=1.0+(np.exp(self.w[8])*(11.0-d)*np.power(s,-self.w[9])*(np.exp((1.0-r)*self.w[10])-1.0)*hard*easy)
        return s*sinc
    def _s_failure(self,s,d,r):
        new_s=self.w[11]*np.power(d,-self.w[12])*(np.power(s+1.0,self.w[13])-1.0)*np.exp((1.0-r)*self.w[14])
        return np.minimum(new_s,s/np.exp(self.w[17]*self.w[18]))
    def step(self,s,d,delta_t,rating):
        s=np.asarray(s,dtype=float); d=np.asarray(d,dtype=float)
        delta_t=np.asarray(delta_t,dtype=float); rating=np.asarray(rating,dtype=float)
        r=self.retention(delta_t,s); short_term=delta_t<1.0; success=rating>1
        s_short=s*np.exp(self.w[17]*(rating-3.0+self.w[18]))*np.power(s,-self.w[19])
        s_short=np.where(rating>=2,np.maximum(s_short/s,1.0)*s,s_short)
        new_s=np.where(short_term,s_short,np.where(success,self._s_success(s,d,r,rating),self._s_failure(s,d,r)))
        return np.clip(new_s,S_MIN,S_MAX),self._next_d(d,rating)

FSRS6_DEFAULT_STDDEV=np.array([
    6.43,9.66,17.58,27.85,0.57,0.28,0.6,0.12,0.39,0.18,0.33,
    0.3,0.09,0.16,0.57,0.25,1.03,0.31,0.32,0.14,0.27,
],dtype=float)
FSRS6_PENALTY_GAMMA=1.0
FSRS6_OPTIM=dict(lr=4e-2,betas=(0.9,0.999),n_epoch=5,batch_size=512)

def fsrs6_penalty(w,gamma=FSRS6_PENALTY_GAMMA,init_w=None,stddev=None):
    w=np.asarray(w,dtype=float)
    init_w=FSRS6_INIT_W if init_w is None else np.asarray(init_w,dtype=float)
    stddev=FSRS6_DEFAULT_STDDEV if stddev is None else np.asarray(stddev,dtype=float)
    return float(gamma*np.sum((w-init_w)**2/stddev**2))

def clip_to_box(w):
    return np.clip(np.asarray(w,dtype=float).copy(),FSRS6_BOUNDS[:,0],FSRS6_BOUNDS[:,1])
