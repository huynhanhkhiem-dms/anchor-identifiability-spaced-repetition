"""Analytic quantities used by the field-gate and full-profile reproduction."""
from __future__ import annotations
import numpy as np
from .memory import power_g, RHO0_FSRS

def dg_dpsi(x, psi, rho0=RHO0_FSRS, h=1e-7):
    return (power_g(x, psi+h, rho0)-power_g(x, psi-h, rho0))/(2.0*h)

def fisher_curvature_constant(psi, rho0=RHO0_FSRS):
    psi=np.asarray(psi,dtype=float)
    b=np.log(rho0)/psi
    a=np.exp(b)
    A=a-1.0-a*b
    return rho0*A**2/(1.0-rho0)

def dg_dx(x, psi, rho0=RHO0_FSRS):
    x=np.asarray(x,dtype=float)
    f=np.power(rho0,-1.0/psi)-1.0
    return -psi*f*np.power(1.0+f*x,-psi-1.0)

def observed_joint_information(u, psi, rho0=RHO0_FSRS, eps=1e-12):
    x=np.exp(np.asarray(u,dtype=float))
    p=np.clip(power_g(x,psi,rho0),eps,1.0-eps)
    dp_psi=dg_dpsi(x,psi,rho0)
    dp_alpha=-x*dg_dx(x,psi,rho0)
    wgt=1.0/(p*(1.0-p))
    return np.array([
        [float(np.sum(dp_psi*dp_psi*wgt)),float(np.sum(dp_psi*dp_alpha*wgt))],
        [float(np.sum(dp_psi*dp_alpha*wgt)),float(np.sum(dp_alpha*dp_alpha*wgt))],
    ])

def observed_efficient_information(u, psi, rho0=RHO0_FSRS):
    I=observed_joint_information(u,psi,rho0)
    if I[1,1]<=0: return 0.0
    return float(max(I[0,0]-I[0,1]**2/I[1,1],0.0))

def observed_se(u, psi, rho0=RHO0_FSRS):
    info=observed_efficient_information(u,psi,rho0)
    return float("inf") if info<=0 else float(1.0/np.sqrt(info))

def prior_precision(sigma_param, gamma=1.0):
    return float(2.0*gamma/float(sigma_param)**2)

def data_precision(n, sigma_u, psi, rho0=RHO0_FSRS):
    return float(n)*fisher_curvature_constant(psi,rho0)*float(sigma_u)**2

def prior_share(n, sigma_u, psi, sigma_param, gamma=1.0, rho0=RHO0_FSRS):
    ip=prior_precision(sigma_param,gamma)
    idata=data_precision(n,sigma_u,psi,rho0)
    return float(ip/(ip+idata))

def prior_share_from_curvature(data_curvature, sigma_param, gamma=1.0):
    ip=prior_precision(sigma_param,gamma)
    return float(ip/(ip+max(float(data_curvature),0.0)))

def finite_difference_probability_jacobian(prob_fn, x, bounds=None, rel_step=1e-5):
    x=np.asarray(x,dtype=float)
    p0=np.asarray(prob_fn(x),dtype=float).ravel()
    npar=x.size
    J=np.empty((p0.size,npar),dtype=float)
    b=None if bounds is None else np.asarray(bounds,dtype=float)
    for j in range(npar):
        span=1.0 if b is None else max(float(b[j,1]-b[j,0]),1e-8)
        h=rel_step*max(1.0,abs(float(x[j])),span)
        lo=-np.inf if b is None else float(b[j,0])
        hi=np.inf if b is None else float(b[j,1])
        if x[j]-h>=lo and x[j]+h<=hi:
            xp=x.copy(); xp[j]+=h
            xm=x.copy(); xm[j]-=h
            J[:,j]=(np.asarray(prob_fn(xp)).ravel()-np.asarray(prob_fn(xm)).ravel())/(2.0*h)
        elif x[j]+h<=hi:
            xp=x.copy(); xp[j]+=h
            J[:,j]=(np.asarray(prob_fn(xp)).ravel()-p0)/h
        elif x[j]-h>=lo:
            xm=x.copy(); xm[j]-=h
            J[:,j]=(p0-np.asarray(prob_fn(xm)).ravel())/h
        else:
            J[:,j]=0.0
    return p0,J

def profile_fisher_from_jacobian(p, J, idx=20, eps=1e-9, rcond=1e-10):
    p=np.clip(np.asarray(p,dtype=float).ravel(),eps,1.0-eps)
    J=np.asarray(J,dtype=float)
    if J.shape[0]!=p.size:
        raise ValueError("probability vector and Jacobian have incompatible sizes")
    keep=[j for j in range(J.shape[1]) if j!=idx]
    sw=1.0/np.sqrt(p*(1.0-p))
    z=J[:,idx]*sw
    Z=J[:,keep]*sw[:,None]
    if Z.shape[1]==0: return float(z@z),0
    coef,_,rank,_=np.linalg.lstsq(Z,z,rcond=rcond)
    resid=z-Z@coef
    info=float(resid@resid)
    scale=max(float(z@z),1.0)
    if info<0 and abs(info)<=1e-10*scale: info=0.0
    return max(info,0.0),int(rank)
