import numpy as np
import emcee

def lnl(p,x,xerr_up,xerr_down):
    mu,s2=p[0],np.exp(2.*p[1])
    if s2<1e-3:
        return -np.inf
    delta_x=x-mu
    ss = s2+(xerr_up**2)*(delta_x>0.)+(xerr_down**2)*(delta_x<=0.)
    # print mu, s2, np.sum(-delta_x**2/2./ss-.5*np.log(ss))
    return np.sum(-delta_x**2/2./ss-.5*np.log(ss))

xx = np.genfromtxt('sawala_dsph_data.dat')
print xx
mu,ss = 18., np.log(2.)
ndim,nwalkers=2,100
p0 = np.random.multivariate_normal(mean=[mu,ss],cov=np.diag([1.,.1]),size=nwalkers)
sampler = emcee.EnsembleSampler(nwalkers,ndim,lnl,args=(xx.T[0],xx.T[1],xx.T[2]))
sampler.run_mcmc(p0,10000)
samples = sampler.chain[:, 50:, :].reshape((-1, ndim))
print np.mean(samples,axis=0)
