import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from orbit import penarrubia_tracks

def king_surface(R,rc,rt):
    return np.power(1./np.sqrt(1.+(R/rc)**2)-1./np.sqrt(1.+(rt/rc)**2),2.)

def plum_surface(R,rc,rt):
    return 1./(1.+(R/rc)**2)**2

def mass_contained(r,rc,c):
    rt = c*rc
    return quad(lambda x: 2.*np.pi*x*king_surface(x,rc,rt),0.,r)[0]

def mass_contained_plummer(r,rc):
    return quad(lambda x: 2.*np.pi*x*plum_surface(x,rc,0.),0.,r)[0]

def half_mass_radius(rc,c):
    M = mass_contained(rc*c,rc,c)
    return brentq(lambda y: mass_contained(y,rc,c)-0.5*M,0.1*rc,5.*rc)

def king_density(r,rc,rt):
    x = np.sqrt((1.+(r/rc)**2)/(1.+(rt/rc)**2))
    return (np.arccos(x)/x-np.sqrt(1.-x*x))/x/x

def mass_contained_sphere(r,rc,c):
    rt = c*rc
    return quad(lambda x: 4*np.pi*x*x*king_density(x,rc,rt),0.,r)[0]

from scipy.optimize import minimize

def fitter(xx,yy):
    def fn(p):
        alpha,beta=p[0],p[1]
        return np.sum((yy-2.**alpha*xx**beta/(1.+xx)**alpha)**2)
    return minimize(fn,[0.,0.5]).x

def find_rh_relation():
	xx = np.logspace(-3.,0.)
	PP = penarrubia_tracks(xx)
	Rc = PP[2]
	ck = PP[3]*5.
	Rh = np.array([half_mass_radius(rc,c) for rc,c in zip(Rc,ck)])
	import matplotlib.pyplot as plt
	plt.plot(xx,Rh/Rh[-1])
	plt.plot(xx,Rc)
	plt.semilogy()
	plt.semilogx()
	alpha,beta=2.41,0.65
	plt.plot(xx,2.**alpha*xx**beta/(1.+xx)**alpha)
	plt.show(block=True)
        print fitter(xx,Rh/Rh[-1])

def gx(xx,alpha,beta):
    return 2**alpha*xx**beta/(1.+xx)**alpha

def find_mh_mh0():
    rh = half_mass_radius(1.,5.)
    mh_mc = mass_contained_sphere(rh,1.,5.)/mass_contained_sphere(1.,1.,5.)
    xx = np.logspace(-2.,0.)
    PP = penarrubia_tracks(xx)
    Rc = PP[2]
    ck = PP[3]*5.

    mc = np.array([mass_contained_sphere(1.,rc,c) for rc,c in zip(Rc,ck)])
    mc = mc/mc[-1]
    norm = xx/mc

    mc = mc*norm

    mh = np.array([mass_contained_sphere(rh,rc,c) for rc,c in zip(Rc,ck)])*norm

    mc_notfixedradius = np.array([mass_contained_sphere(rc,rc,c) for rc,c in zip(Rc,ck)])*norm

    mh_notfixedradius = np.array([mass_contained_sphere(rrh,rc,c) for rrh,rc,c in zip(PP[1],Rc,ck)])*norm

    import matplotlib.pyplot as plt
    plt.plot(xx,PP[1],'.')
    plt.plot(mc_notfixedradius/mc_notfixedradius[-1],PP[1],'.')
    plt.plot(mh/mh[-1],PP[1],'.')
    plt.plot(mh_notfixedradius/mh_notfixedradius[-1],PP[1],'.')
    plt.semilogx()
    plt.semilogy()
    plt.show(block=True)
    plt.clf()
    plt.plot(xx,mc_notfixedradius/mc_notfixedradius[-1],'.')
    plt.plot(xx,mh/mh[-1],'.')
    plt.plot(xx,0.95*xx)
    plt.plot(xx,mh_notfixedradius/mh_notfixedradius[-1],'.')
    plt.semilogx()
    plt.semilogy()
    plt.show(block=True)

    ab_sig0=fitter(mh/mh[-1],PP[0])
    ab_rh=fitter(mh/mh[-1],PP[1])
    ab_rc=fitter(mh/mh[-1],PP[2])
    ab_ck=fitter(mh/mh[-1],PP[3])

    print ab_sig0,ab_rh,ab_rc,ab_ck

    print fitter(xx,PP[1])

    plt.clf()
    plt.plot(mh/mh[-1],PP[0])
    plt.plot(mh/mh[-1],gx(mh/mh[-1],*ab_sig0))
    plt.semilogx()
    plt.semilogy()
    plt.show(block=True)
    plt.clf()
    plt.plot(mh/mh[-1],PP[1])
    plt.plot(mh/mh[-1],gx(mh/mh[-1],*ab_rh))
    plt.semilogx()
    plt.semilogy()
    plt.show(block=True)

if __name__=="__main__":
    find_mh_mh0()
    find_rh_relation()
