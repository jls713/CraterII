import numpy as np
from scipy.optimize import brentq
from scipy.signal import argrelextrema
from scipy.integrate import quad
import sys
sys.path.append('external/')
import nbody_tools
import aa_py


# Crater II properties
lb = np.deg2rad(np.array([282.908, 42.028]))
distance = 117.5


def crater2_eq():
    Gal = np.array([lb[0], lb[1], distance])
    return aa_py.GalacticToEquatorial(Gal)[:2]


vlos = 87.5
caldwell_propermotion = np.array([-0.18, -0.14])
caldwell_propermotion_error = np.array([0.16, 0.19])
caldwell_pmmag = 0.1040092

# Orbit integration
Pot = aa_py.GalPot('/home/jls/work/code/Torus/pot/PJM16_best.Tpot')
Orb = aa_py.Orbit(Pot, 1e-14)

R0 = np.array([8.29, 0.014,
               11.1, 12.24 +
               np.sqrt(8.29 * -Pot.Forces(np.array([8.29, 0., 0.]))[0]),
               7.25])


def crater2_solarreflex():
    Gal = np.array([lb[0], lb[1], distance])
    X = aa_py.GalacticToCartesian(Gal, R0)
    return aa_py.GalacticToEquatorial(
        aa_py.CartesianToGalactic(np.append(X, np.array([0., 0., 0.])),
                                  R0))[-2:]


def integrate(x, t, frac=1, dt=None):
    if dt is None:
        dT = frac * Pot.torb(x)
    else:
        dT = dt
    r = np.zeros((int(t / dT) + 2, 6))
    r[0] = x
    i, T = 1, 0
    times = [0.]
    while(T < t):
        r[i] = Orb.integrate(r[i - 1], dT, dT, False)
        T += dT
        i += 1
        times += [T]
    return r, times

def integrate_orbits_forwards(Eq, total_time=13.7):
    X = aa_py.GalacticToCartesian(aa_py.EquatorialToGalactic(Eq), R0)
    if(Pot.energy(X) > 0.):
        return np.array([]), np.array([])
    # X[3:] *= -1.
    R, T = integrate(X, total_time * 0.977775, frac=0.0001)
    indx = argrelextrema(np.sqrt(R.T[0]**2 + R.T[1]**2 + R.T[2]**2),
                         np.greater)[0]
    if len(indx):
        indx = indx[-1]
        return R[:indx], T[:indx]
    else:
        return R, T


def integrate_orbits_backwards(Eq, total_time=13.7):
    X = aa_py.GalacticToCartesian(aa_py.EquatorialToGalactic(Eq), R0)
    if(Pot.energy(X) > 0.):
        return np.array([]), np.array([])
    X[3:] *= -1.
    R, T = integrate(X, total_time * 0.977775, frac=0.01)
    indx = argrelextrema(np.sqrt(R.T[0]**2 + R.T[1]**2 + R.T[2]**2),
                         np.greater)[0]
    if len(indx):
        indx = indx[-1]
        return R[:indx], T[:indx]
    else:
        return R, T


def pericentre(R):
    return np.min(np.sqrt(R.T[0]**2 + R.T[1]**2 + R.T[2]**2))

def pericentre_full(R):
    return R[np.argmin(np.sqrt(R.T[0]**2 + R.T[1]**2 + R.T[2]**2))]


def apocentre(R):
    return np.max(np.sqrt(R.T[0]**2 + R.T[1]**2 + R.T[2]**2))


def peri_times_backwards(R, T):
    ''' in units of Gyr '''
    T = np.array(T)/nbody_tools.kms2kpcGyr
    return T[argrelextrema(np.sqrt(R.T[0]**2 + R.T[1]**2 + R.T[2]**2),
                           np.less)[0]]

def peri_times(R,T):
    ''' in units of Gyr '''
    return T[-1]/nbody_tools.kms2kpcGyr-peri_times_backwards(R,T)

def apo_times_backwards(R, T):
    ''' in units of Gyr '''
    T = np.array(T)/nbody_tools.kms2kpcGyr
    return T[argrelextrema(np.sqrt(R.T[0]**2 + R.T[1]**2 + R.T[2]**2),
                           np.greater)[0]]

def apo_times(R,T):
    ''' in units of Gyr '''
    return T[-1]/nbody_tools.kms2kpcGyr-apo_times_backwards(R,T)

def period(R, T):
    ''' in units of Gyr '''
    peri = peri_times(R, T)
    apo = apo_times(R, T)
    if len(peri) > len(apo):
        if peri[0] > apo[0]:
            peri = peri[:-1]
        else:
            peri = peri[1:]
    elif len(peri) < len(apo):
        if apo[0] > peri[0]:
            apo = apo[:-1]
        else:
            apo = apo[1:]
    return 2.*np.median(np.abs(peri - apo))

def OmegaA(R,T):
    IA = argrelextrema(np.sqrt(R.T[0]**2 + R.T[1]**2 + R.T[2]**2),
                           np.greater)[0][0]
    VR = np.cross(R[IA][3:],R[IA][:3])
    return np.sqrt(np.sum(VR**2))/np.sum(R[IA][:3]**2)


def OmegaP(R,T):
    IP = argrelextrema(np.sqrt(R.T[0]**2 + R.T[1]**2 + R.T[2]**2),
                           np.less)[0][0]
    VR = np.cross(R[IP][3:],R[IP][:3])
    return np.sqrt(np.sum(VR**2))/np.sum(R[IP][:3]**2)

def crossing_time(r, rs, c=20.):
    return np.pi * np.sqrt(r**3 / (G * mass_profile(r, rs, c=c)))


def find_max_dynamical_radius(T, rs, c=20.):
    return brentq(lambda x: crossing_time(x, rs, c=c) - T,
                  0.01 * rs, 1000. * rs)


def count_pericentres(R):
    return len(argrelextrema(np.sqrt(R.T[0]**2 + R.T[1]**2 + R.T[2]**2),
                             np.less)[0])


def penarrubia_tracks(x):
    ''' x = Mc/Mc(t=0) within sphere of radius Rc (measured at initial time -- fixed) '''
    alpha = {'sigma': 0.,'rh':2.416,  'Rc': 1.5, 'ck':1.75}
    beta = {'sigma': 0.5, 'rh': 0.656, 'Rc': 0.65, 'ck':0.}
    return 2.**alpha['sigma'] * x**beta['sigma'] / (1. + x)**alpha['sigma'],\
        2.**alpha['rh'] * x**beta['rh'] / (1. + x)**alpha['rh'],\
        2.**alpha['Rc'] * x**beta['Rc'] / (1. + x)**alpha['Rc'],\
        2.**alpha['ck'] * x**beta['ck'] / (1. + x)**alpha['ck']

def penarrubia_tracks_Mh(x):
    ''' x = Mh/Mh(t=0) within sphere of radius Rh (measured at initial time -- fixed) '''
    x+=1e-40 ## Fudge to stop zeros breaking everything
    alpha = {'sigma': -0.121,'rh':2.257,  'Rc': 1.351, 'ck':1.715}
    beta = {'sigma': 0.451, 'rh': 0.593, 'Rc': 0.592, 'ck':-0.00387}
    return 2.**alpha['sigma'] * x**beta['sigma'] / (1. + x)**alpha['sigma'],\
        2.**alpha['rh'] * x**beta['rh'] / (1. + x)**alpha['rh'],\
        2.**alpha['Rc'] * x**beta['Rc'] / (1. + x)**alpha['Rc'],\
        2.**alpha['ck'] * x**beta['ck'] / (1. + x)**alpha['ck']


def penarrubia_tracks_Mh_adjusted(x):
    ''' x = Mh/Mh(t=0) within sphere of radius Rh (measured at initial time -- fixed) '''
    x+=1e-40 ## Fudge to stop zeros breaking everything
    alpha = {'sigma': -0.121,'rh':0.6,  'Rc': 1.351, 'ck':1.715}
    beta = {'sigma': 0.451, 'rh': 0.45, 'Rc': 0.592, 'ck':-0.00387}
    return 2.**alpha['sigma'] * x**beta['sigma'] / (1. + x)**alpha['sigma'],\
        2.**alpha['rh'] * x**beta['rh'] / (1. + x)**alpha['rh'],\
        2.**alpha['Rc'] * x**beta['Rc'] / (1. + x)**alpha['Rc'],\
        2.**alpha['ck'] * x**beta['ck'] / (1. + x)**alpha['ck']


G = 4.300918e-6
H0 = 67.8e-3

def dlnMdlnr(r,delta_r=0.1):
    return 2+(np.log(-Pot.Forces(np.array([r+delta_r, 0., 0.]))[0])-np.log(-Pot.Forces(np.array([r-delta_r, 0., 0.]))[0]))/(np.log(r+delta_r)-np.log(r-delta_r))

def Minsider(r):
    return r**2 * (-Pot.Forces(np.array([r, 0., 0.]))[0]) / G

def d2Potdr2(r, delta_r=0.1):
    return (-Pot.Forces(np.array([r+delta_r, 0., 0.]))[0]+Pot.Forces(np.array([r-delta_r, 0., 0.]))[0])/(2.*delta_r)

def d2Potdr2_full(r, delta_r=0.1):
    runit = r/np.sqrt(np.sum(r**2))
    delta_r_vec = delta_r*runit
    return (-np.dot(Pot.Forces(r+delta_r_vec),runit)+np.dot(Pot.Forces(r-delta_r_vec),runit))/(2.*delta_r)

def tidal_radius_fixed_mass(r_gal, Mc, gamma=0., Omega=None):
    if gamma is None:
        if Omega is None:
            print 'Omega or gamma must be non-None'
        return np.power(Mc*G/(Omega**2-d2Potdr2(r_gal)), 1./3.)
    return np.power(Mc / (r_gal**2 *
                          (-Pot.Forces(np.array([r_gal, 0., 0.]))[0]) / G)
                    / (2. - gamma), 1. / 3.) * r_gal


def mass_profile(r, rs, c=20., Delta=200.):
    rhoc = 3 * H0 * H0 / (8. * np.pi * G)
    const = (c * rs)**3 * (Delta * rhoc * 4. / 3. * np.pi) / \
        (np.log(1. + c) - c / (1. + c))
    return const * (np.log((rs + r) / rs) - r / (rs + r))

def tidal_radius_double_power(rgal,rs,rho0,abg,gamma=0.,Omega=None):
    return brentq(lambda x: x -
                  tidal_radius_fixed_mass(rgal,
		  nbody_tools.truncated_double_power_mass(x, rs, rho0, abg),
                  gamma=gamma, Omega=Omega),
                  0.0001 * rs, 10000. * rs)


def tidal_radius(rgal, rs, c=20., Delta=200., gamma=0., Omega=None):
    return brentq(lambda x: x -
                  tidal_radius_fixed_mass(rgal, mass_profile(x, rs, c=c,
                                                          Delta=Delta),
                  gamma=gamma, Omega=Omega),
                  0.0001 * rs, 10000. * rs)


def mass_loss(rp, ra, Nperi, rs, c=20., Delta=200., gamma=0., Omega=None):
    if Omega is None:
        Omega = [None, None]
    if type(gamma) is list:
        gammap=gamma[0]
        gammaa=gamma[1]
    else:
        gammap=gamma
        gammaa=gamma
    rtp = tidal_radius(rp, rs, c=c, Delta=Delta, gamma=gammap, Omega=Omega[0])
    rta = tidal_radius(ra, rs, c=c, Delta=Delta, gamma=gammaa, Omega=Omega[1])
    return np.power(mass_profile(rtp, rs, c=c, Delta=Delta) /
                    mass_profile(rta, rs, c=c, Delta=Delta),
                    Nperi)


def hayashi_rte(mbnd):
    return np.power(10., 1.02 + 1.38 * np.log10(mbnd) +
                    0.37 * (np.log10(mbnd))**2)


def hayashi_ft(mbnd):
    return np.power(10., -0.007 + 0.35 * np.log10(mbnd) +
                    0.39 * (np.log10(mbnd))**2 + 0.23 * (np.log10(mbnd))**3)


def hayashi_profile(r, rs, mbnd, c=20., Delta=200.):
    return nbody_tools.rho_NFW(r, rs, c=c, Delta=Delta) * \
        hayashi_ft(mbnd) / (1. + (r / hayashi_rte(mbnd) / rs)**3)


def hayashi_mass(rlim, rs, mbnd, c=20., Delta=200.):
    return 4. * np.pi * quad(lambda x: x * x *
                             hayashi_profile(x, rs, mbnd, c=c, Delta=Delta),
                             rlim[0], rlim[1])[0]


def mass_loss_hayashi(rp, ra, Nperi, rs, c=20., Delta=200., rlim=0.,
                      rtotal=None, gamma=0., Omega=None):
    if Omega is None:
        Omega = [None, None]
    if type(gamma) is list:
        gammap=gamma[0]
        gammaa=gamma[1]
    else:
        gammap=gamma
        gammaa=gamma
    if Nperi==0:
        return 1.
    rtp = tidal_radius(rp, rs, c=c, Delta=Delta, gamma=gammap, Omega=Omega[0])
    # print rtp
    rta = tidal_radius(ra, rs, c=c, Delta=Delta, gamma=gammaa, Omega=Omega[1])
    mbnd = mass_profile(rtp, rs, c=c, Delta=Delta) / \
        mass_profile(rta, rs, c=c, Delta=Delta)
    # print mbnd
    # mbnd = nbody_tools.truncated_NFW_mass(rtp, rs, c=c, rt=rta, Delta=Delta) / \
    #     nbody_tools.truncated_NFW_mass(rta, rs, c=c, rt=rta, Delta=Delta)
    # print mbnd
    for n in range(Nperi - 1):
        fnn = lambda x: x - \
                     tidal_radius_fixed_mass(rp, hayashi_mass([0., x], rs, mbnd,
                                                             c=c,
                                                             Delta=Delta), gamma=gammap, Omega=Omega[0])
        if fnn(rlim+1e-40*rs)*fnn(100.*rs)>0.:
            return 0.
        rtp = brentq(fnn, rlim+1e-40*rs, 100. * rs)
        mfrac = hayashi_mass([0.,rtp],rs,mbnd,c=c,Delta=Delta)/hayashi_mass([0.,np.inf],rs,mbnd,c=c,Delta=Delta)
        mbnd = mbnd*mfrac
    if rtotal is None:
        return mbnd
    else:
        return hayashi_mass([0.,rtotal],rs,mbnd,c=c,Delta=Delta)/mass_profile(rtotal, rs, c=c, Delta=Delta)
