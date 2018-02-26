import numpy as np
from scipy.special import ellipeinc as E_BT, ellipkinc as F_BT

def binney_tremaine_virial_ratio_full(ba,ca):
    k=(1.-ba*ba)/(1.-ca*ca)
    kp=1.-k
    theta=np.arccos(ca)
    st=np.power(1.-ca*ca,0.5)
    e = E_BT(theta,k)
    f = F_BT(theta,k)
    a1=(f-e)/k
    a2=(e-kp*f-ca/ba*k*st)/k/kp
    a3=(ba/ca*st-e)/kp
    f1 = a1/a3/ca/ca
    f2 = a2/a3*ba*ba/ca/ca
    return f1,f2

def binney_tremaine_virial_ratio_oblate(ca):
    e = np.sqrt(1.-ca*ca)
    return .5*(np.arcsin(e)/e-ca)/(1./ca-np.arcsin(e)/e)/ca/ca

def binney_tremaine_virial_ratio_prolate(ca):
    e = np.sqrt(1.-ca*ca)
    return .5*(1./ca/ca-.5*np.log((1.+e)/(1.-e))/e)/(.5/e*np.log((1.+e)/(1.-e))-1.)*ca*ca

def sigma_tot_to_sigma_los(ba,ca,theta,phi):
    ''' Returns <sigma_tot^2>/<sigma_los^2> '''
    if(ba==1. and ca==1.):
        return 1.
    elif(ba==1.):
        f1=binney_tremaine_virial_ratio_oblate(ca)
        f2=f1
    elif(ba==ca):
        f1=1./binney_tremaine_virial_ratio_prolate(ca)
        f2=1.
    else:
        f1,f2 = binney_tremaine_virial_ratio_full(ba,ca)
    ct,st=np.cos(theta),np.sin(theta)
    cp,sp=np.cos(phi),np.sin(phi)
    fr=ct*ct+st*st*cp*cp*f1+st*st*sp*sp*f2
    return (1.+f1+f2)/3./fr

def major_axis_length(ba,ca,theta,phi):
    ba2=ba*ba
    ca2=ca*ca
    c2t = np.cos(2.*theta)
    c2p = np.cos(2.*phi)
    st2 = np.sin(theta); st2*=st2
    ct2 = np.cos(theta); ct2*=ct2
    sp2 = np.sin(phi); sp2*=sp2
    cp2 = np.cos(phi); cp2*=cp2
    Denom = (ba2*ct2+ca2*st2*(ba2*cp2+sp2))
    return 4./np.sqrt(-(-6.*ba2-4.*ca2-2.*ba2*c2t+4.*ca2*c2t+ba2*np.cos(2.*theta-2.*phi)-2.*ba2*np.cos(2.*phi)+ba2*np.cos(2.*(theta+phi))-2.*(3.+c2t-2.*c2p*st2)+np.sqrt(4.*(3.*(1.+ba2)+2.*ca2+(1.+ba2-2.*ca2)*c2t+2.*(-1.+ba2)*c2p*st2)**2+64.*(-2.*ba2-(1.+ba2)*ca2+(-2.*ba2+(1.+ba2)*ca2)*c2t+2.*(1.-ba)*(1.+ba)*ca2*c2p*st2)))/Denom)

def ellipticity(ba,ca,theta,phi):
    ba2=ba*ba
    ca2=ca*ca
    c2t = np.cos(2.*theta)
    c2p = np.cos(2.*phi)
    st2 = np.sin(theta); st2*=st2
    ct2 = np.cos(theta); ct2*=ct2
    sp2 = np.sin(phi); sp2*=sp2
    cp2 = np.cos(phi); cp2*=cp2
    A = (1.-ca2)*ct2+(1.-ba2)*st2*sp2+ba2+ca2
    B = ((1.-ca2)*ct2-(1.-ba2)*st2*sp2-ba2+ca2)**2+4.*(1.-ba2)*(1.-ca2)*st2*ct2*sp2
    C = ba2*ct2+ca2*st2*(ba2*cp2+sp2)
    return 1.-np.sqrt((A-np.sqrt(B))/(A+np.sqrt(B)))

def minor_axis_angle(ba,ca,theta,phi):
    T = (1.-ba*ba)/(1.-ca*ca)
    return .5*np.arctan2((2.*T*np.sin(phi)*np.cos(phi)*np.cos(theta)),(np.sin(theta)*np.sin(theta)-T*(np.cos(phi)*np.cos(phi)-np.sin(phi)*np.sin(phi)*np.cos(theta)*np.cos(theta))))

def radial_factor(ba,ca,theta,phi):
    return 1./major_axis_length(ba,ca,theta,phi)

def correction_factor(ba,ca,theta,phi):
    return radial_factor(ba,ca,theta,phi)*sigma_tot_to_sigma_los(ba,ca,theta,phi)*np.power(ba*ca,1./3.)

def perpendicular_distance(logr,logs):
    slope, intercept = 0.5, 1.267
    return (slope*logr-logs+intercept)/np.sqrt(slope*slope+1.)
