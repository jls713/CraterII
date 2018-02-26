import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../../code/m2m')
sys.path.append('../../code/nbody')
import m2m_tools as m2m
import nbody_tools
from subprocess import call
import seaborn as sns
import pandas as pd
import aa_py
import os


def contour2D(xdata,ydata,axis,ranges=None,bins=40,with_imshow=False,with_contours=True,label=None,fabsflag=False,text=None,cmap=plt.cm.hot,with_kde=False,kde_bandwidth=0.01,alpha=1.,COLA=sns.color_palette()[0],weights=None):
    ''' Plots 2D histogram of data[cols[0]] against data[cols[1]]
        with the option of underlaying with filled contour '''
    h2,x,y = np.histogram2d(xdata,ydata,bins=bins,range=ranges,normed=True,weights=weights)
    ex = [x[0],x[-1],y[0],y[-1]]
    if(with_imshow):
        c = axis.imshow(np.log(np.rot90(h2/np.max(h2))),cmap=cmap,extent=ex,aspect='auto',vmin=0.,vmax=1.,alpha=alpha)
    if(with_contours):
        c = axis.contour(.5*(x[1:]+x[:-1]),.5*(y[1:]+y[:-1]),
                         np.log(h2.T)-np.log(np.min(h2.T[h2.T>0.])),
                     origin='lower',
                     colors=COLA,
                     label=label)
    return c,(h2,ex)


def generate_nfw(prop,shift=True):
    ''' Generate NFW profile -- if ba!=1 or ca!=1 run M2M first '''

    simFileDM=prop['simFileDM']
    NDM=prop['NDM']
    rtidal=prop['rtidal']
    ba=prop['ba']
    ca=prop['ca']
    simFileGRAVINPUT=prop['simFileGRAVINPUT']
    lmax=0
    if ba!=1. or ca!=1.:
      lmax=12

    Omega = prop['Omega']*(prop['tunit']/(2.*np.pi))

    m2m.prepareM2M(simFileDM+'_init',
          nbody=NDM,nsample=1000,c_over_a=ca,b_over_a=ba,
          scale=1.,nmax=20,lmax=lmax,inner=1.,outer=3.,rs=1.,
          eta=1.,r_t=rtidal/prop['Rscale'], b=0.,Omegap=Omega)

    # if(ba<1. or ca<1.):
    m2m.runM2M(input_file=simFileDM+'_init',
               output_file=simFileDM+'_m2moutput',
               log_file=simFileDM+'_m2mlog',
               parallel_threads=8,
               tstop=70.,
               dT_rs=5.,Omegap=Omega,tfac=prop['tfac'],dT=0.2)
    call(['rm',simFileGRAVINPUT])
    m2m.extract_final_snapshot(simFileDM+'_m2moutput',simFileGRAVINPUT)

    vscale = np.sqrt(prop['G']*prop['Mscale']/prop['Rscale'])

    if(shift):
      nbody_tools.shift_scale_snapshot(simFileGRAVINPUT,
                                     prop['simFileGRAVINPUT_scaled'],
                                     prop['Rscale'],
                                     vscale,
                                     prop['Mscale'],
                                     prop['XV'])

    eps = nbody_tools.optimal_softening(NDM)*prop['Rscale']
    Integration_Time = prop['IntTime']
    nbody_tools.run_under_gravity(prop['simFileGRAVINPUT_scaled'],
                                  prop['simFileGRAVOUTPUT'],
                                  Integration_Time,
                                  pot=prop['pot'],
                                  potpars=prop['potpars'],
                                  potfile=prop['potfile'],
                                  epsilon=eps,
                                  tau_step = prop['tunit']/np.power(2.,7),
                                  logfile=prop['simFileGRAVLOG'],
                                  fea=0.1,fpa=0.5,step=Integration_Time/25.,
                                  Grav=prop['G'],
                                  threads=8)

    return

def generate_plum_in_nfw(prop,nmax=20,int_step_frac=25.,beta_stars=-0.2,
                         stellar_inner_cusp=0.,dark_inner_cusp=1.):
    ''' Generate NFW in Plummer profile -- if ba!=1 or ca!=1 run M2M first '''

    simFileDM=prop['simFileDM']
    simFileS=prop['simFileS']
    NDM=prop['NDM']
    NStars=prop['NStars']
    rtidal=prop['rtidal']
    ba=prop['ba']
    ca=prop['ca']
    simFileDMGRAVINPUT=prop['simFileDMGRAVINPUT']
    simFileSGRAVINPUT=prop['simFileSGRAVINPUT']
    lmax=0
    if ba!=1. or ca!=1.:
      lmax=12

    # if prop['exptype']=='Zhao':
    #   TYPE=False
    # else:
    #   TYPE=True

    Omega = prop['Omega']*(prop['tunit']/(2.*np.pi))

    ## For dark matter r_s=scale

    m2m.prepareM2M(simFileDM+'_init',
         nbody=NDM,nsample=100,c_over_a=ca,b_over_a=ba,
         alpha=1.,nmax=nmax,lmax=lmax,inner=dark_inner_cusp,
         outer=3.,rs=1.,
         eta=1.,r_t=rtidal/prop['Rscale'], b=0.,
         twonu=prop['exptype'],Omegap=Omega)
         # ,r_a=1.)

    # For stars r_s!=scale so generate two files -- one for potential with scale=1 (scale of DM) and one for Anlm (scale=r_s)
    # Here we can choose scale and alpha to better match stars
    m2m.prepareM2M(simFileS+'_init',
         nbody=NStars,nsample=1000,c_over_a=ca,b_over_a=ba,
         scale=prop['rs']/prop['Rscale'],nmax=nmax,lmax=lmax,
         inner=stellar_inner_cusp,outer=5.,
         rs=prop['rs']/prop['Rscale'],
         eta=2.,
         r_t=rtidal/prop['Rscale'],
         # r_t=5.*prop['rs']/prop['Rscale'],
         b=beta_stars,M=prop['Ms']/prop['Mscale'],
         alpha=.51,potfile=simFileDM+'_init',twonu=prop['exptype'],
         Omegap=Omega)

    ## For this one, scale and alpha must be the same as for DM
    m2m.prepareM2M(simFileS+'_pot_init',
         nbody=NStars,nsample=100,c_over_a=ca,b_over_a=ba,
         scale=1.,nmax=nmax,lmax=lmax,
         inner=stellar_inner_cusp,outer=5.,
         rs=prop['rs']/prop['Rscale'],
         eta=2.,
         r_t=rtidal/prop['Rscale'],
         # r_t=5.*prop['rs']/prop['Rscale'],
         b=beta_stars,M=prop['Ms']/prop['Mscale'],
         alpha=1.,potfile=simFileDM+'_init',twonu=prop['exptype'],
         Omegap=Omega)

    # ,r_a=1.)
    call(['rm',simFileDM+'_init.ini'])
    call(['rm',simFileDM+'_init.An'])
    call(['rm',simFileDM+'_init.Sn'])
    m2m.prepareM2M(simFileDM+'_init',
         nbody=NDM,nsample=1000,
         c_over_a=ca,b_over_a=ba,
         scale=1.,nmax=nmax,lmax=lmax,
         inner=dark_inner_cusp,outer=3.,rs=1.,
         eta=1.,r_t=rtidal/prop['Rscale'],
         b=0., alpha=1.,
         potfile=simFileS+'_pot_init',
         twonu=prop['exptype'],Omegap=Omega)
    # ,r_a=1.)

    m2m.sumAnlm(simFileDM+'_init',
               simFileS+'_pot_init',
               prop['simFile']+'_init')

    #    # if(ba<1. or ca<1.):
    m2m.runM2M(input_file=simFileDM+'_init',
               output_file=simFileDM+'_m2moutput',
               log_file=simFileDM+'_m2mlog',
               parallel_threads=8,
               tstop=45.,
               dT_rs=5.,
               accfile=prop['simFile']+'_init',
               m2m_constraints=[["RhoNLM",simFileDM+'_init',""]],
               Omegap=Omega,tfac=prop['tfac'],dT=45.)
    m2m.runM2M(input_file=simFileS+'_init',
               output_file=simFileS+'_m2moutput',
               log_file=simFileS+'_m2mlog',
               parallel_threads=8,
               tstop=45.,
               dT_rs=5.,
               accfile=prop['simFile']+'_init',
               m2m_constraints=[["RhoNLM",simFileS+'_init',""]],
               Omegap=Omega,tfac=prop['tfac'],dT=45.)

    call(['rm',simFileDMGRAVINPUT])
    call(['rm',simFileSGRAVINPUT])
    m2m.extract_final_snapshot(simFileDM+'_m2moutput',simFileDMGRAVINPUT)
    m2m.extract_final_snapshot(simFileS+'_m2moutput',simFileSGRAVINPUT)

    # else:
    #     call(['rm',simFileDMGRAVINPUT])
    #     call(['rm',simFileSGRAVINPUT])
    #     m2m.extract_final_snapshot(simFileDM+'_init',simFileDMGRAVINPUT)
    #     m2m.extract_final_snapshot(simFileS+'_init',simFileSGRAVINPUT)

    vscale = np.sqrt(prop['G']*prop['Mscale']/prop['Rscale'])
    rot=None
    rotaxis=None
    if 'rot' in prop.keys():
        rot=prop['rot']
        rotaxis=prop['rotaxis']
    nbody_tools.stack_shift_scale_snapshot(simFileDMGRAVINPUT,
                                     simFileSGRAVINPUT,
                                     prop['simFileGRAVINPUT_scaled'],
                                     prop['Rscale'],
                                     vscale,
                                     prop['Mscale'],
                                     prop['XV'],
                                     rot=rot,
                                     rotaxis=rotaxis)

    # return

    eps = nbody_tools.optimal_softening(NDM, r_m2=1.)*prop['Rscale']

    Integration_Time = prop['IntTime']

    nbody_tools.run_under_gravity(prop['simFileGRAVINPUT_scaled'],
                                  prop['simFileGRAVOUTPUT'],
                                  Integration_Time,
                                  pot=prop['pot'],
                                  potpars=prop['potpars'],
                                  potfile=prop['potfile'],
                                  epsilon=eps,
                                  tau_step = prop['tunit']/np.power(2.,3),
                                  logfile=prop['simFileGRAVLOG'],
                                  fea=0.2, # tau <= 0.2 sqrt(eps/acc)
                                  fpa=0.2, # tau <= 0.2 sqrt(acc)/pot
                                  step=Integration_Time/int_step_frac,
                                  Grav=prop['G'],
                                  threads=16)

    # nbody_tools.stack_shift_scale_snapshot(simFileDMGRAVINPUT,
    #                                  simFileSGRAVINPUT,
    #                                  prop['simFileGRAVINPUT_scaled'],
    #                                  1.,
    #                                  1.,
    #                                  1.,
    #                                  [0.,0.,0.,0.,0.,0.])

    # eps = nbody_tools.optimal_softening(NDM, r_m2=1.) #/10.

    # Integration_Time = prop['IntTime']

    # nbody_tools.run_under_gravity(prop['simFileGRAVINPUT_scaled'],
    #                               prop['simFileGRAVOUTPUT'],
    #                               Integration_Time/prop['tunit']*2.*np.pi,
    #                               pot=prop['pot'],
    #                               potpars=prop['potpars'],
    #                               potfile=prop['potfile'],
    #                               epsilon=eps,
    #                               tau_step = 2.*np.pi/np.power(2.,3),
    #                               logfile=prop['simFileGRAVLOG'],
    #                               fea=0.2, # tau <= 0.2 sqrt(eps/acc)
    #                               fpa=0.2, # tau <= 0.2 sqrt(acc)/pot
    #                               step=Integration_Time/int_step_frac/prop['tunit']*2.*np.pi,
    #                               Grav=1.,
    #                               threads=16)

def recentre(data, centre=None, mass_weight=True):
    if centre is None:
      centre, rmax = nbody_tools.shrinking_centre(data, mass_weight=mass_weight)
    for i in range(1,4):
        data[:,i]-=centre[i-1]
    for i in range(4,7):
        # data[:,i]-=np.average(data[:,i],weights=data.T[0])
        data[:,i]-=centre[i-1]

    # print 'Recentering using bound_centre'
    # print 'Recentering using bound_centre'
    # print 'Recentering using bound_centre'
    # print 'Recentering using bound_centre'
    return data

def radius(data,ba=1.,ca=1.):
    shape = np.array([1.,ba**2,ca**2])
    return np.sqrt(np.sum(data[:,1:4]**2/shape[np.newaxis,:],axis=1))

def radvel(data):
  r = radius(data)
  rv = (data.T[1]*data.T[4]+data.T[2]*data.T[5]+data.T[3]*data.T[6])/r
  return rv

def centre_of_mass_with_time(output,times):
    f,a=plt.subplots(1,3)
    for t in times:
        nbody_tools.extract_ascii_snapshot(simFileGRAVOUTPUT,
                                       simFileGRAVOUTPUTDATA,
                                       t)
        centre,rmax = nbody_tools.shrinking_centre(
                                    np.genfromtxt(simFileGRAVOUTPUTDATA))
        a[0].plot(centre[0],centre[1],'.')
        a[1].plot(centre[0],centre[2],'.')
        a[2].plot(centre[1],centre[2],'.')
    a[0].set_xlabel(r'$x$')
    a[0].set_ylabel(r'$y$')
    a[1].set_xlabel(r'$x$')
    a[1].set_ylabel(r'$z$')
    a[2].set_xlabel(r'$y$')
    a[2].set_ylabel(r'$z$')
    plt.savefig(output,bbox_inches='tight')

def moment_of_inertia_reduced(m,x,y,z,ba,ca,reduced=True):
    Iij = np.zeros((3,3))
    dn = np.ones_like(m)
    if reduced:
        dn = x*x+y*y/ba/ba+z*z/ca/ca
    Iij[0][0]=np.sum(x*x*m/dn)
    Iij[0][1]=np.sum(x*y*m/dn)
    Iij[1][1]=np.sum(y*y*m/dn)
    Iij[0][2]=np.sum(x*z*m/dn)
    Iij[1][2]=np.sum(y*z*m/dn)
    Iij[2][2]=np.sum(z*z*m/dn)
    Iij[1][0]=Iij[0][1]
    Iij[2][0]=Iij[0][2]
    Iij[2][1]=Iij[1][2]
    Iij = Iij/np.sum(m/dn)
    try:
	eig = np.linalg.eig(Iij)
    	fltr = np.argsort(np.fabs(eig[0]))[::-1]
    	evals=eig[0][fltr]
    	evecs=eig[1][:,fltr]
    	principal = evecs[:,0]
    	ba = np.sqrt(evals[1]/evals[0])
    	ca = np.sqrt(evals[2]/evals[0])
    	return ba,ca,principal,evecs[:,1],evecs[:,2]
    except:
        return 0.,0.,np.zeros(3),np.zeros(3),np.zeros(3)

def moment_of_inertia_diagonalization(data,x0,y0,z0,reduced=False,thresh=0.02,weight_fld='mass',maxit=20):
    if reduced:
        ba_p,ca_p=1.,1.
        m,x,y,z=data[weight_fld],data['x']-x0,data['y']-y0,data['z']-z0
        ba,ca,principal,second,third=moment_of_inertia_reduced(m,x,y,z,ba_p,ca_p,reduced=True)
        if(ba==0. and ca==0.):
            return ba,ca,principal,second,third
        mat = np.vstack((principal,second,third))
        i=0
        while(np.fabs((ba-ba_p)/ba)>thresh and np.fabs((ca-ca_p)/ca)>thresh and i<maxit):
            ba_p,ca_p=ba,ca
            vec = np.vstack((x,y,z))
            x = np.dot(principal,vec)
            y = np.dot(second,vec)
            z = np.dot(third,vec)
            ba,ca,principal,second,third=moment_of_inertia_reduced(m,x,y,z,ba_p,ca_p,reduced=True)
            if(ba==0. and ca==0.):
                return ba,ca,principal,second,third
            mat = np.dot(np.vstack((principal,second,third)),mat)
            i+=1
        if i==maxit:
            return np.nan,np.nan,np.array([np.nan,np.nan,np.nan]),np.array([np.nan,np.nan,np.nan]),np.array([np.nan,np.nan,np.nan])
        return ba,ca,mat[0],mat[1],mat[2]
    else:
        return moment_of_inertia_reduced(data[weight_fld],data['x']-x0,data['y']-y0,data['z']-z0,1.,1.,reduced=reduced)

def load_data(fil,NDM):
    names = ['m','x','y','z','vx','vy','vz','key','pot','ax','ay','az']
    dty = {n:np.float64 for n in names}
    data = pd.read_csv(fil,names=names,sep=r'\s+',comment='#',dtype=dty)
    data['dm']=0.
    data.loc[:NDM,'dm']=1.
    return data

def data_to_pandas(data):
    df = pd.DataFrame()
    names = ['mass','x','y','z','vx','vy','vz']
    for i,n in enumerate(names):
      df[n]=data.T[i]
    df['dm']=data.T[-1]
    return df

# def centre_meanshift(data):
#   from scikitlearn.cluster import MeanShift
#   print meanshift(data.T[])

def grab_snapshot(prop,time, with_centre=False, recentre_sim=True,
                  mass_weight=True, centre_method='bound'):
    ''' Recentres in position and velocity '''

    simFileGRAVOUTPUT=prop['simFileGRAVOUTPUT']
    simFileGRAVOUTPUTDATA = simFileGRAVOUTPUT+'.tmp'

    NDM = prop['NDM']
    snpsht_file = simFileGRAVOUTPUT

    centre_prev = np.zeros(6)

    nbody_tools.extract_ascii_snapshot(snpsht_file,
                                 simFileGRAVOUTPUTDATA,
                                 time)
    data_1 = load_data(simFileGRAVOUTPUTDATA,NDM)
    call(['rm',simFileGRAVOUTPUTDATA])

    if centre_method=='dens':
      centre = nbody_tools.dens_centre(simFileGRAVOUTPUT, times=time)
    elif centre_method=='bound':
      centre = nbody_tools.bound_centre(simFileGRAVOUTPUT, prop['epsilon'],times=time,Kparticles=np.min([512,int(0.05*len(data_1))]))
    elif centre_method=='shrink':
      centre, rmax = nbody_tools.shrinking_centre(data_1,
                                                mass_weight=mass_weight)
    else:
      print 'Centre method',centre_method,'not valid!'

    data_1 = nbody_tools.trim_bound(data_1.values,centre=centre,
                                    Grav=prop['G'],
                                    mass_weight=mass_weight)
    NDM = np.count_nonzero(data_1.T[-1]==1.)

    snpsht_file = prop['simFileGRAVOUTPUT']+'_boundfraction'

    niter = 0
    nitermax = 20
    nbound = len(data_1)
    # print niter, centre, nbound

    # while (np.sqrt(np.sum((centre_prev[:3]-centre[:3])**2))>1e-2 and np.sqrt(np.sum((centre_prev[3:]-centre[3:])**2))>1e-1 and niter<nitermax and nbound>10):
    nbound_prev=nbound+1000
    while(nbound_prev-nbound>20 and nbound>10 and niter<nitermax):
        nbound_prev = nbound
        centre_prev = np.copy(centre)

        m2m.write_nparray_to_snapshot(data_1[:,:7],
                                      snpsht_file)

        nbody_tools.run_under_gravity(snpsht_file,
                                      snpsht_file+'_out',
                                      0.,
                                      pot=prop['pot'],
                                      potpars=prop['potpars'],
                                      potfile=prop['potfile'],
                                      epsilon=prop['epsilon'],
                                      tau_step = prop['tunit']/np.power(2.,3),
                                      logfile=prop['simFileGRAVLOG']+'_bound',
                                      fea=0.2,
                                      fpa=0.2,
                                      step=0.,
                                      Grav=prop['G'],
                                      threads=8)
        call(['mv',snpsht_file+'_out',snpsht_file])

        nbody_tools.extract_ascii_snapshot(snpsht_file,
                                     simFileGRAVOUTPUTDATA,
                                     0.)
        data_1 = load_data(simFileGRAVOUTPUTDATA,NDM)
        call(['rm',simFileGRAVOUTPUTDATA])

        if centre_method=='dens':
          centre = nbody_tools.dens_centre(snpsht_file, times=0.)
        elif centre_method=='bound':
          centre = nbody_tools.bound_centre(snpsht_file, prop['epsilon'],
                                            times=0.,
                                            Kparticles=np.min([512,int(0.05*len(data_1))]))
        elif centre_method=='shrink':
          centre, rmax = nbody_tools.shrinking_centre(data_1,
                                                    mass_weight=mass_weight)
        else:
          print 'Centre method',centre_method,'not valid!'
        data_1 = nbody_tools.trim_bound(data_1.values,centre=centre,
                                        Grav=prop['G'],
                                        mass_weight=mass_weight)
        NDM = np.count_nonzero(data_1.T[-1]==1.)
        niter+=1
        nbound = len(data_1)
        # print niter, centre, nbound

    if snpsht_file is not simFileGRAVOUTPUT:
        if os.path.isfile(snpsht_file):
            call(['rm',snpsht_file])

    if(recentre_sim and len(data_1)):
      data_1 = recentre(data_1, centre=centre, mass_weight=mass_weight)
    if(with_centre):
      return data_1, np.array(centre)
    else:
      return data_1

def grab_snapshots(prop,times):
    ''' Recentres in position and velocity '''
    return grab_snapshot(prop,times[0]),grab_snapshot(prop,times[1])

def measure_density_profile(input_file,centre,nbins=30,at_least_n_per_bin=30):
    m2m.process_density(input_file,input_file+".density")
    data = m2m.read_dehnen_density_file(input_file+".density")
    data['mass']=data['m']
    call(['rm',input_file+".density"])
    ## Density calculation
    data.sort_values(by='rho',ascending=False,inplace=True)
    if len(data)*1./nbins<at_least_n_per_bin:
        nbins = int(len(data)*1./at_least_n_per_bin)
    dens = np.zeros(nbins)
    ra = np.zeros(nbins)
    ba = np.zeros(nbins)
    ca = np.zeros(nbins)
    align = np.zeros(nbins)
    pmaj = np.zeros((nbins,3))
    pint = np.zeros_like(pmaj)
    pmin = np.zeros_like(pmaj)
    for i, datause in enumerate(np.array_split(data, nbins)):
      ba[i],ca[i],pmaj[i],pint[i],pmin[i]=moment_of_inertia_diagonalization(datause,
                                                            # no offset
                                                            0.,0.,0.,
                                                            reduced=False)
      align[i]=np.abs(np.dot(pmaj[i],centre[:3])/np.sqrt(np.sum(centre[:3]**2)))
      dens[i] = np.median(datause['rho'])
      ra[i] = np.median(np.sqrt(datause.x**2+datause.y**2+datause.z**2))
    return dens, ra, ba, ca, align, pmaj, pint, pmin


def compare_1d_profiles(prop,output,times=[0.,1.]):
    plt.clf()
    data_1,data_2=grab_snapshots(prop,times)
    radii1 = radius(data_1,prop['ba'],prop['ca'])
    radii2 = radius(data_2,prop['ba'],prop['ca'])
    minr = np.percentile(radii1, 0.5)
    maxr = np.percentile(radii1, 99.5)
    b = np.logspace(np.log10(minr),np.log10(maxr),100)
    binsize=b[1]-b[0]
    const_weight = 4.*np.pi*np.log(10)*binsize
    weights1 = data_1.T[0]/radii1**3/const_weight
    weights2 = data_2.T[0]/radii2**3/const_weight
    n,b,p=plt.hist(radii1,bins=b,histtype='step',lw=2,weights=weights1)
    n,b,p=plt.hist(radii2,bins=b,histtype='step',lw=2,weights=weights2)
    plt.semilogy()
    plt.semilogx()
    plt.xlabel(r'$r$')
    plt.ylabel(r'$\rho$')
    plt.savefig(output,bbox_inches='tight')

def compare_1d_profiles_2(prop,output,times=[0.,1.],
                          annotation='',vline=None, vline_annotation=''):
    plt.clf()
    data_1,data_2=grab_snapshots(prop,times)
    radii1 = radius(data_1,prop['ba'],prop['ca'])
    radii2 = radius(data_2,prop['ba'],prop['ca'])
    minr = np.percentile(radii1, 0.5)
    maxr = np.percentile(radii1, 99.5)
    b = np.logspace(np.log10(minr),np.log10(maxr),50)
    binsize=b[1]-b[0]
    const_weight = 4.*np.pi*np.log(10)*binsize
    weights1 = data_1.T[0]/radii1**3/const_weight
    weights2 = data_2.T[0]/radii2**3/const_weight
    ## Select DM

    cut = data_2.T[-1]==1.
    ndm,b,p=plt.hist(radii2[cut],bins=b,histtype='step',lw=1.5,
                   weights=weights2[cut],
                   color='k')
    nstars,b,p=plt.hist(radii2[~cut],bins=b,histtype='step',lw=1.5,
                   weights=weights2[~cut],ls='dashed',
                   color='k')
    plt.clf()
    plt.plot(.5*(b[1:]+b[:-1]),ndm,'.-',color='k',
             label=r'$t=13.7\,\mathrm{Gyr}$')
    plt.plot(.5*(b[1:]+b[:-1]),nstars,'.-',color='k')

    cut = data_1.T[-1]==1.
    n,b,p=plt.hist(radii1[cut],bins=b,histtype='step',lw=1.5,
                   weights=weights1[cut],
                   color='k', label=r'$t=0$')
    n,b,p=plt.hist(radii1[~cut],bins=b,histtype='step',lw=1.5,
                   weights=weights1[~cut],
                   color='k')
    plt.semilogy()
    plt.semilogx()
    plt.ylim(np.min(ndm),2.*np.max(ndm))
    plt.xlabel(r'$r/\mathrm{kpc}$')
    plt.ylabel(r'$\rho/M_\odot\,\mathrm{kpc}^{-3}$')
    plt.annotate(annotation, xy=(0.0,0.1), xycoords='axes fraction',
                 fontsize=10)
    if vline:
      plt.axvline(vline, color='gray')
      plt.annotate(vline_annotation, xy=(vline*1.2, np.min(ndm)*10.),
                   xycoords='data',
                   fontsize=12)
    plt.legend()
    plt.savefig(output,bbox_inches='tight')

def compare_1d_sigmar_profiles_2(prop,output,times=[0.,1.],sigmayaxis=12.):
    plt.clf()
    data_1,data_2=grab_snapshots(prop,times)
    radii1 = radius(data_1,prop['ba'],prop['ca'])
    radii2 = radius(data_2,prop['ba'],prop['ca'])
    radialvel1 = radvel(data_1)
    radialvel2 = radvel(data_2)
    minr = np.percentile(radii1, 0.5)
    maxr = np.percentile(radii1, 99.5)
    b = np.logspace(np.log10(minr),np.log10(maxr),20)
    weights1 = data_1.T[0]/radii1**3
    weights2 = data_2.T[0]/radii2**3
    ## Select DM
    cut = data_1.T[-1]==1.
    n,b=np.histogram(radii1[cut],bins=b,
                   weights=weights1[cut])
    na,b=np.histogram(radii1[~cut],bins=b,
                   weights=weights1[~cut])
    cut = data_2.T[-1]==1.
    n2,b=np.histogram(radii2[cut],bins=b,
                   weights=weights2[cut])
    na2,b=np.histogram(radii2[~cut],bins=b,
                   weights=weights2[~cut])
    weights1*=radialvel1*radialvel1
    weights2*=radialvel2*radialvel2
    cut = data_1.T[-1]==1.
    nv,b=np.histogram(radii1[cut],bins=b,
                   weights=weights1[cut])
    nva,b=np.histogram(radii1[~cut],bins=b,
                   weights=weights1[~cut])
    cut = data_2.T[-1]==1.
    nv2,b=np.histogram(radii2[cut],bins=b,
                   weights=weights2[cut])
    nva2,b=np.histogram(radii2[~cut],bins=b,
                   weights=weights2[~cut])
    bc = .5*(b[1:]+b[:-1])
    plt.plot(bc,np.sqrt(nv/n),'.-',ms=5,color=sns.color_palette()[0])
    plt.plot(bc,np.sqrt(nva/na),'.-',ms=5,color=sns.color_palette()[0],ls='dashed')
    plt.plot(bc,np.sqrt(nv2/n2),'.-',ms=5,color=sns.color_palette()[1])
    plt.plot(bc,np.sqrt(nva2/na2),'.-',ms=5,color=sns.color_palette()[1],ls='dashed')
    plt.semilogy()
    plt.semilogx()
    # plt.plot(b,np.power(b,0.1)*5.,color='k')
    plt.xlabel(r'$r/\mathrm{kpc}$')
    plt.ylabel(r'$\sigma/\mathrm{km\,s}^{-1}$')
    plt.ylim(0.1,sigmayaxis)
    plt.savefig(output,bbox_inches='tight')

def shape_evolution_plot(output, ba_dm, ca_dm, align_dm, ba_st, ca_st, align_st, times):
  f,a=plt.subplots(2,1,figsize=[6.,4.])
  plt.subplots_adjust(hspace=0.)
  plt.sca(a[0])
  plt.plot(times,ba_dm,'.-',ms=5,color=sns.color_palette()[0])
  plt.plot(times,ca_dm,'.-',ms=5,color=sns.color_palette()[0],ls='dashed')
  plt.plot(times,ba_st,'.-',ms=5,color=sns.color_palette()[1])
  plt.plot(times,ca_st,'.-',ms=5,color=sns.color_palette()[1],ls='dashed')
  # plt.xlabel(r'$t/\mathrm{Gyr}$')
  plt.ylabel(r'Shapes: $(b/a)$, $(c/a)$')
  plt.sca(a[1])
  plt.plot(times,align_dm,'.-',ms=5,color=sns.color_palette()[0])
  plt.plot(times,align_st,'.-',ms=5,color=sns.color_palette()[1])
  plt.xlabel(r'$t/\mathrm{Gyr}$')
  plt.ylabel(r'$\cos\theta$')
  plt.savefig(output,bbox_inches='tight')

def shape_evolution(prop,output,max_time=None):
    plt.clf()
    times = nbody_tools.get_times(prop['simFileGRAVOUTPUT'])
    if(max_time):
      times=times[times<max_time]
    ba_dm=np.ones(len(times))
    ca_dm=np.ones_like(ba_dm)
    align_dm=np.ones_like(ba_dm)
    ba_st=np.ones(len(times))
    ca_st=np.ones_like(ba_st)
    align_st=np.ones_like(ba_st)
    for i,t in enumerate(times):
      data,centre = grab_snapshot(prop,t,with_centre=True)
      if not len(data):
        continue
      data = data_to_pandas(data)
      ba_st[i],ca_st[i],pr=moment_of_inertia_diagonalization(data[data.dm==0.],
                                                            # no offset
                                                            0.,0.,0.,
                                                            reduced=True)[:3]
      align_st[i]=np.abs(np.dot(pr,centre[:3])/np.sqrt(np.sum(centre[:3]**2)))
      ba_dm[i],ca_dm[i],pr=moment_of_inertia_diagonalization(data[data.dm==1.],
                                                            # no offset
                                                            0.,0.,0.,
                                                            reduced=True)[:3]
      align_dm[i]=np.abs(np.dot(pr,centre[:3])/np.sqrt(np.sum(centre[:3]**2)))
    # times/=nbody_tools.kms2kpcGyr
    return shape_evolution_plot(output, ba_dm, ca_dm, align_dm, ba_st, ca_st, align_st, times)

def shape_profile(prop,time,nbins=20):
    ''' in spherical bins so gives wrong results =- use Walter density '''
    data,centre = grab_snapshot(prop,time,with_centre=True)
    data = data_to_pandas(data)
    radii = np.sqrt(data.x**2+data.y**2+data.z**2)
    ba_dm = np.zeros(nbins)
    ca_dm, align_dm, ba_st, ca_st, align_st = np.zeros_like(ba_dm),  np.zeros_like(ba_dm),  np.zeros_like(ba_dm),  np.zeros_like(ba_dm),  np.zeros_like(ba_dm)

    radii_st_lims = np.percentile(radii[data.dm==0.],[5.,99.5])
    radii_st_bins = np.logspace(np.log10(radii_st_lims[0]),np.log10(radii_st_lims[1]),nbins)
    radii_dm_lims = np.percentile(radii[data.dm==1.],[5.,99.5])
    radii_dm_bins = np.logspace(np.log10(radii_dm_lims[0]),np.log10(radii_dm_lims[1]),nbins)

    for n in range(nbins):
      ba_st[n],ca_st[n],pr=moment_of_inertia_diagonalization(data[(radii<radii_st_bins[n])&(data.dm==0.)],
                                                            # no offset
                                                            0.,0.,0.,
                                                            reduced=True)[:3]
      align_st[n]=np.abs(np.dot(pr,centre[:3])/np.sqrt(np.sum(centre[:3]**2)))
      ba_dm[n],ca_dm[n],pr=moment_of_inertia_diagonalization(data[(radii<radii_dm_bins[n])&(data.dm==1.)],
                                                            # no offset
                                                            0.,0.,0.,
                                                            reduced=True)[:3]
      align_dm[n]=np.abs(np.dot(pr,centre[:3])/np.sqrt(np.sum(centre[:3]**2)))

    return radii_st_bins, ba_st, ca_st, align_st, radii_dm_bins, ba_dm, ca_dm, align_dm

def shape_profile_dehnen(prop,time):
    data,centre = grab_snapshot(prop, time, with_centre=True)
    data = data_to_pandas(data)
    data_dm = data[data.dm==1.]
    data_st = data[data.dm==0.]
    if len(data_st)>32:
        m2m.write_nparray_to_snapshot(data_st.values.T[:7].T,prop['simFile']+'_denstmp')
        dens_st, radii_st, ba_st, ca_st, align_st, pmaj_st, pint_st, pmin_st = \
		measure_density_profile(prop['simFile']+'_denstmp',centre)
        call(['rm',prop['simFile']+'_denstmp'])
    else:
        dens_st, radii_st, ba_st, ca_st, align_st, pmaj_st, pint_st, pmin_st = \
		np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1), \
		np.zeros((1,3)), np.zeros((1,3)), np.zeros((1,3))
    if len(data_dm)>32:
        m2m.write_nparray_to_snapshot(data_dm.values.T[:7].T,prop['simFile']+'_denstmp')
        dens_dm, radii_dm, ba_dm, ca_dm, align_dm, pmaj_dm, pint_dm, pmin_dm = \
		measure_density_profile(prop['simFile']+'_denstmp',centre)
        call(['rm',prop['simFile']+'_denstmp'])
    else:
        dens_dm, radii_dm, ba_dm, ca_dm, align_dm, pmaj_dm, pint_dm, pmin_dm = \
		np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1),\
		np.zeros((1,3)), np.zeros((1,3)), np.zeros((1,3))
    return radii_st, ba_st, ca_st, align_st, dens_st, pmaj_st, pint_st, pmin_st, \
	   radii_dm, ba_dm, ca_dm, align_dm, dens_dm, pmaj_dm, pint_dm, pmin_dm

def all_evolution(prop,max_time=None,prune_radius=None):
    plt.clf()
    times = nbody_tools.get_times(prop['simFileGRAVOUTPUT'])
    if(max_time):
      times=times[times<max_time]

    mass_dm=np.ones(len(times))
    mass_st=np.ones(len(times))
    mass_dm_core=np.ones(len(times))
    mass_st_core=np.ones(len(times))
    mass_dm_core_init=np.ones(len(times))
    mass_st_core_init=np.ones(len(times))
    mass_dm_core_tilde=np.ones(len(times))
    mass_st_core_tilde=np.ones(len(times))
    ba_dm=np.ones(len(times))
    ca_dm=np.ones_like(ba_dm)
    align_dm=np.ones_like(ba_dm)
    ba_st=np.ones(len(times))
    ca_st=np.ones_like(ba_st)
    align_st=np.ones_like(ba_st)
    r_half=np.zeros_like(ba_st)
    eps=np.ones_like(ba_st)
    r_core=np.zeros_like(ba_st)
    eps_king=np.ones_like(ba_st)
    c_king=np.ones_like(ba_st)
    sig_maj=np.zeros_like(ba_st)
    sig_int=np.zeros_like(ba_st)
    sig_min=np.zeros_like(ba_st)
    sig_proj=np.zeros_like(ba_st)
    sig_proj_flat=np.zeros_like(ba_st)
    centre_x=np.zeros_like(ba_st)
    centre_y=np.zeros_like(ba_st)
    centre_z=np.zeros_like(ba_st)
    centre_vx=np.zeros_like(ba_st)
    centre_vy=np.zeros_like(ba_st)
    centre_vz=np.zeros_like(ba_st)
    eq_centre_s=np.zeros_like(ba_st)
    eq_centre_vlos=np.zeros_like(ba_st)

    for i,t in enumerate(times):
      data,centre = grab_snapshot(prop,t,with_centre=True)
      if not len(data):
          continue
      data = data_to_pandas(data)

      ba_dm[i],ca_dm[i],pmaj,pint,pmin=moment_of_inertia_diagonalization(data[data.dm==1.],
                                                            # no offset
                                                            0.,0.,0.,
                                                            reduced=True)
      align_dm[i]=np.abs(np.dot(pmaj,centre[:3])/np.sqrt(np.sum(centre[:3]**2)))

      ba_st[i],ca_st[i],pmaj,pint,pmin=moment_of_inertia_diagonalization(data[data.dm==0.],
                                                            # no offset
                                                            0.,0.,0.,
                                                            reduced=True)
      align_st[i]=np.abs(np.dot(pmaj,centre[:3])/np.sqrt(np.sum(centre[:3]**2)))
      xtil = np.sum(pmaj*data.values.T[1:4].T,axis=1)
      ytil = np.sum(pint*data.values.T[1:4].T,axis=1)
      ztil = np.sum(pmin*data.values.T[1:4].T,axis=1)
      if np.count_nonzero(data.dm==0.):
            vmaj = np.sum(pmaj*data[data.dm==0.].values.T[4:7].T,axis=1)
            vint = np.sum(pint*data[data.dm==0.].values.T[4:7].T,axis=1)
            vmin = np.sum(pmin*data[data.dm==0.].values.T[4:7].T,axis=1)
            vmaj-= np.sum(data.mass[data.dm==0.]*vmaj)/np.sum(data.mass[data.dm==0.])
            vint-= np.sum(data.mass[data.dm==0.]*vint)/np.sum(data.mass[data.dm==0.])
            vmin-= np.sum(data.mass[data.dm==0.]*vmin)/np.sum(data.mass[data.dm==0.])

            sig_maj[i] = np.sqrt(np.sum(data.mass[data.dm==0.]*vmaj**2)/np.sum(data.mass[data.dm==0.]))
            sig_int[i] = np.sqrt(np.sum(data.mass[data.dm==0.]*vint**2)/np.sum(data.mass[data.dm==0.]))
            sig_min[i] = np.sqrt(np.sum(data.mass[data.dm==0.]*vmin**2)/np.sum(data.mass[data.dm==0.]))

      mass_dm[i]=np.sum(data.mass[data.dm==1.])
      mass_st[i]=np.sum(data.mass[data.dm==0.])

      radii = np.sqrt(data.x**2+data.y**2+data.z**2)
      radii_tilde = np.sqrt(xtil**2+(ytil/ba_st[i])**2+(ztil/ca_st[i])**2)

      mass_dm_core_init[i]=np.sum(data.mass[(data.dm==1.)&(radii<prop['rs'])])
      mass_st_core_init[i]=np.sum(data.mass[(data.dm==0.)&(radii<prop['rs'])])

      # data,centre = grab_snapshot(prop,t,with_centre=True,recentre_sim=False)
      # data = data_to_pandas(data)
      data_recentre = recentre(data.values[:,:7],centre=-np.copy(centre))
      for ii,nn in enumerate(['x','y','z','vx','vy','vz']):
        data[nn]=data_recentre[:,ii+1]

      if np.count_nonzero(data.dm==0.):
          results, xtmp, ytmp = nbody_tools.fit_surface_density(data[data.dm==0.].values,centre=np.copy(centre))
          r_half[i]=results[0]
          eps[i]=results[1]
          results, xtmp, ytmp = nbody_tools.fit_surface_density(data[data.dm==0.].values,centre=np.copy(centre),profile="King")
          r_core[i]=results[0]
          eps_king[i]=results[1]
          c_king[i]=results[-1]
          print '#######################'
          print 'King fit ##############'
          print r_core[i], c_king[i]
          print '#######################'
      mass_dm_core[i]=np.sum(data.mass[(data.dm==1.)&(radii<r_half[i])])
      mass_st_core[i]=np.sum(data.mass[(data.dm==0.)&(radii<r_half[i])])
      mass_dm_core_tilde[i]=np.sum(data.mass[(data.dm==1.)&(radii_tilde<r_half[i])])
      mass_st_core_tilde[i]=np.sum(data.mass[(data.dm==0.)&(radii_tilde<r_half[i])])

      ## Solar velocity doesn't matter here
      eq = np.array([aa_py.CartesianToGalactic(ii[1:7],
                                               np.array([8.2,0.,0.,
                                                        11.1,240.,7.25]))
                    for ii in data[data.dm==0.].values])
      eq_centre = aa_py.CartesianToGalactic(centre[:6],
                                               np.array([8.2,0.,0.,
                                                        11.1,240.,7.25]))

      from scipy.optimize import minimize
      from scipy.misc import logsumexp
      def logl(p,x):
        if p[1]<0. or p[3]<0. or p[-1]<0. or p[-1]>1.:
            return np.inf
        return -np.sum(np.logaddexp(-(x-p[0])**2/2./p[1]**2-.5*np.log(p[1]**2)+np.log(1.-p[-1]),(-(x-p[2])**2/2./p[3]**2-.5*np.log(p[3]**2)+np.log(p[-1]))))

      if np.count_nonzero(data.dm==0.):
          ## Changed this to see if it makes a difference
          vproj=eq.T[3]-eq_centre[3]
          vproj-=np.sum(data.mass[data.dm==0.]*vproj)/np.sum(data.mass[data.dm==0.])
          Rvec = centre[:3]-np.array([8.2,0.,0.])
          Rvec /= np.sqrt(np.sum(Rvec**2))
          vprojR=data.vx*Rvec[0]+data.vy*Rvec[1]+data.vz*Rvec[2]
          vprojR = vprojR[data.dm==0]
          vprojR-=np.sum(data.mass[data.dm==0.]*vprojR)/np.sum(data.mass[data.dm==0.])
          sig_proj[i] = minimize(logl,[0.05,sig_maj[i],0.05,2.*sig_maj[i],0.2],args=(vproj,)).x[1]
          sig_proj_flat[i] = minimize(logl,[0.05,sig_maj[i],0.05,2.*sig_maj[i],0.2],args=(vprojR,)).x[1]
          # sig_proj[i]=np.sqrt(np.sum(data.mass[data.dm==0.]*vproj**2)/np.sum(data.mass[data.dm==0.]))
          # print i, sig_maj[i], sig_min[i], sig_proj[i], centre, eq_centre, np.count_nonzero(data.dm==0.)

      print '#####################'
      print t, r_half[i], sig_proj[i]
      print '#####################'

      centre_x[i]=centre[0]
      centre_y[i]=centre[1]
      centre_z[i]=centre[2]
      centre_vx[i]=centre[3]
      centre_vy[i]=centre[4]
      centre_vz[i]=centre[5]
      eq_centre_s[i]=eq_centre[2]
      eq_centre_vlos[i]=eq_centre[3]
    # times/=nbody_tools.kms2kpcGyr

    names = ['times','mass_dm','mass_st','ba_dm','ca_dm','align_dm','ba_st','ca_st','align_st','r_half','eps', 'r_core', 'eps_king', 'c_king', 'sig_maj','sig_int','sig_min','sig_proj','sig_proj_flat','mass_dm_core','mass_st_core','mass_dm_core_tilde','mass_st_core_tilde','mass_dm_core_init','mass_st_core_init',
        'x','y','z','vx','vy','vz','s','vlos']
    outputs = np.vstack((times, mass_dm, mass_st, ba_dm, ca_dm, align_dm, ba_st, ca_st, align_st, r_half, eps, r_core, eps_king, c_king, sig_maj, sig_int, sig_min, sig_proj, sig_proj_flat, mass_dm_core, mass_st_core, mass_dm_core_tilde, mass_st_core_tilde, mass_dm_core_init, mass_st_core_init,centre_x,centre_y,centre_z,
        centre_vx,centre_vy,centre_vz,eq_centre_s,eq_centre_vlos))

    return pd.DataFrame(outputs.T, columns=names)


def bound_mass_evolution(prop, output, max_time=None):
    plt.clf()
    times = nbody_tools.get_times(prop['simFileGRAVOUTPUT'])
    if(max_time):
      times=times[times<max_time]
    mass_dm=np.ones(len(times))
    mass_st=np.ones(len(times))
    for i,t in enumerate(times):
        data,centre = grab_snapshot(prop,t,with_centre=True)
        data = data_to_pandas(data)
        mass_dm[i]=np.sum(data.mass[data.dm==1.])
        mass_st[i]=np.sum(data.mass[data.dm==0.])
    plt.plot(times, mass_dm/mass_dm[0], ls='dotted', label='DM')
    plt.plot(times, mass_st/mass_st[0], ls='dashed', label='Stars')
    plt.plot(times, (mass_st+mass_dm)/(mass_dm[0]+mass_st[0]),
             color='k', label='Total')
    plt.xlabel(r'$t/\mathrm{Gyr}$')
    plt.ylabel(r'$M/M_\mathrm{init}$')
    plt.legend()
    plt.semilogy()
    plt.savefig(output,bbox_inches='tight')

def compare_2d_profiles(prop,output,times=[0.,1.]):
    data_1,data_2=grab_snapshots(prop,times)
    radii1 = radius(data_1,prop['ba'],prop['ca'])
    radii2 = radius(data_2,prop['ba'],prop['ca'])
    maxr = np.percentile(radii1,[90.])[0]
    b = np.linspace(-maxr,maxr,50)
    f,a=plt.subplots(1,2,figsize=[10.,4.])
    plt.sca(a[0])
    for i,D in enumerate([data_1,data_2]):
      contour2D(D.T[1],D.T[2],
                  plt.gca(),
                  with_contours=True,with_imshow=False,bins=b,
                  weights=D.T[0],COLA=[sns.color_palette()[i]])

    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.gca().set_aspect('equal')
    plt.sca(a[1])
    for i,D in enumerate([data_1,data_2]):
      contour2D(D.T[1],D.T[3],
                  plt.gca(),
                  with_contours=True,with_imshow=False,bins=b,
                  weights=D.T[0],COLA=[sns.color_palette()[i]])
    plt.xlabel(r'$x$')
    plt.ylabel(r'$z$')
    plt.gca().set_aspect('equal')
    plt.savefig(output,bbox_inches='tight')

def compare_2d_profiles_2(prop,output,times=[0.,1.]):
    data_1,data_2=grab_snapshots(prop,times)
    radii1 = radius(data_1,prop['ba'],prop['ca'])
    radii2 = radius(data_2,prop['ba'],prop['ca'])
    maxr = np.percentile(radii1,[90.])[0]
    b = np.linspace(-maxr,maxr,50)
    f,a=plt.subplots(2,2,figsize=[4.,4.])
    plt.sca(a[0][0])
    cut = data_1.T[-1]==1.
    for i,D in enumerate([data_1[cut],data_2[cut]]):
      contour2D(D.T[1],D.T[2],
                  plt.gca(),
                  with_contours=True,with_imshow=False,bins=b,
                  weights=D.T[0],COLA=[sns.color_palette()[i]])

    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.gca().set_aspect('equal')
    plt.sca(a[0][1])
    for i,D in enumerate([data_1[cut],data_2[cut]]):
      contour2D(D.T[1],D.T[3],
                  plt.gca(),
                  with_contours=True,with_imshow=False,bins=b,
                  weights=D.T[0],COLA=[sns.color_palette()[i]])
    plt.xlabel(r'$x$')
    plt.ylabel(r'$z$')
    plt.gca().set_aspect('equal')

    plt.sca(a[1][0])
    for i,D in enumerate([data_1[~cut],data_2[~cut]]):
      contour2D(D.T[1],D.T[2],
                  plt.gca(),
                  with_contours=True,with_imshow=False,bins=b,
                  weights=D.T[0],COLA=[sns.color_palette()[i]])

    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.gca().set_aspect('equal')
    plt.sca(a[1][1])
    for i,D in enumerate([data_1[~cut],data_2[~cut]]):
      contour2D(D.T[1],D.T[3],
                  plt.gca(),
                  with_contours=True,with_imshow=False,bins=b,
                  weights=D.T[0],COLA=[sns.color_palette()[i]])
    plt.xlabel(r'$x$')
    plt.ylabel(r'$z$')
    plt.gca().set_aspect('equal')
    plt.savefig(output,bbox_inches='tight')
