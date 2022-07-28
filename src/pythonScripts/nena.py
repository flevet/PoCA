import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def NeNA(NeNA_dist):
    dir_name = 'e:'
    NeNA_acc, NeNA_err=plot_NeNA(NeNA_dist,dir_name)
    hd="the average lecalization accuracy by NeNA is at %.1f [nm]" %(float(NeNA_acc[0]))
    outname=dir_name + '/NeNA_lac.txt'
    np.savetxt(outname, NeNA_dist, fmt='%.5e', delimiter='   ',header=hd,comments='# ')
    coeffs = np.array([float(NeNA_acc[0])],dtype=np.float64)
    return coeffs

def CFunc2dCorr(r,a,rc,w,F,A,O):
    y=(r/(2*a*a))*np.exp((-1)*r*r/(4*a*a))*A+(F/(w*np.sqrt(np.pi/2)))*np.exp(-2*((r-rc)/w)*((r-rc)/w))+O*r
    return y

def Area(r,y):
    Areaf=abs(np.trapz(y, r))
    return Areaf

def CFit_resultsCorr(r,y):
    A=Area(r,y)
    p0 = np.array([10.0,15,100,(A/2),(A/2),((y[98]/200))])
    popt, pcov = curve_fit(CFunc2dCorr,r,y,p0)
    return popt, pcov

def plot_NeNA(NeNA_dist,dir_name):
    Min=0
    Max=150
    Int=1
    Inc=int((Max-Min)/Int)
    x=np.arange(Min,Max,Int,dtype='float')
    y=np.histogram(NeNA_dist, bins=Inc, range=(Min,Max), density=True)[0]
    acc, acc_err=CFit_resultsCorr(x,y)
##    NeNA_func=CFunc2dCorr(x,acc[0],acc[1],acc[2],acc[3],acc[4],acc[5])
##    name=dir_name+'/NeNA_lac.pdf'
##    f, axarr = plt.subplots(1, sharex=False)
##    axarr.bar(x, y, color='gray', edgecolor='black',width=Int)
##    axarr.plot(x,NeNA_func, 'b')
##    axarr.set_xlim([Min,Max])
##    axarr.set_xlabel('loc_acc [nm]')
##    axarr.set_ylabel('Intensity [a.u.]')
##    plt.savefig(name, format='pdf')
    return acc, acc_err

##filename="e:/nearestDistances.txt"
##data = np.loadtxt(filename, delimiter="\n")
##NeNA(data)
