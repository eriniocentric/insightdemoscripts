#!/usr/bin/python

import sys
import os
import os.path
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd

from astropy.io import fits
from astropy.table import Table, Column, MaskedColumn, hstack
from astropy.io import ascii
from astropy.nddata import NDData

rasep=4
decsep=4
wavesep=4

cat=ascii.read('/work/05178/cxliu/maverick/detect/test/catalog/cat.cx3',names=['ID','RA','DEC','wave','sn','chi2','amplitude','sigma','continuum','ncat0','nalines','ngood','fwhm'])
#cat['shot']=cat['ID'][12]

#sort by decreasing S/N to start on highest S/N obs
#cat[::-1].sort('sn')

ncat=np.size(cat)
cat['multi_idx']=-1*np.ones(ncat)
cat['nobs']=-1*np.ones(ncat)

output=list()
detectlist=list()
#for loop to group objects according to separations defined above

groupcount=1

for count in range(0,ncat):
    if cat['multi_idx'][count] < 0:
        sel=np.where((abs(cat['RA'][count]-cat['RA'])<(rasep/3600.)) & (abs(cat['DEC'][count]-cat['DEC'])<(decsep/3600.)) & (abs(cat['wave'][count]-cat['wave'])<wavesep)) 
        cat['multi_idx'][sel]=groupcount
        cat['nobs'][sel]=np.size(sel)
        
        output.append([groupcount,np.median(cat['RA'][sel]),np.median(cat['DEC'][sel]),np.median(cat['wave'][sel]),np.median(cat['sn'][sel]),np.std(cat['sn'][sel]),np.max(cat['nalines'][sel]),np.max(cat['ncat0'][sel]),np.max(cat['ngood'][sel])])
#        if np.size(sel) > 3: detectlist.append([groupcount,cat['ID'][sel]])
        groupcount+=1


#save output table 
ascii.write(cat,'cosdeep_group_info.dat',overwrite=True)

sel2=np.where( (cat['ncat0']>1) )
nuniq=np.size(np.unique(cat['multi_idx'][sel2]))

print "number of unique objects found in 2 or more shots is", nuniq

sel2=np.where( (cat['ncat0']>2) )
nuniq=np.size(np.unique(cat['multi_idx'][sel2]))

print "number of unique objects found in 3 or more shots is", nuniq


outputarr = Table(rows=output, names=('multi_idx','ra','dec','wave','sn','snrms','nalines','nobs','ngood'),dtype=('i4','f8','f8','f8','f8','f8','i4','i4','i4') )

ascii.write(outputarr,'groups.dat',overwrite=True)

#first plot completeness

nbin=15
snbin=3
snarray=np.arange(nbin)*snbin
compl=np.zeros(nbin)
compl_std=np.zeros(nbin)

compl_2frame=np.zeros(nbin)
compl_3frame=np.zeros(nbin)
compl_5frame=np.zeros(nbin)
compl_2frame_std=np.zeros(nbin)
compl_3frame_std=np.zeros(nbin)
compl_5frame_std=np.zeros(nbin)


compl_lfwhm=np.zeros(nbin)
compl_hfwhm=np.zeros(nbin)

compl_alines=np.zeros(nbin)

for count in xrange(nbin-1):
    sel=np.where( (outputarr['sn']>snarray[count]) & (outputarr['sn']<snarray[count+1]) & (outputarr['ngood']>0)) 
    if np.size(sel) > 0: 
        compl[count]=float(np.sum(outputarr['nobs'][sel]))/float(np.sum(outputarr['ngood'][sel]))
        compl_std[count]=np.std(np.divide(outputarr['nobs'][sel],outputarr['ngood'][sel],dtype=float))

    sel2=np.where( (outputarr['sn']>snarray[count]) & (outputarr['sn']<snarray[count+1]) & (outputarr['ngood']>0) & (outputarr['nobs']>1))
    if np.size(sel2) > 0:
        compl_2frame[count]=float(np.sum(outputarr['nobs'][sel2]))/float(np.sum(outputarr['ngood'][sel2]))
        compl_2frame_std[count]=np.std(np.divide(outputarr['nobs'][sel2],outputarr['ngood'][sel2],dtype=float))

    sel3=np.where( (outputarr['sn']>snarray[count]) & (outputarr['sn']<snarray[count+1]) & (outputarr['ngood']>0) & (outputarr['nobs']>2))
    if np.size(sel3) > 0:
        compl_3frame[count]=float(np.sum(outputarr['nobs'][sel3]))/float(np.sum(outputarr['ngood'][sel3]))
        compl_3frame_std[count]=np.std(np.divide(outputarr['nobs'][sel3],outputarr['ngood'][sel3],dtype=float))

    sel5=np.where( (outputarr['sn']>snarray[count]) & (outputarr['sn']<snarray[count+1]) & (outputarr['ngood']>0) & (outputarr['nobs']>4))
    if np.size(sel5) > 0:
        compl_5frame[count]=float(np.sum(outputarr['nobs'][sel5]))/float(np.sum(outputarr['ngood'][sel5]))
        compl_5frame_std[count]=np.std(np.divide(outputarr['nobs'][sel5],outputarr['ngood'][sel5],dtype=float))


    sellfwhm=np.where( (cat['fwhm']<2.01) & (cat['ncat0']>2) & (cat['sn']>snarray[count]) & (cat['sn']<snarray[count+1]))  
    if np.size(sellfwhm) > 0:
        compl_lfwhm[count]=float(np.sum(cat['ncat0'][sellfwhm]))/float(np.sum(cat['ngood'][sellfwhm]))
    
    selhfwhm=np.where( (cat['fwhm']>2.01) & (cat['ncat0']>2) & (cat['sn']>snarray[count]) & (cat['sn']<snarray[count+1]))
    if np.size(selhfwhm) > 0:
        compl_hfwhm[count]=float(np.sum(cat['ncat0'][selhfwhm]))/float(np.sum(cat['ngood'][selhfwhm]))

    
snarray=snarray+1

plt.figure()
sel=np.where(compl>0.1)
plt.errorbar(snarray[sel]+snbin/2,compl[sel],xerr=snbin/2,yerr=compl_std[sel],label='all cat0 objects',color='red',linestyle='--',ecolor='salmon')
sel2=np.where(compl_2frame>0.1)
plt.errorbar(snarray[sel2]+snbin/2,compl_2frame[sel2],xerr=snbin/2,yerr=compl_2frame_std[sel2],label='Objects detected in 2 or more frames',color='green',linestyle='--',ecolor='lightgreen')
sel3=np.where(compl_3frame>0.1)
plt.errorbar(snarray[sel3]+snbin/2,compl_3frame[sel3],xerr=snbin/2,yerr=compl_3frame_std[sel3],label='Objects detected in 3 or more frames',color='orange',linestyle='--',ecolor='navajowhite')
sel5=np.where(compl_5frame>0.1)
plt.errorbar(snarray[sel5]+snbin/2,compl_5frame[sel5],xerr=snbin/2,yerr=compl_5frame_std[sel5],label='Objects detected in 5 or more frames',color='blue',linestyle='--',ecolor='cornflowerblue')
plt.xlabel('Median S/N')
plt.ylabel('N_detected/N_observed')
plt.legend()
plt.savefig("completeness_detects.png")
plt.close()




plt.figure()
sel=np.where(compl_lfwhm>0.01)
plt.plot(snarray[sel],compl_lfwhm[sel],label='3 Frames & FWHM < 2')
sel2=np.where(compl_hfwhm>0.01)
plt.plot(snarray[sel2],compl_hfwhm[sel2],label='3 Frames & FWHM > 2')
plt.xlabel('S/N')
plt.ylabel('N_detected/N_observed')
plt.legend()
plt.savefig("completeness_fwhmbins.png")
plt.close()


plt.figure()
plt.hist(cat['sn'],label='All COSDEEP detects catalogs')
sel=np.where( (outputarr['nobs']>1) )
plt.hist(cat['sn'][sel],label='Objects in 2 or more frames')
sel=np.where( (outputarr['nobs']>4) )
plt.hist(cat['sn'][sel],label='Ojbects in 5 or more frames')
plt.yscale('log')
plt.ylabel('N Detections')
plt.xlabel('S/N')
plt.legend()
plt.savefig("hist_SN.png")
plt.close()

plt.close('all')
