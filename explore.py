#!/usr/bin/python

import sys
import os
import os.path
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from astropy.io import fits
from astropy.table import Table, Column, MaskedColumn, hstack
from astropy.io import ascii
from astropy.nddata import NDData

#read in input catalog... this is a list of known line emitters in the cosmos repeat field
input=ascii.read('inputcat')

#read in output catalog from extractions of each emitter for each cosrepeat shot
datarsp=ascii.read('fullcat')

#read in output catalog from all the detect catalog matches for each known emitter
datadetect=ascii.read('outputall_no_v')


#read in lookup tables for IFU location of each emitter
#objectifu=ascii.read('ifulist')
#read in number of times the IFU was active out of 11 cosmos repeat shots
#ifunshot=ascii.read('ifunshot')

objectngood=ascii.read('object_ngood')

cosshot=ascii.read('COSshot')
dates=cosshot['col1']
ndate=np.size(dates)

objects=input['col1']

nobj=np.size(input)

#create structured array to store output info for 
rsp_stats=np.zeros(nobj, dtype={'names':('OBJECTID','snmedian','snrms','fraction','ndet','ngood'),'formats':('S14','f8','f8','f8','i4','i4')})

#create structured array to store output for detect catalogs
detect_stats=np.zeros(nobj, dtype={'names':('OBJECTID','snmedian','snrms','fraction','ndet','ngood'),'formats':('S14','f8','f8','f8','i4','i4')})

#create structured array to store flags for each object/shot combo
output=np.zeros(nobj*ndate, dtype={'names':('OBJECTID','shot','rsp_sngt5_flag','rsp_sngt8_flag','detectcat_no',
'detectcatflag','bothflag','rsp_sngt12_flag'),'formats':('S14','S12','i4','i4','i4','i4','i4','i4')})

counter=0
counter2=0
objcount=0

for object in objects:
    for date in dates:
        
        sel2=np.where( (datarsp['ID']==object) & (datarsp['SHOT']==date) & (datarsp['sn'] > 4.99) )
        sel2b=np.where( (datarsp['ID']==object) & (datarsp['SHOT']==date) & (datarsp['sn'] > 7.99) )
        
        sel3=np.where( (datadetect['col1']==object) & (datadetect['col5']==date))
        sel4=np.where( (datarsp['ID']==object) & (datarsp['SHOT']==date) & (datarsp['sn'] > 11.99) )

        flag1=np.size(sel2)
        flag2=np.size(sel2b)
        flag3=np.size(sel3)
        flag4=0
        flag5=0
        flag6=np.size(sel4)

        #flag to indicate 1 if an object/date was found in detect cats since multiple objects can be associated with single
        if (np.size(sel3)>0): flag4=1
        
        if ( (flag2>0) & (flag4>0)): flag5=1

        output[counter2]=(object,date,flag1,flag2,flag3,flag4,flag5,flag6)

        objcount += 1
        counter2 += 1
        
    selrsp=np.where( (datarsp['ID']==object) & (datarsp['sn'] > 1) )
    nrsp=np.size(selrsp)
  
    seldetect=np.where( (datadetect['col1']==object) )
    ndetect=np.size(seldetect)


#    sel_ifu=np.where(objectifu['ID']==object)
#    ifu_no=objectifu['IFU'][sel_ifu]
#
#    sel_obj_frame=np.where(ifunshot['IFU']==ifu_no)
#    n_obj_frame=ifunshot['nshot'][[sel_obj_frame]]
    sel_obj=np.where(objectngood['col1']==object)
    n_obj_frame=objectngood['col2'][sel_obj]
        
    if (nrsp >= 1) & (n_obj_frame>0): #49 unique objects are found
        rsp_stats[counter]=(object,np.median(datarsp['sn'][selrsp]),np.std(datarsp['sn'][selrsp]),float(nrsp)/float(n_obj_frame),nrsp,n_obj_frame)

    else:
        rsp_stats[counter]=(object,0,0,0,0,0)                                


    if (ndetect >= 1) & (n_obj_frame>0): # 27 unique objects are found

        #redo selection to count repeats only once                                                                                      
        seldetect2=np.where( (output['OBJECTID']==object) & (output['detectcatflag']==1) )
        ndetect=np.size(seldetect2)
        detect_stats[counter]=(object,np.median(datadetect['col10'][seldetect]),np.std(datadetect['col10'][seldetect]),float(ndetect)/float(n_obj_frame),ndetect,n_obj_frame)
    else:
        detect_stats[counter]=(object,0,0,0,0,0)
             
    counter += 1

#Print out some info:

print "Number of S/N>8 objects in the combined rsp catalogs are", np.size(np.where( (output['rsp_sngt8_flag']==1 ))) 
print "Number of S/N>5 objects in the combined rsp catalogs are", np.size(np.where( (output['rsp_sngt5_flag']==1 )))
print "Number of matched objects in the combined detect catalogs are", np.size(np.where( (output['detectcatflag']==1 )))
print "Number of objects found in both the rsp and detect catalogs are", np.size(np.where( (output['detectcatflag']==1 ) & (output['rsp_sngt5_flag']==1 )))

#figure out which object/shot combos worked in rsp but not in detects:

sel=np.where( (output['detectcatflag']==0) & (output['rsp_sngt8_flag']==1) )

ascii.write(output[sel],'missingindets',overwrite=True)

sel2=np.where( (output['detectcatflag']==1) & (output['rsp_sngt5_flag']==0) )


#Now for some plotting

#first plot completeness

nbin=16
snarray=np.arange(nbin)*3
complrsp=np.zeros(nbin)
compldet=np.zeros(nbin)
complrsp_std=np.zeros(nbin)
compldet_std=np.zeros(nbin)

for count in xrange(nbin-1):
    sel=np.where( (rsp_stats['snmedian']>snarray[count]) & (rsp_stats['snmedian']<snarray[count+1]) & (rsp_stats['ngood']>0) )
    if np.size(sel) > 0: 
        complrsp[count]=float(np.sum(rsp_stats['ndet'][sel]))/float(np.sum(rsp_stats['ngood'][sel]))
        complrsp_std[count]=np.std(np.divide(rsp_stats['ndet'][sel],rsp_stats['ngood'][sel],dtype=float))

    sel=np.where( (detect_stats['snmedian']>snarray[count]) & (detect_stats['snmedian']<snarray[count+1]) & (detect_stats['ngood']>0) )
    if np.size(sel) > 0: 
        compldet[count]=float(np.sum(detect_stats['ndet'][sel]))/float(np.sum(detect_stats['ngood'][sel]))
        compldet_std[count]=np.std(np.divide(detect_stats['ndet'][sel],detect_stats['ngood'][sel],dtype=float))


plt.figure()
sel=np.where(complrsp>0)
plt.errorbar(snarray[sel]+1.5,complrsp[sel],xerr=1.5,yerr=complrsp_std[sel],color='blue',marker='o',ecolor='cornflowerblue',elinewidth=2,label='Forced Photometry')
sel=np.where(compldet>0)
plt.errorbar(snarray[sel]+1.5,compldet[sel],xerr=1.5,yerr=compldet_std[sel],color='red',marker='o',ecolor='salmon',elinewidth=2,label='Cross Matching Detect Catalogs')
plt.xlabel('Median S/N')
plt.ylabel('N_detected/N_observed')
plt.legend()
plt.savefig("completeness.png")

x=np.arange(2)

#try to plot this as function of S/N and also find a way to show unique points
plt.figure()
plt.plot(x,x)
plt.plot(detect_stats['fraction'],rsp_stats['fraction'],'ro')
plt.xlabel('Detect Catalog Recovery Fraction')
plt.ylabel('RSP catalog recovery fraction')
plt.savefig('fraction_comps.png')
plt.close()

x=np.arange(50)
plt.figure()
plt.axis([0,40,0,40])
plt.plot(x,x,color='red')
plt.errorbar(rsp_stats['snmedian'],detect_stats['snmedian'],yerr=detect_stats['snrms'],xerr=rsp_stats['snrms'],fmt='o',color='blue',ecolor='lightgray')
plt.xlabel('Median SN from rsp calls')
plt.ylabel('Median SN from detect catalog matches')
plt.savefig('detectsn_vs_rspsn.png')
plt.close()

plt.figure()
plt.errorbar(detect_stats['snmedian'],detect_stats['fraction'],yerr=None,xerr=detect_stats['snrms'],fmt='o',color='blue',ecolor='lightgray')
plt.xlabel('Median S/N')
plt.ylabel('Fraction Recovered in Detect Cats')
plt.ylim(-0.05,1.05)
plt.xlim(-2,50)
plt.savefig('detectfraction_vs_sn.png')
plt.close()

plt.figure()
plt.errorbar(rsp_stats['snmedian'],rsp_stats['fraction'],yerr=None,xerr=rsp_stats['snrms'],fmt='o',color='blue',ecolor='lightgray')
plt.xlabel('Median S/N')
plt.ylabel('Fraction Recovered in RSP calls')
plt.ylim(-0.05,1.05)
plt.xlim(-2,50)
plt.savefig('fraction_vs_sn.png')
plt.close()

plt.figure()
sel3=np.where( rsp_stats['snrms']>0)
plt.hist(rsp_stats['snrms'][sel3])
plt.xlabel('SN_rms')
plt.ylabel('n')
plt.savefig('rsp_snrms.png')
plt.close()

plt.figure()
sel4=np.where( detect_stats['snrms']>0)
plt.hist(detect_stats['snrms'][sel3])
plt.xlabel('detect_SNrms')
plt.ylabel('n')
plt.savefig('detect_snrms.png')
plt.close()

#plt.plot(data['qual1'][sel],data['qual2'][sel],'ro')
#plt.plot(data['qual1'],data['qual2'],'b.')
#plt.ylabel('N_matched/N_stars')
#plt.xlabel('N_matched/N_detected')
#plt.hlines(0.2, -0.03,0.05)
#plt.vlines(0.05,-0.01, 0.2)
#plt.savefig('astrometric_qc.png')
