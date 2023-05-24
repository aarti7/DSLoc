#!/usr/bin/env python3

import scipy.signal

from matplotlib.offsetbox import AnchoredText
from matplotlib.patches import FancyArrowPatch
import plotly.graph_objects as go
import seaborn as sns

from PIL import Image

# import statsmodels
# from statsmodels.graphics import tsaplots

import os, sys
import argparse
import re
import itertools
import urllib.request
import pdb
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import numpy as np

import h5py
import datetime 
import pickle as pkl


import pandas as pd
from pathlib import Path
import scipy.constants
# print("SciPy also thinks that the speed of light is c = %.1F" %scipy.constants.c ) 

import time

import math 


DEF_DATADIR="./"
DEF_DFNAME="measurements.hdf5"
DEF_BRANCH="wtx"
MEAS_ROOT="saveiq_w_tx"


###############################################################
###############################################################    


def get_avg_power(samps):
    return 10.0 * np.log10(np.sum(np.square(np.abs(samps)))/len(samps))

###############################################################
###############################################################    

def psdcalc(this_measurement, rate):
    ##PSD
    nsamps=len(this_measurement)
    
    this_measurement = this_measurement - np.mean(this_measurement)

    window = np.hamming(nsamps)
    result = np.multiply(window, this_measurement)
    result_fft = np.fft.fft(result, nsamps)     # result_magsq = np.square(np.abs(result))
    result_fft_shifted = np.fft.fftshift(result_fft)
    result_shifted_magsq = np.square(np.abs(result_fft_shifted)) # abs = sqrt(i^2+q^2)
    
    psd_is_linear =  np.nan_to_num(result_shifted_magsq)
    psd_is_db =  np.nan_to_num(10.0 * np.log10(result_shifted_magsq))
    freqs = np.fft.fftshift(np.fft.fftfreq(nsamps, 1/rate))

    return psd_is_db, psd_is_linear, freqs


###############################################################
###############################################################    


def match(rate, nsamps, matched_row_ingt, this_time, wl):

    #A.
    this_speed = matched_row_ingt['Speed (meters/sec)'] # this_speed = gt_loc_df[gt_loc_df['timeCT'].str.contains(this_time[0:-7])]['speed']

    #B. 
    expected_doppler = this_speed/wl # print('expectedDoppler is', expected_doppler.values, "speed is" , this_speed.values)

    #C.
    freq_res = rate/nsamps # print( "freq_res", freq_res)

    #D.
    mth_bin_for_expected_doppler_in_psdfreqs_notXaxis = int(np.ceil(expected_doppler.values/ freq_res) )

    # E. 
    if len(matched_row_ingt.columns) == 6:
        direction_w_TN = matched_row_ingt['Track'] # only the new experiments have the Track data in an additional column of the GT_LOC_CSV
    else:
        direction_w_TN = 0

    return this_speed, expected_doppler,  freq_res, mth_bin_for_expected_doppler_in_psdfreqs_notXaxis, direction_w_TN



###############################################################
###############################################################    

def leaves_to_DF(leaves, allsampsandtime):
    # print('h5dump leaves is \n', leaves)
    endpoints=[]
    df_allrx= pd.DataFrame()
    df_allti = pd.DataFrame()
    for k in range(0,len(leaves),2): 
        
        tx =  leaves[k].split('/')[1]
        gtx = leaves[k].split('/')[2] if args.whichbranch == 'wtx' else 'noTXhappened'
        grx = leaves[k].split('/')[3] if args.whichbranch == 'wtx' else leaves[k].split('/')[2]
        rx =  leaves[k].split('/')[4] if args.whichbranch == 'wtx' else leaves[k].split('/')[3]
        samp = leaves[k].split('/')[5] if args.whichbranch == 'wtx' else leaves[k].split('/')[4]
        time =  leaves[k+1].split('/')[5] if args.whichbranch == 'wtx' else leaves[k+1].split('/')[4]

        endpoints.append(rx)
        print("reading hdf5 of RX:",rx)

        if args.whichbranch == 'wtx':
            print("in tx leaf")
            ds_samples = allsampsandtime[tx][gtx][grx][rx][samp]
            ds_times = allsampsandtime[tx][gtx][grx][rx][time]
        else:
            print("in WO_TX leaf.. ******************************************** SHOULDNT HAPPEN since you gave arguments for wtx brach! ")
            ds_samples = allsampsandtime[tx][grx][rx][samp]
            ds_times = allsampsandtime[tx][grx][rx][time]

        sm = { rx : list( ds_samples[()]  )}
        ti = { f'{rx}_ti' : list( ds_times[()]  )}

        df_samples = pd.DataFrame(sm)
        df_time = pd.DataFrame(ti)

        df_allrx = pd.concat([df_allrx, df_samples], axis=1)
        df_allti = pd.concat([df_allti, df_time], axis=1)

    
    columns_array = df_allrx.columns.values

    ## Was removing in old experiments as hospital was not white-rabbit synched!
    # if 'cbrssdr1-hospital-comp' in columns_array: 
    #     df_allrx = df_allrx.drop('cbrssdr1-hospital-comp', axis=1) 
    #     df_allti = df_allti.drop('cbrssdr1-hospital-comp_ti', axis=1) 

    # if 'cbrssdr1-fm-comp' in columns_array:
    #     df_allrx = df_allrx.drop('cbrssdr1-fm-comp', axis=1)  
    #     df_allti = df_allti.drop('cbrssdr1-fm-comp_ti', axis=1)

    return tx, endpoints, df_allrx, df_allti




###############################################################
###############################################################    


def freq_off_averaged_for_full_df(df_allrx, df_allti, gt_loc_df, fsr, numbsamp, wl, thistx):

    bus_frequency_offset_ranges = mep_ppm[thistx[4:]]

    columns_array = df_allrx.columns.values
    n_endpoints_is_subplots = len(columns_array)

    freqoff_dict = {key: [] for key in columns_array}
    mean_frqoff_perrx_dict = {}
    
    speed_time_dict = {'speeds':[],'times':[]} 

    how_many_zero_vel = 0
    how_many_nonzero_vel = 0

    for n in range(0, len(df_allrx)):
        for p in   range(0, n_endpoints_is_subplots): #    [1]:#     [3]:# 
            this_measr_time = df_allti.iloc[n,p]
            
            if len(this_measr_time)==0:
                print("no iq data rxd at this time for this column", p, "msrmnt=", n)
                break

            # print("np is", n, p)

            this_measr_timeuptoseconds = this_measr_time[0:-7] # NOTE: ground truth is per second but has no resolution further than 'seconds'

            matched_row_ingt = gt_loc_df[gt_loc_df['Time (UTC)'].str.contains(this_measr_timeuptoseconds)] 

            if len(matched_row_ingt) == 0:
                print(f'somehow bus didnt record {n}th GPS measurment! for this row', n)
                break

            this_speed, expected_doppler, freq_res, THEbin_for_Doppler, direction_w_TN = match(fsr, numbsamp,  matched_row_ingt, this_measr_timeuptoseconds, wl)
            
            this_measurement  = df_allrx.iloc[n,p]
            psddb, psdlinear, fr = psdcalc(this_measurement, fsr)


            # plt.clf()
            # plt.plot(fr,psddb, label=f"{n}{columns_array[p][9:12]}\n mean{np.mean(psddb)}\n max {psddb[np.argmax(psddb)]} \n frq {fr[np.argmax(psddb)]}\n{this_speed.values[0]}", color = 'g' if this_speed.values[0] !=0 else 'r')
            # plt.legend( loc='lower left')
            # plt.title('regardless of the speed: to check if rx happened and pick range!')
            # plt.show()

            
            if this_speed.values[0]==0: # to ensure only for cases  when there was no motion. 

                this_measurement  = df_allrx.iloc[n,p]

                result_fft_temp, psdlinear, freqs_temp = psdcalc(this_measurement, fsr) #sq_abs_fftshift_fft

                # Weird inteference at honors! I think LO but I did set it = 0.6 times SR ! aaaaa!!!
                if columns_array[p] == 'cbrssdr1-honors-comp':
                    result_fft_temp[freqs_temp > 90000] = result_fft_temp[90000]
                    result_fft_temp[freqs_temp <-90000] = result_fft_temp[90000]

                all_peaks_idxs, all_peaks_vals = scipy.signal.find_peaks(result_fft_temp) 
                idx_psd_max = all_peaks_idxs[np.argmax(result_fft_temp[all_peaks_idxs])] 

                val_psd_max = result_fft_temp[idx_psd_max]
                val_freq_max = freqs_temp[idx_psd_max]  

                # print("freq offset is", val_freq_max, "max power is" , val_psd_max, n, p)
                # print("mean of psd is", np.mean(result_fft_temp), "std is", np.std(result_fft_temp), "3times std of psd is", 3*np.std(result_fft_temp))

                # plt.clf()
                # plt.plot(freqs_temp, result_fft_temp,  label=f"{n}{df_allrx.columns[p][9:12]}\n mean{np.mean(result_fft_temp)}  \n max {val_psd_max}  \n frq {val_freq_max}", color = 'r' if this_speed.values[0] ==0 else 'g')
                # plt.legend(loc='lower left')
                # plt.title('only for the zero speeds')
                # plt.show()

                threshold = -21 #-23.5 # np.mean(result_fft_temp) + 3*np.std(result_fft_temp)

                # to ensure signal was indeed "seen"
                if val_psd_max > threshold and val_freq_max < bus_frequency_offset_ranges[1] and val_freq_max > bus_frequency_offset_ranges[0]:
                    # print("val_psd_max" , val_psd_max)

                    freqoff_dict[df_allrx.columns[p]].append(val_freq_max)
                how_many_zero_vel = how_many_zero_vel+1
            else:
                how_many_nonzero_vel = how_many_nonzero_vel+1

        # if len(matched_row_ingt) !=0:
        #     speed_time_dict['speeds'].append(this_speed.values[0])
        #     speed_time_dict['times'].append(this_measr_timeuptoseconds.split(' ')[-1])
        #     # print("n is",n, this_speed.values[0])
    
    print("Freq offsets when no motion", freqoff_dict)
    print("\nStationary signal was NOT seen AT ALL for the RX\n", [key for key, value in freqoff_dict.items() if len(value) == 0]) 

    if len([key for key, value in freqoff_dict.items() if len(value) == 0]) != n_endpoints_is_subplots:
        mean_frqoff_perrx_dict = {key: sum(values) / len(values) if len(values) != 0 else None for key, values in freqoff_dict.items()}
        naverage = sum(value for value in mean_frqoff_perrx_dict.values() if value !=None) / len([value for value in mean_frqoff_perrx_dict.values() if value != None])
        mean_frqoff_perrx_dict = {key: ( naverage if value == None else value) for key, value in mean_frqoff_perrx_dict.items()}
    else:
        print("\n\nStationary signal never seen at any BS, check your bus_frequency_offset_ranges !!!!!!")
        exit()

    return mean_frqoff_perrx_dict, how_many_zero_vel/n_endpoints_is_subplots, how_many_nonzero_vel/n_endpoints_is_subplots # , speed_time_dict



###############################################################
# All possible receiver/sensor coordinates
###############################################################

all_coords = {
  # 'meb': (40.768796, -111.845994),
  # 'browning': (40.766326, -111.847727),
  'hospital': (40.77105, -111.83712),  
  'ustar': (40.76899, -111.84167),
  'bes': (40.76134, -111.84629),
  # 'fm': (40.75807, -111.85325),
  'honors': (40.76440, -111.83695),
  'smt': (40.76740, -111.83118),
  # 'dentistry': (40.758183, -111.831168),
  # 'law73': (40.761714, -111.851914),
  # 'ustar_nuc': (40.76852, -111.84045),
  # 'mario': (40.77283, -111.84088),
  # 'wasatch': (40.77108, -111.84316),
  # 'ebc': (40.76702, -111.83807),
  # 'guesthouse': (40.76749, -111.83607),
  # 'moran': (40.76989, -111.83869)
  }


###############################################################    
##################  All possible mep offsets ##################
###############################################################
mep_ppm = {
# '6180': [7000,8000],
'6181': [7500,8500],
# '6182': [7000,8000],
# '6185': [7000,8000],
'6183': [6500,7500],
'4555': [7500,8500],
'4603': [6000,7000],
'4604': [6000,7000],
'4734': [8500,10000],
'4407': [8500,9500]
}
# '6183': [7000,8000], They changed based on new day! manually putting the range AFTER running the 'freq_off_averaged_for_full_df'
# '4555': [3000,4000],
# '4734': [8500,9500],
# '6183': [1900,5000],

###############################################################
###############################################################    



def do_psd_storing(attrs, allsampsandtime, leaves):
    
    #### GT_LOC_CSV #############################
    print('\nCurrent working directory is:', os.getcwd(), "\n")

    loc_files = list(Path(args.datadir).rglob('*.csv'))
    print('Location file to read is:', loc_files, "\n")
    
    if len(loc_files) == 0:
        print(" No csv file is present!!!!!!!")
        exit()
    else:
        gt_loc_df = pd.read_csv(loc_files[0], header=0)
        gt_loc_df['Time (UTC)'] = pd.to_datetime(gt_loc_df['Time (UTC)']).dt.tz_convert('US/Mountain').dt.strftime("%Y-%m-%d %H:%M:%S")
        
        # print("The very first bus position tuple: \n", (gt_loc_df['Lat'].iloc[0], gt_loc_df['Lon'].iloc[0]))
        print("The max bus speed in this experiment was:", gt_loc_df['Speed (meters/sec)'].max(), 'Overall length of CSV', gt_loc_df.shape, "\n")
    

    #### ATTRS #############################
    
    rate = attrs['rxrate']
    nsamps = attrs['nsamps']

    freq_resolution = rate/nsamps

    tx_CENTER_FREQUENCY  = attrs['txfreq']
    rx_CENTER_FREQUENCY = attrs['rxfreq']
    print('tx and rx frequency <<<MATCHED>>> \n' if (tx_CENTER_FREQUENCY==rx_CENTER_FREQUENCY) else 'tx and rx frequency <<<DONT>>> match')
    
    MAXIMUM_BUS_SPEED_POSSIBLE =  21 #mps!

    MAXIMUM_SPEED_SEEN = gt_loc_df['Speed (meters/sec)'].max() #+1 #20 meters per second == 44.7 miles per hour. 30 mps == 67.1 mph
    

    bus_go_vroom_flag = False
    if MAXIMUM_SPEED_SEEN > MAXIMUM_BUS_SPEED_POSSIBLE:
        bus_go_vroom_flag = True
        print(" \n\n BUS DROVE FASTER THAN YOU ASSUMED! !!!!!! \n\n")     
        bus_go_vroom_indexes_gt_df = gt_loc_df.loc[gt_loc_df['Speed (meters/sec)'] >=MAXIMUM_BUS_SPEED_POSSIBLE].index
        print("gt dataframe indexes of higher speed", np.array(bus_go_vroom_indexes_gt_df))
        exit()

    WAVELENGTH = scipy.constants.c / tx_CENTER_FREQUENCY 
    FD_MAXPOSSIBLE =  MAXIMUM_BUS_SPEED_POSSIBLE / WAVELENGTH
    print("\nFD_MAXPOSSIBLE is          == " , FD_MAXPOSSIBLE, "\n" )


    #### get all DATA #############################

    txis, endpoints, df_allrx, df_allti = leaves_to_DF(leaves, allsampsandtime)
    bus_frequency_offset_ranges = mep_ppm[txis[4:]]
    
    print("\nThis experiment's bus is   == ", txis, "\n \
        With preset offset-finding frequency ranges == ",
     bus_frequency_offset_ranges,"\n")

    n_endpoints_is_subplots = len(df_allrx.columns.values)        # len(endpoints)

    print('Number of total measurements ==', len(df_allrx), "\n") 
    print('The Base Stations are        ==', df_allrx.columns, "\n") # print( "End points ==", endpoints) #print( "oh boi, do they match", df_allrx.columns.values == endpoints)

    #### get freq offset  ############################# 
    frqoff_perrx, numb_stationary_msrmnts, n_moving_msrmnts = freq_off_averaged_for_full_df(df_allrx, df_allti, gt_loc_df, rate, nsamps, WAVELENGTH, txis) #, allVT
    

    print("Averaged freq offset is           ==", frqoff_perrx, "\n")
    print("Number of stationary measurements ==", numb_stationary_msrmnts, "\n")
    print("Number of moving measurements     ==", n_moving_msrmnts, "\n")
    
    # print("all hardcoded values are:", MAXIMUM_BUS_SPEED_POSSIBLE, numtaps_forfilterout_weird_signal)
    
    ##########################################################

    
    psd_dict = {key: [] for key in df_allrx.columns.values}
    psd_dict["speed_postuple"] = []
    psd_dict["dt_time"] = []

    colorbox = ['r','g']
    runtime = f'{int(time.time())}' 
    print("runtime", runtime)



    Path(args.datadir+'/pickledfiles/').mkdir(parents=True, exist_ok=True)
    pickled_dir = Path(args.datadir+'/pickledfiles/'+runtime)
    pickled_dir.mkdir(parents=True, exist_ok=True)

    if args.saveplots =='True':
        Path(args.datadir+'/psd_plots').mkdir(parents=True, exist_ok=True)
        psd_plots_dir = Path(args.datadir+'/psd_plots/'+runtime)
        psd_plots_dir.mkdir(parents=True, exist_ok=True)
        
        Path(args.datadir+'/overall_plots').mkdir(parents=True, exist_ok=True)
        overall_plots_dir = Path(args.datadir+'/overall_plots/'+runtime)
        overall_plots_dir.mkdir(parents=True, exist_ok=True)


    # plt.ion() # you can generate multiple fig and each will be ion mode!

    for n in range(0, len(df_allrx) ):        
       
        if args.saveplots =='True':
            fig_overall, axs_overall = plt.subplots(3, n_endpoints_is_subplots,  num="overall", figsize=(16, 7), sharey= False)  # CREATE THE FIGURE OUTSIDE the loop!
        
        print('for nth measurement = ', n) 
        for p in range(0, n_endpoints_is_subplots):    
            print('pth column', p)            
             
            this_measr_time = df_allti.iloc[n,p]  

            if len(this_measr_time)==0:
                print("AGAIN: no iq data rxd at this time for this column", p, "msrmnt=", n)
                break 

            # Time - will be pickling!
            this_measr_timeuptoseconds = this_measr_time[0:-7] # NOTE: ground truth is per second but has no resolution for 'subseconds'                                                
            
            
            # get respective row from GT_LOC_CSV
            matched_row_ingt = gt_loc_df[gt_loc_df['Time (UTC)'].str.contains(this_measr_timeuptoseconds)]           
            
            
            if bus_go_vroom_flag: # at least wont check for those days when bus didnt run faster!
                speed_index = matched_row_ingt.index.values[0]
                if speed_index in bus_go_vroom_indexes_gt_df:
                    print(matched_row_ingt)
                    print("BUS DROVE FASTER THAN YOU ASSUMED for this index, so skipping this row!!! \n\n")
                    break

            if len(matched_row_ingt) == 0:
                print(f'somehow bus didnt record {n}th GPS measurment!')
                break


            # Location - will be pickling!
            current_bus_pos_tuple = (matched_row_ingt['Lat'].values[0],matched_row_ingt['Lon'].values[0])

            # Speed - will be pickling!
            this_speed, expected_doppler, freq_res, THEbin_for_Doppler, direction_w_TN = match(rate, nsamps,  matched_row_ingt, this_measr_timeuptoseconds, WAVELENGTH)
            
            cc = [ 1 if this_speed.values[0] !=0 else 0][0] # Just a helpful flag

            fdmaxis = expected_doppler.values[0] # # correctly named as 'max' # print('gps \'speed\' was the magnitude of velocity vector. speed*anycos(theta) will be <= speed. So this \'speed\' gives the maximum doppler possible from this \'speed\'')                                                           
                                                                                                                

            butDSwillbeseenatpsd_is_freq_right = THEbin_for_Doppler*freq_res # cause of the fft resolution# also it is already, ceil cause positive
            # butDSwillbeseenatpsd_is_freq_left =  # here it should be np.floor but no need to do separately as a simple -ve sign is equal to making floor
            
            # IQ measurements
            this_measurement  = df_allrx.iloc[n,p]

            # Power!
            power_orignal = get_avg_power(this_measurement) # print("a_power_orignal", power_orignal)


            # # ## just another check!
            aa, bb, dd = psdcalc(this_measurement, rate)                         
            aa_psd_max = aa[np.argmax(aa)] #gives fftfres based!!
            aafreqmax = dd[np.argmax(aa)] # print("freq_maxpeak_fdfc", aafreqmax, fdmaxis)   


            # ############## PLOT_C for .ion() for continuous viewing!
            # plt.clf()
            # plt.plot(dd, aa, color = colorbox[cc])
            # plt.title(f"{n} {df_allrx.columns[p].split('-')[1]} speed={this_speed.values[0]} freqmax={aafreqmax}")
            # plt.draw()
            # plt.pause(0.1) 
            # #############



            ### plot_A begins### 
            if args.saveplots =='True':
                axs_overall[0][p].clear()

                axs_overall[0][p].plot(dd,aa, color=colorbox[cc])
                axs_overall[0][p].annotate( 'maxfreq\n{}'.format(round(aafreqmax,1)), xy=(aafreqmax, aa_psd_max), xytext=(aafreqmax, aa_psd_max+2), 
                    arrowprops=dict(facecolor='yellow', shrink=0.9),)

                axs_overall[0][p].set_title(f"Before correcton:\n {n} {df_allrx.columns[p].split('-')[1]} speed={this_speed.values[0]}" , color = colorbox[cc])
                axs_overall[0][p].set_ylim(-80,10)
                # plt.draw()
                plt.figure("overall").canvas.draw() 
                                                                                                                                                                                                    


            foffset_approximated = frqoff_perrx[df_allrx.columns[p]]                                            # It, though, was average of freqs that came from fft_res, but averaging removed the fft_res effect, so need to do that again in the next line
            foffset_approximated = int(np.ceil(foffset_approximated/freq_res))*freq_res
            
            Ts = 1/rate                                                                                         # calc sample period i.e. how long does one sample take to get collected
            NTs = len(this_measurement)* Ts
            t = np.arange(0, NTs, Ts)                                                                           # for N samples worth of duration, create the time vector, with a resolution =1/fs
                                                                                                                
            corrected_signal_withfdANDfresidue = this_measurement * np.exp(-1j*2*np.pi*foffset_approximated*t)  # remove most of the offset! ONLY a small residue will be left from here on!           


            ##### JUST to see! another check!
            ss_idx = int(np.ceil(9000/ freq_res)) ### 9000 is a random choice, just to plot a small section of indexes. Full signal and all the indexes still there!

            xx,yy,zz = psdcalc(corrected_signal_withfdANDfresidue, rate)       
            maxpw = xx[int(len(zz)/2)-ss_idx+1:int(len(zz)/2)+ss_idx][np.argmax(xx[int(len(zz)/2)-ss_idx+1:int(len(zz)/2)+ss_idx])]                             # # just a check!
            maxfq = zz[int(len(zz)/2)-ss_idx+1:int(len(zz)/2)+ss_idx][np.argmax(xx[int(len(zz)/2)-ss_idx+1:int(len(zz)/2)+ss_idx])]                                                                  #  gives fftfres based!!
            
            if args.saveplots =='True':
                axs_overall[1][p].clear()
                axs_overall[1][p].plot( zz[int(len(zz)/2)-ss_idx+1:int(len(zz)/2)+ss_idx], xx[int(len(zz)/2)-ss_idx+1:int(len(zz)/2)+ss_idx], color = colorbox[cc])
                axs_overall[1][p].set_ylim(-80,10)
                axs_overall[1][p].set_title(f" Avg foff {round(foffset_approximated,1)}", y=0.3 ,color = 'm' if maxpw >-19 else 'k' )
                axs_overall[1][p].annotate( 'maxfreq\n{}'.format(round(maxfq,1)), xy=(maxfq, maxpw), xytext=(maxfq, maxpw+2), 
                    arrowprops=dict(facecolor='yellow', shrink=0.9), color = 'm' if maxpw >-19 else 'k')
                plt.figure("overall").canvas.draw() 


            power_corrected  = get_avg_power(corrected_signal_withfdANDfresidue)  # print("b_power_corrected (before the honors LPF condition)", power_corrected)


            ###### Weird inteference! I think LO but I did set it = 0.6 times SR ! aaaaa!!!
            
            if df_allrx.columns.values[p] == 'cbrssdr1-honors-comp':
            # if True: # do it for all BS, not just honors!

                # print("in LPF condition!")

                numtaps_forfilterout_weird_signal  = 1001
                fc                                 = 25000
                myLPF                              = sig.firwin(numtaps_forfilterout_weird_signal, cutoff=fc, fs=rate)

                this_measurement_after_LPF = sig.convolve(this_measurement, myLPF, mode='same')
                power_x_lpf           = get_avg_power(this_measurement_after_LPF)
                # print("c) power_x_lpf", power_x_lpf)
                x_lpf  = np.nan_to_num(10.0*np.log10(np.square(np.abs(np.fft.fftshift(np.fft.fft(this_measurement_after_LPF)))))) # magnitude
                # # qw,er,ty = psdcalc(this_measurement_after_LPF, rate)  

                corrected_signal_withfdANDfresidue_then_LPF = sig.convolve(corrected_signal_withfdANDfresidue, myLPF, mode='same')
                power_xcorrected_lpf          = get_avg_power(corrected_signal_withfdANDfresidue_then_LPF)
                # print("d) power_xcorrected_lpf", power_xcorrected_lpf)
                xc_lpf = np.nan_to_num(10.0*np.log10(np.square(np.abs(np.fft.fftshift(np.fft.fft(corrected_signal_withfdANDfresidue_then_LPF))))))  # magnitude
                # # qwcs,ercs,tycs = psdcalc(corrected_signal_withfdANDfresidue_then_LPF, rate) 
                                       
                xLPF_corrected_signal_withfdANDfresidue = this_measurement_after_LPF * np.exp(-1j*2*np.pi*foffset_approximated*t)               
                power_x_lpf_corrected = get_avg_power(xLPF_corrected_signal_withfdANDfresidue)
                # print("e) power_x_lpf_corrected", power_x_lpf_corrected)
                xlpf_cs = np.nan_to_num(10.0*np.log10(np.square(np.abs(np.fft.fftshift(np.fft.fft(xLPF_corrected_signal_withfdANDfresidue))))))  # magnitude


                # ##### PLOT_weird starts!
                # figlpf, axlpf = plt.subplots(3, 1, figsize=(9, 7), num = "LPF4honors")
                # ##### unfiltered: original signal and frequncy corrected original signal
                # # axlpf[0, 0].plot(dd,aa, label = "a_x",   c = "red")
                # # axlpf[1, 0].plot(zz,xx, label = "b_xc",    c = "blue")
                # # axlpf[0, 0].set_title("x")
                # # axlpf[1, 0].set_title("x freq corrected")

                # freq_plf = np.fft.fftshift(np.fft.fftfreq(len(this_measurement), d=1/rate)) # shifted frequencies.... per original iq..for everyone?

                # axlpf[0].scatter(dd,aa, label = "x",   c = "red", s= 5)                
                # axlpf[0].plot(freq_plf, x_lpf, label = "xLPF",    c = "magenta")
                # axlpf[0].set_title(f"(c) x after LPF, RSSI  {round(power_x_lpf,5)}")

                # axlpf[1].scatter(zz,xx, label = "xc",    c = "blue", s= 5)               
                # axlpf[1].plot(freq_plf, xc_lpf,   label = "xcLPF", c = "green")
                # axlpf[1].set_title(f"(d) xfreqcorrected then LPF, RSSI {round(power_xcorrected_lpf,5)}")  

                # axlpf[2].scatter(freq_plf, x_lpf,     label = "xLPF",    c = "magenta", s=5)
                # axlpf[2].scatter(freq_plf, xc_lpf,    label = "xcLPF",   c = "green", s=5)
                # axlpf[2].plot(freq_plf, xlpf_cs,      label = "xLPFc",  c= "black", linestyle  = "dashed", linewidth= .5)
                # axlpf[2].set_title(f"(e) xLPF then freqCorrected, RSSI {round(power_x_lpf_corrected,5)}")          
                
                # for i, ax in enumerate(axlpf.flat):
                #     ax.grid(True)
                #     ax.legend()
                #     if i < 2:
                #         ax.set_xticks([])

                # this_bs = df_allrx.columns.values[p].split('-')[1]
                # plt.suptitle(f"{n} {p} {this_bs}")
                # date_time_namestring = (args.datadir).split('/')[-1].split('s_')[-1]
                # plt.savefig(f"{psd_plots_dir}/LPF_{this_bs}_{date_time_namestring}_{n}_{p}.eps",format='eps')
                # plt.close("LPF4honors")
                # ##### PLOT_weird ends!

                ###### assign filtred values back to the old variables for rest of the code (only for honors)
                corrected_signal_withfdANDfresidue = xLPF_corrected_signal_withfdANDfresidue # corrected_signal_withfdANDfresidue_then_LPF
                power_corrected = power_x_lpf_corrected # power_xcorrected_lpf

            
            # RSSI FUll fiterted SIGNAL  - will be pickling!
            power_corrected  = get_avg_power(corrected_signal_withfdANDfresidue) # print("b_power_corrected (after the honors condition)", power_corrected)

            
            fig_plt = plt.figure("fig_pltfunction")
            [pxxc_linear, fxxc] = plt.psd(corrected_signal_withfdANDfresidue, NFFT= nsamps, Fs = rate) ## pxxc_DB, pxxc_linear, fxxc = psdcalc(corrected_signal, rate) # not using my custom psd calculation function ###### USING THE plt.psd DEFAULT FUNCTION TO acutally obtain the psd values.. it is 1.linear and 2. per hertz!
            # print("pxxc_linear", pxxc_linear)
            pxxc_DB = 10.0 * np.log10(pxxc_linear) 
            plt.close("fig_pltfunction") # you have to manually close the plt.psd else it will plot by dafault. Imp: for plotting it converts linear to db automatically!
            

            # narrowspectrum (ns)
            ns = FD_MAXPOSSIBLE + 100 #40 #hz #a little more buffer on both sides
            
            fdidx  = (fxxc >= -ns) & ( fxxc <= ns)            
            fxxc_ns = fxxc[fdidx]
            pxxc_ns_DB = pxxc_DB[fdidx]
            pxxc_ns_linear = pxxc_linear[fdidx]  # - will be pickling!


            
            if args.saveplots  =='True':          

                fleft_maxvelds   =  int(np.floor(-FD_MAXPOSSIBLE/ freq_res))*freq_res # r
                fright_maxvelds  =  int(np.ceil ( FD_MAXPOSSIBLE/ freq_res))*freq_res # r

                axs_overall[2][p].clear()
                axs_overall[2][p].plot(fxxc_ns, pxxc_ns_DB, color = colorbox[cc]) # axs[2][p].set_title(f"{df_allrx.columns[p].split('-')[1]}", y=-0.3)
                axs_overall[2][p].set_ylim(-170,-100)
                
                axs_overall[2][p].axvline(x = fright_maxvelds, ls='-', c= 'r')
                axs_overall[2][p].axvline(x = fleft_maxvelds,  ls='-', c= 'r')
                axs_overall[2][p].axvline(x =  butDSwillbeseenatpsd_is_freq_right, ls = ':', c= 'b')  
                axs_overall[2][p].axvline(x = -butDSwillbeseenatpsd_is_freq_right, ls = ':', c= 'b')
                axs_overall[2][p].set_title(f"FD_MAXPOSSIBLE {round(FD_MAXPOSSIBLE,1)}\n fd_for_this_speed {round(butDSwillbeseenatpsd_is_freq_right,1)}", y= -0.4) 
                plt.figure("overall").canvas.draw()
                ### plot_A ends ### 



                #### plot_B_begins ### 
                fig, axx = plt.subplots(2,1, num="narrowspectrum",figsize=(16, 7))
                for i, ax in enumerate(axx.flat): ax.grid(True)
                
                freqs = fxxc_ns
                psd_is = pxxc_ns_DB

                axx[0].bar(freqs, pxxc_ns_linear, color='g', width = .5)
                axx[0].set_ylim(0,10**(-11))
                
                axx[1].plot(freqs, pxxc_ns_DB)
                axx[1].set_ylim(-170,-100)
                axx[1].axvline(x = fright_maxvelds, ls='--', c= 'r')
                axx[1].axvline(x = fleft_maxvelds,  ls='--', c= 'r')
                axx[1].axvline(x =   butDSwillbeseenatpsd_is_freq_right,  ls='--', c= 'b')
                axx[1].axvline(x = -(butDSwillbeseenatpsd_is_freq_right), ls='--', c= 'b')
                this_bs = df_allrx.columns.values[p].split('-')[1]
                plt.figure("narrowspectrum").suptitle(f"{n} {p} {this_bs} v={this_speed.values[0]} blue: this speed ds=+/-{round(butDSwillbeseenatpsd_is_freq_right,2)}, \n red: max_DS+/-{round(fleft_maxvelds,2)},  RSSI={round(power_corrected,5)}", color = colorbox[cc])
                plt.figure("narrowspectrum").tight_layout()
                date_time_namestring = (args.datadir).split('/')[-1].split('s_')[-1]
                plt.figure("narrowspectrum").canvas.draw() 


                plt.savefig(f"{psd_plots_dir}/narrow_psd_{date_time_namestring}_{n}_{p}.eps",format='eps')
                # plt.figure("narrowspectrum").show() 
                plt.close("narrowspectrum")
                ####plot_B_ends### 


            #pickling!
            psd_dict[df_allrx.columns[p]].append((pxxc_ns_linear, power_corrected)) # FORMAT of each value is :list(tuple(firstisarray,secondisnumpy_of_float64))
            # print("psdTOTALlength", len( psd_dict[df_allrx.columns[p]][0][0]))

            if p==n_endpoints_is_subplots-1: # only after the last BS, we store time, loc, and speed
                psd_dict["speed_postuple"].append([this_speed.values[0], current_bus_pos_tuple])
                psd_dict["dt_time"].append(this_measr_timeuptoseconds)


        if args.saveplots  =='True':
            plt.figure("overall").suptitle(f"{n} v={this_speed.values[0]} ds=+/-{butDSwillbeseenatpsd_is_freq_right}", color = colorbox[cc])
            plt.figure("overall").tight_layout()
            date_time_namestring = (args.datadir).split('/')[-1].split('s_')[-1]
            plt.figure("overall").canvas.draw() 
            
            plt.figure("overall").savefig(f"{overall_plots_dir}/overall_{date_time_namestring}_{n}.eps",format='eps')
            # plt.figure("overall").show() 
            plt.close("overall")  

        print(f'\n\n{n}th row measurment done\n\n\n')


    # plt.close()
    # plt.ioff()

    fn = f"{pickled_dir}"+"/"+ f"{args.datadir}".split('/')[-1]+f"_{runtime}"+'.pickle'
    pkl.dump(psd_dict, open(fn, 'wb' ) )
    print("pickled!")
    loaded_psd = pkl.load(open(fn, 'rb') )
    print("how many rows in this pickled file", len(loaded_psd['dt_time']))


def get_dataset_keys(f):
    keys = []
    f.visit(lambda key : keys.append(key) if isinstance(f[key], h5py.Dataset) else None)
    return keys

def mainn(args):

    dsfile = h5py.File("%s/%s" % (args.datadir, args.dfname), "r")

    dsfile_with_meas_root = dsfile[MEAS_ROOT]
    all_leaves_2 = get_dataset_keys(dsfile_with_meas_root)

    nl = len(all_leaves_2)
    print( "\n\n\nLength of entire tree is", nl)

    # print("entire Tree without TX and with TX\n",all_leaves_2)
    # [print(i) for i in all_leaves_2]
    
    if nl <20:
        print("less than 5 BaseStations which is required for current training setup!!")
        exit()

    timestamp_leaf = all_leaves_2[0].split('/')[0]
    

    if all_leaves_2[0].split('/')[1] == 'wo_tx':
        print("Only wo_TX data collection happened.\n You should confirm what you are looking for!!")
        leaves_withoutTX = all_leaves_2
    else:
        leaves_withTX = all_leaves_2[0:int(nl/2)] # first half are tx leaves
        leaves_withoutTX = all_leaves_2[int(nl/2):] # second half are wo_tx leaves


    commonANDdeepest_root_foralltxrx  = dsfile_with_meas_root[timestamp_leaf]
    needed_attrs = commonANDdeepest_root_foralltxrx.attrs 
    
    do_psd_storing(needed_attrs, commonANDdeepest_root_foralltxrx, leaves_withTX if args.whichbranch =='wtx' else leaves_withoutTX)


def parse_args_def():
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--dircommon",  type=str, default=DEF_DATADIR, help="The directory in which resides one or more folders that each have one data file. Default: %s" % DEF_DATADIR)
    parser.add_argument("-d", "--datadir",    type=str, default=DEF_DATADIR, help="The name of the HDF5 format data file. Default: %s" % DEF_DATADIR)
    parser.add_argument("-f", "--dfname",     type=str, default=DEF_DFNAME, help="The name of the HDF5 format data file. Default: %s" % DEF_DFNAME)
    parser.add_argument("-b", "--whichbranch",type=str, default=DEF_BRANCH, help="Must specify which branch, wtx or wotx, to plot. Default: %s" % DEF_BRANCH)
    parser.add_argument("-s", "--saveplots",  type=str, default="False", help="Whether or not to draw and save the plots. Resource consuming so may cause terminal to break")
    
    return parser #not parsing them here, so not doing parser.parse_args()!

if __name__ == "__main__":

    parser_itself = parse_args_def()

    args_old = parser_itself.parse_args() # parsing them here!

    for f in sorted(Path(args_old.dircommon).iterdir()):   
        if f.is_dir():
            args = parser_itself.parse_args(['--datadir', str(f), '--saveplots',  args_old.saveplots ]) #update the argument here!            
            mainn(args)

