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

import pytz
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


def get_full_DS_spectrum(this_measurement, rate):
    nsamps               = len(this_measurement)
    freqs                = np.fft.fftshift(np.fft.fftfreq(nsamps, 1/rate))
    
    this_measurement     = this_measurement - np.mean(this_measurement)

    window               = np.hamming(nsamps)
    result               = np.multiply(window, this_measurement)
    result_fft           = np.fft.fft(result, nsamps)
    
    result_fft_shifted   = np.fft.fftshift(result_fft)

    # result_shifted_magsq = np.square(np.abs(result_fft_shifted)) # abs = sqrt(i^2+q^2)
    
    # psd_is_linear        =  np.nan_to_num(result_shifted_magsq)
    # psd_is_db            =  np.nan_to_num(10.0 * np.log10(result_shifted_magsq))

    return result_fft_shifted, freqs



###############################################################
###############################################################    

def psdcalc(this_measurement, rate):
    ##PSD
    nsamps=len(this_measurement)
    freqs = np.fft.fftshift(np.fft.fftfreq(nsamps, 1/rate))
    
    this_measurement = this_measurement - np.mean(this_measurement)

    window = np.hamming(nsamps)
    result = np.multiply(window, this_measurement)
    result_fft = np.fft.fft(result, nsamps)     # result_magsq = np.square(np.abs(result))
    
    result_fft_shifted = np.fft.fftshift(result_fft)
    result_shifted_magsq = np.square(np.abs(result_fft_shifted)) # abs = sqrt(i^2+q^2)
    
    psd_is_linear =  np.nan_to_num(result_shifted_magsq)
    psd_is_db =  np.nan_to_num(10.0 * np.log10(result_shifted_magsq))

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

    direction_w_TN = matched_row_ingt['Track'].values[0] if any(colname == 'Track' for colname in matched_row_ingt.columns.values) else 0

    # if len(matched_row_ingt.columns) == 6:
    #     direction_w_TN = matched_row_ingt['Track'] # only the new experiments have the Track data in an additional column of the GT_LOC_CSV
    # else:
    #     direction_w_TN = 0



    return this_speed, expected_doppler,  freq_res, mth_bin_for_expected_doppler_in_psdfreqs_notXaxis, direction_w_TN



###############################################################
###############################################################    

def leaves_to_DF(leaves, allsampsandtime):
    # print('h5dump leaves is \n', leaves)
    endpoints=[]
    df_allrx= pd.DataFrame()
    df_allti = pd.DataFrame()

    exp_timestamp = leaves[0].split('/')[0] # is same for all the k leaves!
    
    for k in range(0,len(leaves),2): 
        
        tx =  leaves[k].split('/')[1]
        gtx = leaves[k].split('/')[2] if args.whichbranch == 'wtx' else 'noTXhappened'
        grx = leaves[k].split('/')[3] if args.whichbranch == 'wtx' else leaves[k].split('/')[2]
        rx =  leaves[k].split('/')[4] if args.whichbranch == 'wtx' else leaves[k].split('/')[3]
        samp = leaves[k].split('/')[5] if args.whichbranch == 'wtx' else leaves[k].split('/')[4] 
        time =  leaves[k+1].split('/')[5] if args.whichbranch == 'wtx' else leaves[k+1].split('/')[4]

        endpoints.append(rx)
        # print("reading hdf5 of RX:",rx)

        if args.whichbranch == 'wtx':
            # print("in tx leaf")
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

    return tx, endpoints, df_allrx, df_allti, exp_timestamp




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
    
    threshold = -21 
    print("Hardcoded value:","threshold ==", threshold)


    no_measr_time_idx_n = []
    no_gps_mesrnt_idx_n = []


    # plt.ion()
    for n in range(0, len(df_allrx)):
        for p in   range(0, n_endpoints_is_subplots): #    [1]:#     [3]:# 
            
            this_measr_time = df_allti.iloc[n,p]

            if len(this_measr_time)==0:
                # print("AGAIN: no iq data rxd at this time for this column", p, "msrmnt=", n)

                no_measr_time_idx_n.append(n)

                '''
                example     : Shout_meas_02-14-2023_14-49-21
                nth indexs  : 83,84,85,86 (no measurement collected only for pth = 4. Rest of p columns do have a msrmnt_timestamp.)
                              87 (no measurement collected for any p. So, no associated msrmnt_timestamp at all.)
    
                Note        : Each 'msrmnt_timestamp' in hdf5 is stored as the 'start_time' of nth/current msrmnt as noted by orch per orch's clock, and it is same for msrmnt reported by all the receivers (they must be time-synchd).
                            : OTOH, log files show the missed measurment(s) reported from a rx(s) as noted by orch per orch's clock, so any missed msrmnt time stamp is few seconds later. 
                              Total polling is 11 seconds, so that can be the max difference between waitres_log_file 'missed_timestamp' and the start of the msrmt 'msrmnt_timestamp'.
                '''
                ## when in storing loop, cant un-append the rows in dict if this pth was NOT the first!
                
                ## but neither can assign values like this coz if the gps was not present which is the next condition, you'll still have stored this msrmnt for which no gps_loc is there!
                # data_and_label_dict[df_allrx.columns[p]].append([]) 


                break # breaking will skip the pth column. But that would make the labelling also not happen if in storing loop!!




            # print("np is", n, p)

            this_measr_timeuptoseconds = this_measr_time[0:-7] # NOTE: ground truth is per second but has no resolution further than 'seconds'

            matched_row_ingt = gt_loc_df[gt_loc_df['Time (UTC)'].str.contains(this_measr_timeuptoseconds)] 

            if len(matched_row_ingt) == 0:
                # print(f'somehow bus didnt record {n}th GPS measurment! for this row', n)
                no_gps_mesrnt_idx_n.append(n)

                '''
                example     : Shout_meas_01-30-2023_19-53-09
                nth indexs  : 85th ()

                '''
                
                break


            this_speed, expected_doppler, freq_res, THEbin_for_Doppler, direction_w_TN = match(fsr, numbsamp,  matched_row_ingt, this_measr_timeuptoseconds, wl)
            
            this_measurement  = df_allrx.iloc[n,p]
            psddb, psdlinear, fr = psdcalc(this_measurement, fsr)


            # plt.clf()
            # plt.plot(fr,psddb, label=f"{n}{columns_array[p][9:12]}\n mean{np.mean(psddb)}\n max {psddb[np.argmax(psddb)]} \n frq {fr[np.argmax(psddb)]}\n{this_speed.values[0]}", color = 'g' if this_speed.values[0] !=0 else 'r')
            # plt.legend( loc='lower left')
            # plt.title('regardless of the speed: to check if rx happened and pick range!')
            # plt.draw()
            # plt.pause(0.2)
            # # plt.show()

            
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

                # threshold = -21 #-23.5 # np.mean(result_fft_temp) + 3*np.std(result_fft_temp)

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
    
    # print("Freq offsets when no motion", freqoff_dict)
    print("\nStationary signal was NOT seen AT ALL for the RX\n", [key for key, value in freqoff_dict.items() if len(value) == 0]) 


    if len([key for key, value in freqoff_dict.items() if len(value) == 0]) != n_endpoints_is_subplots:
        mean_frqoff_perrx_dict = {key: sum(values) / len(values) if len(values) != 0 else None for key, values in freqoff_dict.items()}
        naverage = sum(value for value in mean_frqoff_perrx_dict.values() if value !=None) / len([value for value in mean_frqoff_perrx_dict.values() if value != None])
        mean_frqoff_perrx_dict = {key: ( naverage if value == None else value) for key, value in mean_frqoff_perrx_dict.items()}
    else:
        print("\n\n Stationary signal never seen at any BS in the <bus_frequency_offset_ranges> range you specified, check your bus_frequency_offset_ranges !!!!!!\n\n\n")
        exit()
    
    print("MISSED WAITRESS!", no_measr_time_idx_n, no_gps_mesrnt_idx_n)

    return mean_frqoff_perrx_dict, how_many_zero_vel/n_endpoints_is_subplots, how_many_nonzero_vel/n_endpoints_is_subplots, no_measr_time_idx_n, no_gps_mesrnt_idx_n # , speed_time_dict



###############################################################
# All possible receiver/sensor coordinates
###############################################################

all_BS_coords = {
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
'4407': [8500,9500],
'4603': [6000,7000],
'4734': [8500,10000],
'6181': [7500,8500],
'6183': [6500,7500],
# '6183': [1900,5000],
}
# The frequency offset changed based on new day! manually putting the range AFTER running the 'freq_off_averaged_for_full_df'
# '6183': [7000,8000],
# '6183': [6500,7500],
# '4555': [3000,4000],
# '4734': [8500,9500],
# '6182': [7000,8000],
# '6185': [7000,8000],
# '6180': [7000,8000],
# '4604': [6000,7000],
# '4555': [7500,8500],

###############################################################
###############################################################    



def do_data_storing(attrs, allsampsandtime, leaves):
    """
    """
    """
    For each data_dir, this function stores one pickle file that has two dictionaries.

    1. Dict 1 has 
        a. complex-valued spectrum (in one column for each basestation involved) of the received signal that has been low-pass filtered, approximate-frequency-offset removed, and narrowed to expected Doppler frequecies.
        b. corresponding to same timstamps of each signal receptioon, in last 5 columns respective labels (transmitter latitude, transmitter laongitude , transmitter speed, transmitter track, common timestamp of every signal recording) are stored.  

    Dict 1 format:
                        basestation_1              basestation_2                 .....            basestation_M           latitude         longitude           speed      track           timestamp
    row_1 :       [complex_value_spectrum11]   [complex_value_spectrum12]                [complex_value_spectrum1M]       [lat11]           [lon11]          [speed11]   [track11]      [timestamp11]
    row_2 :  
    .                      .                               .                                           .                      .               .                  .           .                .
    .                      .                               .                                           .                      .               .                  .           .                .   
    .                      .                               .                                           .                      .               .                  .           .                .
    row_N :                .                               .                                           .                      .               .                  .           .                .    




    2. Dict 2 has metadata of this entire experimet data_dir. ( name of transmitter, timestamp of the start of the experiment, center frequency, sampling rate, number of smaples collected, expected Doppler frequency ranges) 

    """
    """
    """


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
    
    MAXIMUM_BUS_SPEED_POSSIBLE =  21 #units are meters_per_second!
    print("Hardcoded value:","MAXIMUM_BUS_SPEED_POSSIBLE ==", MAXIMUM_BUS_SPEED_POSSIBLE)

    MAXIMUM_SPEED_SEEN = gt_loc_df['Speed (meters/sec)'].max() #+1 #20 meters per second == 44.7 miles per hour. 30 mps == 67.1 mph
    

    bus_go_vroom_flag = False
    if MAXIMUM_SPEED_SEEN > MAXIMUM_BUS_SPEED_POSSIBLE:
        print("max speed seen", MAXIMUM_SPEED_SEEN)
        bus_go_vroom_flag = True
        print(" \n\n BUS DROVE FASTER THAN YOU ASSUMED! !!!!!! \n\n")     
        bus_go_vroom_indexes_gt_df = gt_loc_df.loc[gt_loc_df['Speed (meters/sec)'] >=MAXIMUM_BUS_SPEED_POSSIBLE].index
        print("gt dataframe indexes of higher speed", np.array(bus_go_vroom_indexes_gt_df))
        # exit()

    WAVELENGTH = scipy.constants.c / tx_CENTER_FREQUENCY 
    FD_MAXPOSSIBLE =  MAXIMUM_BUS_SPEED_POSSIBLE / WAVELENGTH
    print("\nFD_MAXPOSSIBLE is          == " , FD_MAXPOSSIBLE, "\n" )

    ## narrowspectrum (ns)
    ns = FD_MAXPOSSIBLE + 100 #a little more buffer on both sides #hz
    print("Hardcoded value:","Narrowed Spectrum freq_span ==", ns)



    #### get all DATA #############################

    txis, endpoints, df_allrx, df_allti, exp_timestamp = leaves_to_DF(leaves, allsampsandtime)
    bus_frequency_offset_ranges = mep_ppm[txis[4:]]
    
    print("\nThis experiment's bus is   == ", txis, "\n\nWith preset offset-finding frequency ranges == ",
     bus_frequency_offset_ranges,"\n")

    n_endpoints_is_subplots = len(df_allrx.columns.values)   

    print('Number of total measurements ==', len(df_allrx), "\n") 
    print('The Base Stations are        ==', df_allrx.columns, "\n")


    #### get freq offset  ############################# 
    
    frqoff_perrx, numb_stationary_msrmnts, n_moving_msrmnts, no_measr_time_idx_n, no_gps_mesrnt_idx_n  = freq_off_averaged_for_full_df(df_allrx, df_allti, gt_loc_df, rate, nsamps, WAVELENGTH, txis) #, allVT

    print("Number of stationary measurements ==", numb_stationary_msrmnts, "\n")
    print("Number of moving measurements     ==", n_moving_msrmnts, "\n")
    print("Number of missed measurements     ==", len(no_measr_time_idx_n), "\n")
    print("Number of missed ground truth     ==", len(no_gps_mesrnt_idx_n), "\n")
    print("Averaged freq offset is           ==", frqoff_perrx, "\n")

    print("unique indexes to be missed are:", len(np.unique(no_measr_time_idx_n + no_gps_mesrnt_idx_n)))
    


    # store all the ATTRS in metadata_dict #############################
    metadata_dict = {
    "tx" : txis,
    "nsamps" : nsamps,
    "rate" : rate, 
    "tx_CENTER_FREQUENCY" : tx_CENTER_FREQUENCY,
    "narrow_spectrum":ns,
    "exp_datetime" : datetime.datetime.fromtimestamp(int(exp_timestamp)).astimezone( pytz.timezone("America/Denver")) #.strftime("%Y-%m-%d %H:%M:%S")
    }

    # "exp_datetime" : datetime.datetime.strptime(metadata_dict['exp_datetime'], "%Y-%m-%d %H:%M:%S").astimezone( pytz.timezone("America/Denver"))
    

    ##########################data dict################################

    
    data_and_label_dict = {key: [] for key in df_allrx.columns.values}
    data_and_label_dict["position_lat"]   = []
    data_and_label_dict["position_lon"]   = []
    data_and_label_dict["gpsspeed"]       = []
    data_and_label_dict["gpstrack"]       = []
    data_and_label_dict["dt_time"]        = []


    runtime = f'{int(time.time())}' 
    print("runtime", runtime)


    for n in range(0, len(df_allrx) ):        

        if n in no_measr_time_idx_n or n in no_gps_mesrnt_idx_n:
            # print(f"skipping for {n}th measurment as it is missing!")
            continue

        # print('for nth measurement = ', n) 

        for p in range(0, n_endpoints_is_subplots):    
            # print('pth column', p)            
             
            this_measr_time = df_allti.iloc[n,p]  

            ## Time - will be pickling!
            this_measr_timeuptoseconds = this_measr_time[0:-7] # NOTE: ground truth is per second but has no resolution for 'subseconds'                                                
            
            
            ## Get respective row from GT_LOC_CSV
            matched_row_ingt = gt_loc_df[gt_loc_df['Time (UTC)'].str.contains(this_measr_timeuptoseconds)]           
    
            
            if bus_go_vroom_flag: # at least wont check for those days when bus didnt run faster!
                speed_index = matched_row_ingt.index.values[0]
                if speed_index in bus_go_vroom_indexes_gt_df:
                    print(matched_row_ingt)
                    print(" this index is in high speed indexes, so skipping this row!!! \n\n")
                    break


            # Location - will be pickling!
            current_bus_pos_lat = matched_row_ingt['Lat'].values[0]
            current_bus_pos_lon = matched_row_ingt['Lon'].values[0]

            ## Speed - will be pickling!
            this_speed, expected_doppler, freq_res, THEbin_for_Doppler, gps_track = match(rate, nsamps,  matched_row_ingt, this_measr_timeuptoseconds, WAVELENGTH)
            
            fdmaxis = expected_doppler.values[0] # # correctly named as 'max' # print('gps \'speed\' was the magnitude of velocity vector. speed*anycos(theta) will be <= speed. So this \'speed\' gives the maximum doppler possible from this \'speed\'')                                                           
                                                                                                                
            butDSwillbeseenatpsd_is_freq_right = THEbin_for_Doppler*freq_res # cause of the fft resolution# also it is already, ceil cause positive
            # butDSwillbeseenatpsd_is_freq_left =  # here it should be np.floor but no need to do separately as a simple -ve sign is equal to making floor
            
            ## IQ measurements
            this_measurement  = df_allrx.iloc[n,p]

            ## Frequency Correction!
            foffset_approximated = frqoff_perrx[df_allrx.columns[p]]                                            # It, though, was average of freqs that came from fft_res, but averaging removed the fft_res effect, so need to do that again in the next line
            foffset_approximated = int(np.ceil(foffset_approximated/freq_res))*freq_res
            
            Ts = 1/rate                                                                                         # calc sample period i.e. how long does one sample take to get collected
            NTs = len(this_measurement)* Ts
            t = np.arange(0, NTs, Ts)                                                                           # for N samples worth of duration, create the time vector, with a resolution =1/fs
                                                                                                                
            corrected_signal_withfdANDfresidue = this_measurement * np.exp(-1j*2*np.pi*foffset_approximated*t)  # remove most of the offset! But a small residue +-offset will be left!           


            ## LPF the signal to have frequencies only between 25khz!
            if True: # do it for all BS!

                # print("in LPF condition!")

                numtaps_forfilterout_weird_signal  = 1001
                fc                                 = 25000 #hz
                myLPF                              = sig.firwin(numtaps_forfilterout_weird_signal, cutoff=fc, fs=rate)
                # print("hardcoded value:","fc =", fc)

                this_measurement_after_LPF = sig.convolve(this_measurement, myLPF, mode='same')                                       
                xLPF_corrected_signal_withfdANDfresidue = this_measurement_after_LPF * np.exp(-1j*2*np.pi*foffset_approximated*t)               


                ###### assign filtred values back to the old variables for rest of the code (only for honors)
                corrected_signal_withfdANDfresidue = xLPF_corrected_signal_withfdANDfresidue # made sense -0527 that we first LPF then freq correct xlpf. #  also, the rssi of xlpf->thenFC  is same is  rssi of just xlpf (no fc). FCing didnt change power now?


             ## new ! The full "complex-valued spectrum", not psd! # separately done than psdcalc! 
            [full_spectrum, fxxc] = get_full_DS_spectrum(corrected_signal_withfdANDfresidue, rate)

            ## narrowspectrumed here!          
            fdidx  = (fxxc >= -ns) & ( fxxc <= ns)            
            fxxc_ns = fxxc[fdidx]
            full_spectrum_narrowed = full_spectrum[fdidx]
            # psd_cvds_narrowed = np.nan_to_num(10.0 * np.log10(np.square(np.abs(full_spectrum_narrowed)) ))
            # plt.plot(fxxc_ns, psd_cvds_narrowed)

            ## pickling here!
            data_and_label_dict[df_allrx.columns[p]].append(full_spectrum_narrowed) 
            
            if p==n_endpoints_is_subplots-1: # when at the last Rx's index, that is only after the last BS has been picked, we store loc, speed, track, and time in their respective columns
                # print("labelling here! ")
                data_and_label_dict["position_lat"].append(current_bus_pos_lat)
                data_and_label_dict["position_lon"].append(current_bus_pos_lon)
                data_and_label_dict["gpsspeed"].append(this_speed.values[0])
                data_and_label_dict["gpstrack"].append(gps_track)                
                data_and_label_dict["dt_time"].append(this_measr_timeuptoseconds)


        # print(f'\n\n{n}th row measurment done\n\n\n')


    # Path(args.datadir+'/pickledfiles/').mkdir(parents=True, exist_ok=True)
    # pickled_dir = Path(args.datadir+'/pickledfiles/'+runtime)
    # pickled_dir.mkdir(parents=True, exist_ok=True)
    # fn = f"{pickled_dir}"+"/"+ f"{args.datadir}".split('/')[-1]+f"_{runtime}"+'.pickle'
    

    fn = f"{args.dircommon}"+"/"+ f"{args.datadir}".split('/')[-1]+f"_{runtime}"+'.pickle'

    pkl.dump((data_and_label_dict, metadata_dict), open(fn, 'wb' ) )
    print("pickled!")
    loaded_psd = pkl.load(open(fn, 'rb') )
    final_totallength = [len(val) for k, val in enumerate(loaded_psd[0].values()) ]
    print("how many rows we got in this pickled data file", final_totallength )


def get_dataset_keys(f):
    keys = []
    f.visit(lambda key : keys.append(key) if isinstance(f[key], h5py.Dataset) else None)
    return keys

def mainn(args):

    dsfile = h5py.File("%s/%s" % (args.datadir, args.dfname), "r")

    dsfile_with_meas_root = dsfile[MEAS_ROOT]
    all_leaves_2 = get_dataset_keys(dsfile_with_meas_root)

    num_leaves = len(all_leaves_2)
    print( "\n\n\nLength of entire tree is", num_leaves)

    # print("entire Tree without TX and with TX\n",all_leaves_2)
    # [print(i) for i in all_leaves_2]
    
    # if num_leaves < 20:
    #     print("less than 5 BaseStations which is required for current training setup!!")
    #     exit()

    timestamp_leaf = all_leaves_2[0].split('/')[0]
    

    if all_leaves_2[0].split('/')[1] == 'wo_tx':
        print("Only wo_TX data collection happened. There is no branch for w_tx\n You should confirm what you are looking for!!")
        leaves_withoutTX = all_leaves_2
    else:
        leaves_withTX = all_leaves_2[0:int(num_leaves/2)] # first half are tx leaves
        leaves_withoutTX = all_leaves_2[int(num_leaves/2):] # second half are wo_tx leaves


    commonANDdeepest_root_foralltxrx  = dsfile_with_meas_root[timestamp_leaf]
    needed_attrs = commonANDdeepest_root_foralltxrx.attrs 
    
    do_data_storing(needed_attrs, commonANDdeepest_root_foralltxrx, leaves_withTX if args.whichbranch =='wtx' else leaves_withoutTX)


def parse_args_def():
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--dircommon",  type=str, default=DEF_DATADIR, help="The directory in which resides one or more folders that each have one data file. Default: %s" % DEF_DATADIR)
    parser.add_argument("-d", "--datadir",    type=str, default=DEF_DATADIR, help="The name of the HDF5 format data file. Default: %s" % DEF_DATADIR)
    parser.add_argument("-f", "--dfname",     type=str, default=DEF_DFNAME, help="The name of the HDF5 format data file. Default: %s" % DEF_DFNAME)
    parser.add_argument("-b", "--whichbranch",type=str, default=DEF_BRANCH, help="Must specify which branch, wtx or wotx, to plot. Default: %s" % DEF_BRANCH)
    
    return parser #not parsing them here, so not doing parser.parse_args()!

if __name__ == "__main__":

    parser_itself = parse_args_def()

    args_old = parser_itself.parse_args() # parsing them here!

    for f in sorted(Path(args_old.dircommon).iterdir()):   
        if f.is_dir():
            args = parser_itself.parse_args(['--datadir', str(f) ]) #update the argument here!            
            mainn(args)

