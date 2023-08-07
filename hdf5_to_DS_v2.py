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


# DEF_DATADIR="./mcondata"

DEF_DATADIR="./"
DEF_DFNAME="measurements.hdf5"
DEF_BRANCH="wtx"
MEAS_ROOT="saveiq_w_tx"


###############################################################
###############################################################    



def meridian_convergence(lat1, lat2):
    delta_lat = lat2 - lat1 #Calculate the differences in latitude and longitude between the two points.
    delta_lon = lon2 - lon1 #Calculate the differences in latitude and longitude between the two points.
    angle = arccos(sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(lon2 - lon1))  #Calculate the angle using the formula for the spherical law of cosines: where arccos is the inverse cosine function.
    degrees = angle * 180 / pi #Convert the angle from radians to degrees if desired.
    # radians = degrees * pi / 180 # Convert the latitude and longitude of each point from degrees to radians. This can be done using the formula:

    return degrees

def meridian_convergence(direction_w_TN):
    '''
    lat_y  =  labels.iloc[ii][-1][0]
    long_x =  labels.iloc[ii][-1][1]
    ax_ori.scatter( long_x, lat_y, c='C{}'.format(i), marker='o', s=2)

    # Thus:
    #   longitude:  x axis
    #   latitude:   y axis


    # lati
    # | 
    # |
    # |
    # |
    # |_________________ longi



    # Convert to UTM easting and northing!!
    # (lat=y, lon=x) --->>>>UTM ----->>> easting=x, northing=y !! THIS IS WHERE IT FLIPS!
    
    # northing
    # | 
    # |
    # |
    # |
    # |_________________ easting

     
    # the UTM x-axis is referred to as easting, and the UTM y-axis is referred to as northing

    # Thus:
    #   easting:  x axis
    #   northing:   y axis

    easting_x  = labels.iloc[ii][-1][0], 
    northing_y = labels.iloc[ii][-1][1]
    ax_ori.scatter( easting_x, northing_y, c='C{}'.format(i), marker='o', s=2)

    '''

    # THE GRID NORTH 
    # CA = (long_yourpoint - long_CM) × sin (lat_yourpoint)
    direction_w_GN = direction_w_TN
    return direction_w_GN

def calcDistLatLong(coord1, coord2):
    # pdb.set_trace()
    R = 6373000.0 # approximate radius of earth in meters

    lat1 = math.radians(coord1[0])#.values[0])
    lon1 = math.radians(coord1[1])#.values[0])
    lat2 = math.radians(coord2[0])
    lon2 = math.radians(coord2[1])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a    = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    dist = R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return dist


def getrmsFreq(dspd,frqs):
    ## power can only be linear not DB
    # right now between blue lines
    # pdb.set_trace()

    # in case of yes plt.psd!
    # (Pdb) dspd/np.sum( dspd) 
    # array([0.14251675, 0.2761585 , 0.58132475])

    # but making it back to normal psd i.e. removing feq division, still gives same ratios!
    # (Pdb) (dspd*220000) / np.sum( dspd*220000)
    # array([0.14251675, 0.2761585 , 0.58132475])

    # SO, yes or no plt.psd to calculate PSD, RMS stays the same!

    total_power = np.sum(dspd)
    fractional_power = dspd/total_power
    # print("fractional_power", fractional_power, "\nfrqs",frqs)
    bar_frqs = np.sum(fractional_power*frqs)
    #RMSdopplerCalculation.png
    rmsFreqval = np.sqrt(np.sum(fractional_power*(frqs-bar_frqs)**2))

    return round(rmsFreqval,2)


def fancy(axx, rmsis_ns_linear, this_speed, residueoffset, distanceis,maxpw,  DS, max_noise_power_db, lenn):
    ypos= max_noise_power_db + 3
    ap = FancyArrowPatch(posA=(25, ypos),posB=(35, ypos),mutation_scale=20, alpha= 0.5, color ='r', arrowstyle='->' if np.sign(residueoffset)+1 else '<-')
    at = AnchoredText( f"RMS:{round(rmsis_ns_linear,2)} hz\n |v|:{round(this_speed.values[0],2)} mps \n fd={round(DS,2)} \n powerthreshold:{round(max_noise_power_db,1)}, \n feqsLEN:{lenn}, \n DIST:{distanceis}, \n maxPW:{maxpw}", prop=dict(size=10), frameon=True,pad=0.9, loc='upper right')
    at.patch.set_boxstyle("round, pad=0.,rounding_size=0.2")
    axx.add_artist(ap)
    axx.add_artist(at)
    return axx


def offset_estimation_souden(this_measurement):
    acf = np.convolve(this_measurement,np.conj(this_measurement)[::-1])  # half = int(len(acf)/2)# var_rxx0 = acf[half]
    m_lags = np.array(range(1,len(this_measurement))) # starting with lag=1 and till N-1

    first_ofsecondhalf = int(len(acf)/2) # includes variance and rest of rxx. len==sizeof input!
    var_rxx0 = acf[first_ofsecondhalf] #half and then first! #R_xx(0) # the autocorrelation at lag 𝜏=0 is simply the variance of the time series.
    m_positive_rxxm  = acf[first_ofsecondhalf+1:]

    m_angles_radians = np.angle(m_positive_rxxm) 
    print("your pi is intact" if max(m_angles_radians)<=np.pi else "YIKES")

    f_est = m_angles_radians * (1/(2*np.pi))
    f_est_weighted = np.divide( f_est, m_lags)

    p_factor = 20
    f_offset_souden = np.sum(f_est_weighted[:p_factor])/p_factor

    # Rx  = toeplitz(acf,np.hstack((acf[0], np.conj(acf[1:])))) #https://www.gaussianwaves.com/2015/05/auto-correlation-matrix-in-matlab/
    return f_offset_souden


def freq_off_alternate(df_allrx, df_allti, gt_loc_df, fsr, numbsamp, wl, thistx):

    bus_frequency_offset_ranges = mep_ppm[thistx[4:]]
    print("freq ranges to see are:", bus_frequency_offset_ranges)

    columns_array = df_allrx.columns.values
    n_endpoints_is_subplots = len(columns_array)

    freqoff_dict = {key: [] for key in columns_array}

    how_many_nonzero_vel = 0

    for n in range(0, len(df_allrx)):
        for p in   range(0, n_endpoints_is_subplots): #    [1]:#     [3]:# 
            this_measr_time = df_allti.iloc[n,p]
            
            if len(this_measr_time)==0:
                print("no measurments for this column", p, "msrmnt=", n)
                break
            this_measr_timeuptoseconds = this_measr_time[0:-7] # NOTE: ground truth is per second but has no resolution for 'seconds'

            matched_row_ingt = gt_loc_df[gt_loc_df['Time (UTC)'].str.contains(this_measr_timeuptoseconds)] 

            if len(matched_row_ingt) ==0:
                print(f'some how bus didnt record {n}th GPS measurment! for this row', n)
                break

            this_speed, expected_doppler, freq_res, THEbin_for_Doppler, direction_w_TN = match(fsr, numbsamp,  matched_row_ingt, this_measr_timeuptoseconds, wl)
            
            if this_speed.values[0]==0: # to ensure only for cases  when there was no motion. 

                this_measurement  = df_allrx.iloc[n,p]

            else:
                how_many_nonzero_vel = how_many_nonzero_vel+1

    
    print("Freq offsets when no motion", freqoff_dict)

    return freqoff_dict


###############################################################
###############################################################    

def get_avg_power(samps):
    return 10.0 * np.log10(np.sum(np.square(np.abs(samps)))/len(samps))


def get_full_DS_spectrum(this_measurement, rate):
    ## full spectrum
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


    return this_speed, expected_doppler,  freq_res, mth_bin_for_expected_doppler_in_psdfreqs_notXaxis, direction_w_TN


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
        print("reading hdf5 of RX:",rx)

        if args.whichbranch == 'wtx':
            # print("in tx leaf")
            ds_samples = allsampsandtime[tx][gtx][grx][rx][samp]
            ds_times = allsampsandtime[tx][gtx][grx][rx][time]
        else:
            print("in WO_TX leaf.. ******************************************** SHOULDNT HAPPEN! ")
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


mep_ppm = {
'4407': [8500,9500],  #feb16
'4603': [6000,7000],  #feb3
'4734': [8500,10000], #feb16, feb14
'6181': [7500,8500],  #jan30
'6183': [6500,7500],  #jan30(3)
# '6183': [1900,5000], #feb9(3), feb6(1), 
}



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

            this_measr_timeuptoseconds = this_measr_time[0:-7] # NOTE: ground truth is per second but has no resolution for 'seconds'

            matched_row_ingt = gt_loc_df[gt_loc_df['Time (UTC)'].str.contains(this_measr_timeuptoseconds)] 

            if len(matched_row_ingt) ==0:
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
            # plt.title('regardless of the speed')
            # plt.show()

            
            if this_speed.values[0]==0: # to ensure only for cases  when there was no motion. 

                this_measurement  = df_allrx.iloc[n,p]

                result_fft_temp, psdlinear, freqs_temp = psdcalc(this_measurement, fsr) #sq_abs_fftshift_fft

                # Weird inteference! I think LO but I did set it = 0.6 times SR ! aaaaa!!!
                if columns_array[p] == 'cbrssdr1-honors-comp':
                    result_fft_temp[freqs_temp > 90000] = result_fft_temp[90000]
                    result_fft_temp[freqs_temp <-90000] = result_fft_temp[90000]


                all_peaks_idxs, all_peaks_vals = scipy.signal.find_peaks(result_fft_temp) 
                idx_psd_max = all_peaks_idxs[np.argmax(result_fft_temp[all_peaks_idxs])] 

                val_psd_max = result_fft_temp[idx_psd_max]
                val_freq_max = freqs_temp[idx_psd_max]  #### THIS IS THE ESTIMATED OFFSET!!!

                # print("freq offset is", val_freq_max, "max power is" , val_psd_max, n, p)
                # print("mean of psd is", np.mean(result_fft_temp), "std is", np.std(result_fft_temp), "3times std of psd is", 3*np.std(result_fft_temp))

                # plt.clf()
                # plt.plot(freqs_temp, result_fft_temp,  label=f"{n}{df_allrx.columns[p][9:12]}\n mean{np.mean(result_fft_temp)}  \n max {val_psd_max}  \n frq {val_freq_max}", color = 'r' if this_speed.values[0] ==0 else 'g')
                # plt.legend(loc='lower left')
                # plt.title('speed = zero')
                # plt.show()

                # threshold = -21 #-23.5 # np.mean(result_fft_temp) + 3*np.std(result_fft_temp)

                if val_psd_max > threshold and val_freq_max < bus_frequency_offset_ranges[1] and val_freq_max > bus_frequency_offset_ranges[0]: # to ensure signal was indeed "seen"
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
    print("Stationary signal wasnt seen for RX", [key for key, value in freqoff_dict.items() if len(value) == 0]) 


    if len([key for key, value in freqoff_dict.items() if len(value) == 0]) != n_endpoints_is_subplots:
        mean_frqoff_perrx_dict = {key: sum(values) / len(values) if len(values) != 0 else None for key, values in freqoff_dict.items()}
        naverage = sum(value for value in mean_frqoff_perrx_dict.values() if value !=None) / len([value for value in mean_frqoff_perrx_dict.values() if value != None])
        mean_frqoff_perrx_dict = {key: ( naverage if value == None else value) for key, value in mean_frqoff_perrx_dict.items()}
    else:
        print("stationary signal never seen at any BS, check your bus_frequency_offset_ranges !")
        exit()
    
    print("MISSED WAITRESS!", no_measr_time_idx_n, no_gps_mesrnt_idx_n)

    return mean_frqoff_perrx_dict, how_many_zero_vel/n_endpoints_is_subplots, how_many_nonzero_vel/n_endpoints_is_subplots, no_measr_time_idx_n, no_gps_mesrnt_idx_n # , speed_time_dict



def do_data_storing(attrs, allsampsandtime, leaves):
    
    """
    For each data_dir, this function stores one pickle file that has two dictionaries.

    1. Dict#1 has 
        a. complex-valued spectrum ( under the key as each basestation name) of the received signal that has been low-pass filtered, approximate-frequency-offset removed, and the section of spectrum narrowed to max_plus_buffer_expected_Doppler_frequency.
        b. corresponding to the matching timstamps of each signal reception, in the last key named 'speed_postuple' respective labels are stored as one tuple (that is, transmitter speed, transmitter track, (transmitter latitude, transmitter laongitude) , matching timestamp of every signal recording).  

    

    Dict#1 format:

    {                   basestation_1              basestation_2                 .....            basestation_M                             speed_postuple        
    
    value_1 :       [complex_value_spectrum11]   [complex_value_spectrum12]                [complex_value_spectrum1M]       ( speed_r1, track_r1, (lat_r1, lon_r1), timestamp_r1)
      
    .                      .                               .                                           .                                     .                              
    .                      .                               .                                           .                                     .                               
    .                      .                               .                                           .                                     .                               
    
    value_N :              .                               .                                           .                                     .                                      


    }                 



    2. Dict#2 has metadata of this entire experimet data_dir. ( name of transmitter, number of samples collected, sampling rate, center frequency, limit set for max_plus_buffer_expected_Doppler_frequency, timestamp of the start of the experiment,) 

     
    Dict#1 format:
    {
    "tx" : txis,
    "nsamps" : nsamps,
    "rate" : rate, 
    "tx_CENTER_FREQUENCY" : tx_CENTER_FREQUENCY,
    "narrow_spectrum":ns,
    "exp_datetime" : datetime.datetime.fromtimestamp(int(exp_timestamp)).astimezone( pytz.timezone("America/Denver")) #.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    """



    ####GT#############################
    print('Current working directory is:', os.getcwd(), "\n")
    loc_files = list(Path(args.datadir).rglob('*.csv'))
    print('Location file to read is:', loc_files, "\n")
    
    if len(loc_files) == 0:
        print(" No BUS Location csv file present!!!!!!!")
        # exit(1)
    else:
        gt_loc_df = pd.read_csv(loc_files[0], header=0)
        gt_loc_df['Time (UTC)'] = pd.to_datetime(gt_loc_df['Time (UTC)']).dt.tz_convert('US/Mountain').dt.strftime("%Y-%m-%d %H:%M:%S")
        
        # print(" The very first bus position tuple: \n", (gt_loc_df['Lat'].iloc[0], gt_loc_df['Lon'].iloc[0]))
        print(" The max bus speed in this experiment was:", gt_loc_df['Speed (meters/sec)'].max(), 'Overall length of CSV', gt_loc_df.shape, "\n")
    

    ####ATTRS#############################
    
    rate = attrs['rxrate']
    nsamps = attrs['nsamps']

    freq_resolution = rate/nsamps

    tx_CENTER_FREQUENCY  = attrs['txfreq']
    rx_CENTER_FREQUENCY = attrs['rxfreq']
    print('tx and rx frequency <<<MATCHED>>> \n' if (tx_CENTER_FREQUENCY==rx_CENTER_FREQUENCY) else 'tx and rx frequency <<<DONT>>> match')
    
    MAXIMUM_BUS_SPEED_POSSIBLE =  21 #units are meters_per_second!
    print("Hardcoded value:","MAXIMUM_BUS_SPEED_POSSIBLE ==", MAXIMUM_BUS_SPEED_POSSIBLE)

    MINIMUM_BUS = 4.5 # mps !! mph?

    MAXIMUM_SPEED_SEEN = gt_loc_df['Speed (meters/sec)'].max() # +1 #20 #meters per second == 44.7 miles per hour   30 mps == 67.1 mph
    

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





    #### all DATA#############################

    txis, endpoints, df_allrx, df_allti, exp_timestamp = leaves_to_DF(leaves, allsampsandtime)
    bus_frequency_offset_ranges = mep_ppm[txis[4:]]

    print("\nThis experiment's bus is   == ", txis, "\n\nWith preset offset-finding frequency ranges == ",
     bus_frequency_offset_ranges,"\n")


    n_endpoints_is_subplots = len(df_allrx.columns.values)   

    print('Number of total measurements ==', len(df_allrx), "\n") 
    print('The Base Stations are        ==', df_allrx.columns, "\n")






    #### freq offset removal ############################# frqoff_perrx  = {'cbrssdr1-bes-comp': 8596.155802408854, 'cbrssdr1-honors-comp': 8605.904414735991, 'cbrssdr1-hospital-comp': 8606.960155345776, 'cbrssdr1-smt-comp': 8608.430321536847, 'cbrssdr1-ustar-comp': 8585.357666015625}
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
    data_and_label_dict["speed_postuple"]   = []


    runtime = f'{int(time.time())}' 
    print("runtime", runtime)








    for n in range(0, len(df_allrx) ):

        if n in no_measr_time_idx_n or n in no_gps_mesrnt_idx_n:
            # print(f"skipping for {n}th measurment as it is missing!")
            continue

        # fig, axs = plt.subplots(3, n_endpoints_is_subplots, figsize=(16, 7), sharey= False)  # CREATE THE FIGURE OUTSIDE the loop!
        

        # print('for nth measurement = ', n) 

        for p in range(0, n_endpoints_is_subplots):    #[4]   [0]:#   [1]:#     
            # print('pth column', p)            
             
            this_measr_time = df_allti.iloc[n,p]  

            ## Time - will be pickling!
            this_measr_timeuptoseconds = this_measr_time[0:-7]                                                  # NOTE: ground truth is per second but has no resolution for 'subseconds'

            ## Get respective row from GT_LOC_CSV            
            matched_row_ingt = gt_loc_df[gt_loc_df['Time (UTC)'].str.contains(this_measr_timeuptoseconds)]           


            if bus_go_vroom_flag: # at least wont check for those days when bus didnt run faster!
                speed_index = matched_row_ingt.index.values[0]
                if speed_index in bus_go_vroom_indexes_gt_df:
                    print(matched_row_ingt)
                    print(" this index is in high speed indexes, so skipping this row!!! \n\n")
                    break



            # Location - will be pickling!
            current_bus_pos_tuple = (matched_row_ingt['Lat'].values[0],matched_row_ingt['Lon'].values[0])

            # Speed  - will be pickling!
            this_speed, expected_doppler, freq_res, THEbin_for_Doppler, gps_track = match(rate, nsamps,  matched_row_ingt, this_measr_timeuptoseconds, WAVELENGTH)
                                                                                                                # direction_GN= meridian_convergence(direction_w_TN)
            fdmaxis = expected_doppler.values[0]                                                                # correctly named as 'max' # print('gps \'speed\' was the magnitude of velocity vector. speed*anycos(theta) will be <= speed. So this \'speed\' gives the maximum doppler possible from this \'speed\'')
                                                                                                                # Following 2 lines are WORNG CAUSE IT IS REDUNDANT . It is the same ting as butDSwillbeseenatpsd_is_freq_right                                                                                                                # but_fdmax_willbeseenatpsd_is_freq_right = int(np.ceil(fdmaxis/freq_res))*freq_res  #cause of the fft resoltuon                                                                                                                # # but_fdmax_willbeseenatpsd_is_freq_left = int(np.floor(-fdmaxis/freq_res))*freq_res  # here it should be np.floor but no need to do separately as a simple -ve sign is equal to making floor
            butDSwillbeseenatpsd_is_freq_right = THEbin_for_Doppler*freq_res                                    # cause of the fft resoltuon# also it is already,ceil cause positive
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


            ############## if storing spectrum, not PSD  ###################            
            ## new ! The full "complex-valued spectrum", not psd! # separately done than psdcalc! 
            [full_spectrum, fxxc] = get_full_DS_spectrum(corrected_signal_withfdANDfresidue, rate)
            
            ## narrowspectrumed here!
            ns_freq_res = int(np.ceil(ns/ freq_res))*freq_res
            fdidx  = (fxxc >= -ns_freq_res) & ( fxxc <= ns_freq_res)             
            fxxc_ns = fxxc[fdidx]
            full_spectrum_narrowed = full_spectrum[fdidx]
            # psd_cvds_narrowed = np.nan_to_num(10.0 * np.log10(np.square(np.abs(full_spectrum_narrowed)) ))
            # plt.plot(fxxc_ns, psd_cvds_narrowed)

            ## pickling here!
            data_and_label_dict[df_allrx.columns[p]].append(full_spectrum_narrowed) 
            ################################################################

            ## OR 

            # ############## if storing PSD, not spectrum  ###################
            # # get the PSD not spectrum
            # [pxxc_linear, fxxc] = plt.psd(corrected_signal_withfdANDfresidue, NFFT= nsamps, Fs = rate)          ## pxxc_DB, pxxc_linear, fxxc = psdcalc(corrected_signal, rate)
            # pxxc_DB = 10.0 * np.log10(pxxc_linear)

            # ## narrowspectrumed here!          
            # ns_freq_res = int(np.ceil(ns/ freq_res))*freq_res
            # fdidx  = (fxxc >= -ns_freq_res) & ( fxxc <= ns_freq_res)            
            # fxxc_ns = fxxc[fdidx]
            # pxxc_ns_DB = pxxc_DB[fdidx]
            # pxxc_ns_linear = pxxc_linear[fdidx]  # - will be pickling!

            # ## pickling here!
            # data_and_label_dict[df_allrx.columns[p]].append(pxxc_ns_linear)            # print(len( psd_dict[df_allrx.columns[p]][0]))
            # ################################################################



            if p==n_endpoints_is_subplots-1: # when at the last Rx's index, that is only after the last BS has been picked, we store loc, speed, track, and time in their respective columns
                # print("labelling here! ")
                data_and_label_dict["speed_postuple"].append([this_speed.values[0], gps_track, current_bus_pos_tuple, this_measr_timeuptoseconds])

        # print(f'\n\n{n}th measurment done\n\n\n')
        
    

    print(len(fxxc_ns))
    
    # ##git
    fn = f"{args.dircommon}"+"/"+ f"{args.datadir}".split('/')[-1]+f"_{runtime}"+'.pickle'
    
    ## solo
    # fn = f"{args.datadir}".split('/')[-1]+ f"_{runtime}"+'.pickle'

    pkl.dump((data_and_label_dict, metadata_dict), open(fn, 'wb' ) )
    print("pickled!\n\n\n\n")
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
        leaves_withTX = all_leaves_2[0:int(num_leaves/2)]
        leaves_withoutTX = all_leaves_2[int(num_leaves/2):]


    commonANDdeepest_root_foralltxrx  = dsfile_with_meas_root[timestamp_leaf]
    needed_attrs = commonANDdeepest_root_foralltxrx.attrs

    do_data_storing(needed_attrs, commonANDdeepest_root_foralltxrx, leaves_withTX if args.whichbranch =='wtx' else leaves_withoutTX)# leaves_withoutTX) #all_leaves_2) # leaves_withTX)


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
