from libs import * 

###### manual fixing for the outliers in CFOs! Should be automated with the find_peaks to remove measurements with low snr and interference !
lowerlimit_purple_5perfect_3bads_dict = {
# "01-30-2023_15-40-55":["D1",   8000, 5600],
"02-09-2023_15-08-10":["D10",  3000, 0],
"02-09-2023_17-12-28":["D11", 10000, 1775],
"02-03-2023_12-55-47":["D7",   8000, 5600],
"02-14-2023_10-45-17":["D13", 10000, 6200],
"02-14-2023_12-48-02":["D14", 10000, 6100],
"02-14-2023_14-49-21":["D15", 10000, 8300],
"02-16-2023_16-59-03":["D21", 10000, 5000],
}


######################################## Get spectrum of iqs ###
################################################################

def get_full_DS_spectrum(this_measurement, rate):   
    
    """
    Description: Low pass filter the signal to have frequencies only between +/- fcutoff!
    Requirement: 
    Input:
    Output:
    """

    this_measurement     = this_measurement - np.mean(this_measurement)
    nsamps               = len(this_measurement)

    window               = np.hamming(nsamps)
    result               = np.multiply(window, this_measurement)
    result_fft           = np.fft.fft(result, nsamps)
    result_fft_shifted   = np.fft.fftshift(result_fft)

    freqs                = np.fft.fftshift(np.fft.fftfreq(nsamps, 1/rate))

    return result_fft_shifted, freqs

##############################################################
                        # CFO removal
###############################################################

## calculate cfo
def fit_freq_on_time(tt,ff, deg):
    '''
    '''
    coeff = np.polyfit(tt,ff, deg) # nth dgree fit. Retuns in decreasin order of powers! = [deg, deg-1, … ,0 ]; len(coeff) = deg+1
    # coeff = PNOM.fit(tt,ff, 2)
    # coeff = Cheby.fit(tt,ff, 2)
    return coeff

def mylpf(this_measurement, fsr, fc): 
    
    """
    Description: Low pass filter the signal to have frequencies only between +/- fcutoff!
    Requirement: 
    Input:
    Output:
    """

    numtaps_forfilterout_weird_signal  = 1001
    LPF                                = sig.firwin(numtaps_forfilterout_weird_signal, cutoff=fc, fs=fsr) #has defulatwindow='hamming'    
    # print("hardcoded value: fc =", fc)
    res = sig.convolve(this_measurement, LPF, mode='same')
    return res
    
def get_cfo(fn, df_allrx, df_allti, gt_loc_df, fsr, lpf_fc, exp_start_timestampUTC, degreeforfitting, pwr_threshold):
    '''
    '''
    print("\n\nCalculating the cfo.... ")
    columns_names_array     = df_allrx.columns.values
    n_endpoints_is_subplots = len(columns_names_array)
    n_total_measurements    = len(df_allrx)

    ##########################

    freqoff_dict       = {key: [] for key in columns_names_array} #1
    mean_frqoff_perrx_dict = {} # will require .update() cause no keys are named #2
    medn_frqoff_perrx_dict = {}

    freqoff_time_dict  = {key: [] for key in columns_names_array} # will NOT require .update() cause  keys are named #3
    # freqoff_dist_dict  = {key: [] for key in columns_names_array} # will NOT require .update() cause  keys are named #4
    
    fitd_frqoff_perrx_dict = {} # will require .update() cause no keys are named #5
    

    how_many_zero_vel = 0
    how_many_nonzero_vel = 0

    no_measr_time_idx_n = []
    no_gps_mesrnt_idx_n = []

    ##########################
    # plt.ion()

    for n in range(0, n_total_measurements):
        for p in range(0, n_endpoints_is_subplots): #    [1]:#     [3]:# 
            # print("np is", n, p)

            this_measr_time = df_allti.iloc[n,p]
            
            ##########################
            if len(this_measr_time) == 0:
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

                break # breaking will skip the pth column. But that would make the labelling also not happen if in storing loop!! Break: exit the loop Continue: skip the iteration

            this_measr_timeuptoseconds = this_measr_time[0:-7] # NOTE: ground truth is per second but has no resolution for 'seconds'

            if not isinstance(this_measr_timeuptoseconds, str):
                this_measr_timeuptoseconds = this_measr_timeuptoseconds.decode('utf-8')

            matched_row_ingt = gt_loc_df[gt_loc_df['Time (UTC)'].str.contains(this_measr_timeuptoseconds)] 

            
            ##########################
            if len(matched_row_ingt) == 0:
                # print(f'somehow bus didnt record {n}th GPS measurment! for this row', n)
                no_gps_mesrnt_idx_n.append(n)

                '''
                example     : Shout_meas_01-30-2023_19-53-09
                nth indexs  : 85th ()

                '''              
                break

            this_speed = matched_row_ingt['Speed (meters/sec)']


            ##########################
            if this_speed.values[0] == 0: # to ensure only for cases  when there was no motion. 
                
                this_measurement  = df_allrx.iloc[n,p]
                this_measurement  = mylpf(this_measurement, fsr, lpf_fc)

                result_fft,      freqs = get_full_DS_spectrum(this_measurement, fsr)
                result_fft_db          = np.nan_to_num(10.0 * np.log10(np.square(np.abs(result_fft))))  #sq_abs_fftshift_fft
                
                listofidxsofresult, _  = find_peaks(result_fft_db)#, threshold=5)#, prominence=30)   #Returns:Indices of peaks in x that satisfy all given conditions. # (Pdb) idxs[[0,-1]]# array([     2, 131070])


                # if len(listofidxsofresult_lin) >=200:
                if len(listofidxsofresult) ==0: 
                    print("no peak detected!!!")
                    break

                idx_ofmaxpsd_but_idx_in_list=np.argmax(result_fft_db[listofidxsofresult])
                idx_maxpsd_in_result=listofidxsofresult[idx_ofmaxpsd_but_idx_in_list]

                val_psd_max                 = result_fft_db[idx_maxpsd_in_result]
                val_freq_max                = freqs[idx_maxpsd_in_result]  #### THIS IS THE APPROAX OFFSET!!!

                lowerlimitspectrum = lowerlimit_purple_5perfect_3bads_dict[fn][2] if  fn in lowerlimit_purple_5perfect_3bads_dict else 0
                upperlimitspectrum = lowerlimit_purple_5perfect_3bads_dict[fn][1] if  fn in lowerlimit_purple_5perfect_3bads_dict else lpf_fc

                # to ensure signal was indeed "seen"
                if val_psd_max > pwr_threshold and val_freq_max < upperlimitspectrum and val_freq_max > lowerlimitspectrum: 
                # manually fixing for 5 fixables
                # if val_psd_max > pwr_threshold and val_freq_max < 8000 and val_freq_max > 5600:  # D7:  02-03-2023_12-55-47
                # if val_psd_max > pwr_threshold and val_freq_max < 10000 and val_freq_max > 6200: # D13: 02-14-2023_10-45-17
                # if val_psd_max > pwr_threshold and val_freq_max < 10000 and val_freq_max > 5600:  # D14: 02-14-2023_12-48-02
                # if val_psd_max > pwr_threshold and val_freq_max < 10000 and val_freq_max > 8300:  # D15: 02-14-2023_14-49-21
                # if val_psd_max > pwr_threshold and val_freq_max < 10000 and val_freq_max > 5000:  # D21: 02-16-2023_16-59-03


                    freqoff_dict[df_allrx.columns[p]].append(val_freq_max)
                    freqoff_time_dict[df_allrx.columns[p]].append([val_freq_max, this_measr_timeuptoseconds])
                    # freqoff_dist_dict[df_allrx.columns[p]].append([val_freq_max, calcDistLatLong(  all_BS_coords[columns_names_array[p].split('-')[1]] ,  matched_row_ingt.iloc[0][3:5]  )])
                    # print(n, p, "val_psd_max" , val_psd_max)  if p<2 else ''

                how_many_zero_vel = how_many_zero_vel+1
        
            else:
                how_many_nonzero_vel = how_many_nonzero_vel+1

    # plt.ioff()
    # plt.close()


    n_stationary_msrmnts = how_many_zero_vel/n_endpoints_is_subplots
    n_moving_msrmnts     = how_many_nonzero_vel/n_endpoints_is_subplots
    
    print("\nNumber of stationary measurements     ==", n_stationary_msrmnts)
    print("Number of moving measurements         ==", n_moving_msrmnts)
    print("Unique indexes to be missed are:", len(np.unique(no_measr_time_idx_n + no_gps_mesrnt_idx_n)))

    print("\nMISSED measurements!\n","1. due to missed TIME or waitres", no_measr_time_idx_n, "\n2. due to missed GPS", no_gps_mesrnt_idx_n)    
    print("\nStationary signal seen per RX", [len(value) for key, value in freqoff_dict.items() ]) 
    print("Stationary signal wasnt seen for RX", [key for key, value in freqoff_dict.items() if len(value) == 0]) 
    # print("Freq offsets when no motion", freqoff_dict,"\n")


    # print("if lpf limits and not not per bus, noisy peaks seen so more in number but very different values", [kvp[1] for ii, kvp in enumerate(freqoff_dict.items()) if ii<2],"\n")
    # print("if lpf limits and not not per bus, noisy peaks seen so more in number but very different values", [len(kvp[1]) for ii, kvp in enumerate(freqoff_dict.items()) if ii<2],"\n")
    

    """
    Doing 2 methods for freq offset removal! 
    One: polyfit. 
    Two: Averaging. 
    After that, first check the returned polyfit_dict and if any BS_key is NONE, then take that BS's offset from the average_dict?
  

    METHOD 1: 
    if doing polyfit method on frequencies WITH time so using freqoff_time_dict!!!! 
    for cases when a BS didnt see any stationary signal even once, no polytfit is performed and foff==None for now!"""

    if len([key for key, value in freqoff_time_dict.items() if len(value) == 0]) != n_endpoints_is_subplots:
        for one_key, value_all_tuples in freqoff_time_dict.items():
            # print("polyfit cfo estimating for",one_key)
            if len(value_all_tuples) !=0:
                tts=[]
                ffs=[]
                for each_tuple in value_all_tuples:
                    
                    delta_t_since_start = convert_strptime_to_currUTCtsdelta(each_tuple[1], exp_start_timestampUTC)
                    tts.append(   delta_t_since_start  )
                    ffs.append(each_tuple[0])
                
                fitd_frqoff_perrx_dict.update({one_key: fit_freq_on_time(tts,ffs, degreeforfitting)})

            else:
                fitd_frqoff_perrx_dict.update({one_key: None}) 
                print("POLYFIT METHOD: Stationary signal wasnt seen for the RX", one_key, "so, no fitted freq offset obtained! How will you extrapolate foffset then?\
                 \nEither change the powerthreshold or take the avereage method's solutionm(doing latter one for now)")
                # exit()
    
    else:
        print("stationary signal never seen at any BS, check your bus_frequency_offset_ranges !")
        exit()  


     
    # """ METHOD_2:
    # if doing averaging method on frequencies WITH time so using freqoff_time_dict!!!! 
    # for cases when a BS didnt see any stationary signal even once, average of other BS is taken!
    # """
    
    # if len([key for key, value in freqoff_time_dict.items() if len(value) == 0]) != n_endpoints_is_subplots:
    #     for key, value_all_tuples in freqoff_time_dict.items():
    #         if len(value_all_tuples) != 0:
    #             value_list=[]
    #             for each_tuple in value_all_tuples:
    #                 value_list.append(each_tuple[0] )
    #             mean_frqoff_perrx_dict.update({key: sum(value_list) / len(value_all_tuples) })
    #         else:
    #             mean_frqoff_perrx_dict.update({key: None})

    #     naverage = sum(value for value in mean_frqoff_perrx_dict.values() if value !=None) / len([value for value in mean_frqoff_perrx_dict.values() if value != None])
    #     mean_frqoff_perrx_dict = {key: ( naverage if value == None else value) for key, value in mean_frqoff_perrx_dict.items()}
    
    # else:
    #     print("stationary signal never seen at any BS, check your bus_frequency_offset_ranges !")
    #     exit() 


    """ METHOD_3: should be the same results as METHOD_2!!
    if doing the simple averaging method but on frequencies without time so using freqoff_dict!!
    for cases when a BS didnt see any stationary signal even once, average of other BS is taken!
    """

    if len([key for key, value in freqoff_dict.items() if len(value) == 0]) != n_endpoints_is_subplots:
        mean_frqoff_perrx_dict = {key: sum(values) / len(values) if len(values) != 0 else None for key, values in freqoff_dict.items()}
        naverage = sum(value for value in mean_frqoff_perrx_dict.values() if value !=None) / len([value for value in mean_frqoff_perrx_dict.values() if value != None])
        mean_frqoff_perrx_dict = {key: ( naverage if value == None else value) for key, value in mean_frqoff_perrx_dict.items()}
        medn_frqoff_perrx_dict = {key: np.median(values) if len(values) != 0 else None for key, values in freqoff_dict.items()}

    else:
        print("stationary signal never seen at any BS, check your bus_frequency_offset_ranges !")
        exit()

    ##########################


    print("\n\nMethod0: Median freq offset  ==", medn_frqoff_perrx_dict, "\n")
    print("Method1: Avg freq offset     ==", mean_frqoff_perrx_dict, "\n")
    print(f"Method2: Polyfit parameters fr degree = {degreeforfitting} are ==\n", fitd_frqoff_perrx_dict, "\n")
    


    summary_cfo_dict = {
    'exp_start_timestampUTC': exp_start_timestampUTC,
    'meanmethod': mean_frqoff_perrx_dict, 
    'fitdmethod': fitd_frqoff_perrx_dict, 
    'allcfotime': freqoff_time_dict
    }
    print("......done getting the CFO\n\n")
    # plot_all_off_dictionaries(ff, f"{args.dirdata}".split('meas_')[1], cfo_summary_dict, "./", f'{int(time.time())}')
    
    return summary_cfo_dict, no_measr_time_idx_n, no_gps_mesrnt_idx_n 

## remove cfo
def convert_strptime_to_currUTCtsdelta(currstrptime, expstartUTCts):
    '''
    '''
    curr_utc_timestamp  = int(pytz.timezone("US/Mountain").localize(datetime.strptime(currstrptime, "%Y-%m-%d %H:%M:%S")).astimezone(pytz.timezone("UTC")).timestamp())
    delta_t_since_start = curr_utc_timestamp - int(expstartUTCts)
    return delta_t_since_start

def get_interpolated_foff(coef, at_this_time):
    '''
    '''
    polyn = np.poly1d(coef)
    interpolated_freq = polyn(np.array([at_this_time]))
    return interpolated_freq

def do_cfo_removal(summary_cfo_dict, degreeforfitting, this_measr_timeuptoseconds, this_bs, this_measurement, rate):
    '''
    '''

    fitd_feqoff_perrx_dict   = summary_cfo_dict['fitdmethod']
    
    mean_frqoff_perrx_dict   = summary_cfo_dict['meanmethod']
    
    frqoff_time_perrx_dict   = summary_cfo_dict['allcfotime']

    exp_start_timestampUTC   = summary_cfo_dict['exp_start_timestampUTC']

    delta_t_since_start      = convert_strptime_to_currUTCtsdelta(this_measr_timeuptoseconds, exp_start_timestampUTC)
    
    if (np.all(fitd_feqoff_perrx_dict[this_bs]!=None)) and (len(frqoff_time_perrx_dict[this_bs]) >= degreeforfitting+1): # that is, if fit hsppened and was proper, that is len(coeff) = deg+1
        # print( 'Polyfit', len(frqoff_time_perrx_dict[this_bs]))
        method               = 'Polyfit'   ## Frequency Correction METHOD 1: polyfit!
        foffset_approximated = get_interpolated_foff(fitd_feqoff_perrx_dict[this_bs], delta_t_since_start)[0]
    else:
        method               = 'Mean'      ## Frequency Correction METHOD 3 or 2: averaging
        # print( 'Averaging method', this_bs)
        foffset_approximated = mean_frqoff_perrx_dict[this_bs]                                            
    
    freq_res                 = rate/len(this_measurement)
    foffset_approximated     = int(np.ceil(foffset_approximated/freq_res))*freq_res # fix as per the resolutionv # It, though, was average of freqs that came from fft_res, but averaging removed the fft_res effect, so need to do that again in the next line
    Ts                       = 1/rate                                                                                         # calc sample period i.e. how long does one sample take to get collected
    NTs                      = len(this_measurement)* Ts
    t                        = np.arange(0, NTs, Ts)                                                                       # for N samples worth of duration, create the time vector, with a resolution =1/fs
    res_cfo_rmvd_signal      = this_measurement * np.exp(-1j*2*np.pi*foffset_approximated*t)               

    return res_cfo_rmvd_signal, foffset_approximated, method


