from libs import * 

###### manual fixing for the outliers in CFOs! Should be automated with the find_peaks to remove measurements with low snr and interference !
lowerlimit_purple_5perfect_3bads_dict = {
# "01-30-2023_15-40-55":["D1",   8000, 5600],
"02-09-2023_15-08-10":["D10",  3000, 0],
"02-09-2023_17-12-28":["D11", 10000, 1775],
"02-03-2023_12-55-47":["D7",   8000, 5600],
"02-14-2023_10-45-17":["D13", 10000, 6200],
"02-14-2023_12-48-02":["D14", 10000, 6200],
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
    
def get_cfo_either_lin_or_db_pwr(fn, df_allrx, df_allti, gt_loc_df, fsr, lpf_fc, exp_start_timestampUTC, pwr_threshold, degreeforfitting, cfo_mthd, overall_plots_dir):
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

    high_SNR_idx_dict = {key: [] for key in columns_names_array} 
    high_SNR_n_list = []
    
    ##########################
    # plt.ion()

    for n in range(0, n_total_measurements):

        for p in range(0, n_endpoints_is_subplots): #    [1]:#     [3]:# 
            # print("np is", n, p)

            this_measr_time = df_allti.iloc[n,p]
            
            ##########################
            if len(this_measr_time) == 0:
                # print(this_measr_time)
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
                # print(f'AGAIN: somehow bus didnt record {n}th GPS measurment! for this row', n)
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
                result_fft_lin         = np.nan_to_num(np.square(np.abs(result_fft))) 
                result_fft_db          = np.nan_to_num(10.0 * np.log10(result_fft_lin))  #sq_abs_fftshift_fft
                

                """

                # # if 0: #p==1:# and n in to_test:
                #     # plt.clf()
                #     # print(mxpsd, "at", frq_mxpsd) 

                #     print(n, "len=", len(listofidxsofresult_lin) )# ,round(val_freq_max,2), round(val_psd_max,2))
                #     print(f"#Peaks: {len(listofidxsofresult_lin)}, freq offset is", round(val_freq_max,3), "max power is", round(val_psd_max,3), "n=",n, "p=",p)

                #     plt.plot(freqs, result_fft_lin, label=f"{n}{df_allrx.columns[p][9:12]}. frq({val_freq_max})", color = 'r' if this_speed.values[0] ==0 else 'g') #\n mean{np.mean(result_fft_temp_db)} \n max {val_psd_max}  \n
                #     # plt.axhline(y = pwr_threshold, color = 'b', linestyle = '-') 
                #     plt.plot(freqs[listofidxsofresult_lin], result_fft_lin[listofidxsofresult_lin], \
                #         'o', color='g', label=f'number of Peaks: {len(listofidxsofresult_lin)}')
                #     # plt.scatter(val_freq_max, val_psd_max, marker='o', s=100,  color='g', label=f'max at {val_freq_max}')
                #     # plt.ylim(-60, 10)
                #     plt.axhline(y = np.mean(result_fft_lin), color = 'm', linestyle = '-', lw=5); plt.axhline(y = meanwithin_spectrum, color = 'g', linestyle = '-', lw=5) ; plt.axhline(y = pwr_thrs_lin_promienence, color = 'b', linestyle = '-', lw=5) 

                #     plt.xlim(0, 10000)
                #     plt.ylim(0, 0.04)
                #     plt.legend(loc='upper left')
                #     plt.grid(True)
                #     plt.title('speed = zero')
                #     plt.show()


                #     # To regerenrate the plot in pdb

                #     #####  only signal, no peaks!
                #     fig,ax=plt.subplots(num="mine_linear1"); plt.ion(); plt.plot(freqs, result_fft_lin, 'r-', label=f'{n}{df_allrx.columns[p][9:12]}'); plt.legend(loc='upper left'); plt.axhline(y = np.mean(result_fft_lin), color = 'm', linestyle = '--'); plt.axhline(y = meanwithin_spectrum, color = 'g', linestyle = '--' ) ; plt.axhline(y = pwr_thrs_lin_promienence, color = 'b', linestyle = '--'); plt.grid(True); plt.xlim(-lpf_fc, lpf_fc); plt.ylim(0, 0.04); plt.show()
                
                    


                #     ###### signal AND peaks!
                #     fig,ax=plt.subplots(num="mine2")
                #     plt.ion(); plt.plot(freqs, result_fft_lin, 'r-', freqs[listofidxsofresult_lin], result_fft_lin[listofidxsofresult_lin], 'bx', label=f'{n}{df_allrx.columns[p][9:12]}\n#Peaks:{len(listofidxsofresult_lin)}'); plt.legend(loc='upper left'); plt.grid(True); plt.xlim(0, 10000); plt.ylim(0, 0.04); plt.show()
                #     # color black
                #     plt.ion(); plt.plot(freqs, result_fft_lin, 'r-', freqs[listofidxsofresult_lin], result_fft_lin[listofidxsofresult_lin], 'k^', label=f'{n}{df_allrx.columns[p][9:12]}\n#Peaks:{len(listofidxsofresult_lin)}'); plt.legend(loc='upper left'); plt.grid(True); plt.xlim(0, 10000); plt.ylim(0, 0.04); plt.show()
                #     # color yellow
                #     plt.ion(); plt.plot(freqs, result_fft_lin, 'r-', freqs[listofidxsofresult_lin], result_fft_lin[listofidxsofresult_lin], 'y^', label=f'{n}{df_allrx.columns[p][9:12]}\n#Peaks:{len(listofidxsofresult_lin)}'); plt.legend(loc='upper left'); plt.grid(True); plt.xlim(0, 10000); plt.ylim(0, 0.04); plt.show()





                #     ######  To plot any random nth smapl!
                #     fig,ax=plt.subplots(num="mine3")
                #     nn=to_test[1]
                #     listofidxsofresult_lin, _= find_peaks(np.nan_to_num(np.square(np.abs(get_full_DS_spectrum(mylpf(df_allrx.iloc[nn,p], fsr, lpf_fc), fsr)[0]))), threshold=fpk_thr, distance=fpk_dst)                       
                #     ###### color blue
                #     plt.ion(); plt.plot(freqs, np.nan_to_num(np.square(np.abs(get_full_DS_spectrum(mylpf(df_allrx.iloc[nn,p], fsr, lpf_fc), fsr)[0]))), 'r-', freqs[listofidxsofresult_lin], np.nan_to_num(np.square(np.abs(get_full_DS_spectrum(mylpf(df_allrx.iloc[nn,p], fsr, lpf_fc), fsr)[0])))[listofidxsofresult_lin], 'bx', label=f'{nn}{df_allrx.columns[p][9:12]}\n#Peaks: {len(listofidxsofresult_lin)}'); plt.legend(loc='upper left'); plt.grid(True); plt.xlim(0, 10000); plt.ylim(0, 0.04); plt.show()
                #     ######  color green
                #     plt.ion(); plt.plot(freqs, np.nan_to_num(np.square(np.abs(get_full_DS_spectrum(mylpf(df_allrx.iloc[nn,p], fsr, lpf_fc), fsr)[0]))), 'r-', freqs[listofidxsofresult_lin], np.nan_to_num(np.square(np.abs(get_full_DS_spectrum(mylpf(df_allrx.iloc[nn,p], fsr, lpf_fc), fsr)[0])))[listofidxsofresult_lin], 'go', label=f'{nn}{df_allrx.columns[p][9:12]}\n#Peaks: {len(listofidxsofresult_lin)}'); plt.legend(loc='upper left'); plt.grid(True); plt.xlim(0, 10000); plt.ylim(0, 0.04); plt.show()


                
                #     ######  db scale!!! For experiemtn 2023-02-14 21:49:39

                #      left_bases, right_basesndarray: The peaks’ bases as indices in x to the left and right of each peak. The higher base of each pair is a peak’s lowest contour line.!checked this!
                #     
                #     listofidxsofresult_lin, pr=find_peaks(10.0 * np.log10(np.nan_to_num(np.square(np.abs(get_full_DS_spectrum(mylpf(df_allrx.iloc[nn,p], fsr, lpf_fc), fsr)[0])))), prominence=44)

                #     plt.plot(freqs[pr['left_bases']],10.0 * np.log10(np.nan_to_num(np.square(np.abs(get_full_DS_spectrum(mylpf(df_allrx.iloc[nn,p], fsr, lpf_fc), fsr)[0]))))[pr['left_bases']], 'k*')
                #     plt.plot(freqs[pr['right_bases']],10.0 * np.log10(np.nan_to_num(np.square(np.abs(get_full_DS_spectrum(mylpf(df_allrx.iloc[nn,p], fsr, lpf_fc), fsr)[0]))))[pr['right_bases']], 'b*')
                    
                #     print(10.0 * np.log10(np.nan_to_num(np.square(np.abs(get_full_DS_spectrum(mylpf(df_allrx.iloc[nn,p], fsr, lpf_fc), fsr)[0]))))[listofidxsofresult_lin])
                #     print(10.0 * np.log10(np.nan_to_num(np.square(np.abs(get_full_DS_spectrum(mylpf(df_allrx.iloc[nn,p], fsr, lpf_fc), fsr)[0]))))[pr['right_bases']]) #(checked this here!) 
                #     print(10.0 * np.log10(np.nan_to_num(np.square(np.abs(get_full_DS_spectrum(mylpf(df_allrx.iloc[nn,p], fsr, lpf_fc), fsr)[0]))))[pr['right_bases']] + pr['prominences'])
                #     print(10.0 * np.log10(np.nan_to_num(np.square(np.abs(get_full_DS_spectrum(mylpf(df_allrx.iloc[nn,p], fsr, lpf_fc), fsr)[0]))))[pr['right_bases']] + pr['prominences'] == 10.0 * np.log10(np.nan_to_num(np.square(np.abs(get_full_DS_spectrum(mylpf(df_allrx.iloc[nn,p], fsr, lpf_fc), fsr)[0]))))[listofidxsofresult_lin])                    

                #     nn=276
                #     fig,ax=plt.subplots(num="pdb276")
                #     listofidxsofresult_lin, pr=find_peaks(10.0 * np.log10(np.nan_to_num(np.square(np.abs(get_full_DS_spectrum(mylpf(df_allrx.iloc[nn,p], fsr, lpf_fc), fsr)[0])))))
                #     plt.ion(); plt.plot( freqs, 10.0 * np.log10(np.nan_to_num(np.square(np.abs(get_full_DS_spectrum(mylpf(df_allrx.iloc[nn,p], fsr, lpf_fc), fsr)[0])))), 'r-', freqs[listofidxsofresult_lin], 10.0 * np.log10(np.nan_to_num(np.square(np.abs(get_full_DS_spectrum(mylpf(df_allrx.iloc[nn,p], fsr, lpf_fc), fsr)[0]))))[listofidxsofresult_lin], 'go', label=f'{nn}{df_allrx.columns[p][9:12]}\n#Peaks: {len(listofidxsofresult_lin)}'); plt.legend(loc='upper left'); plt.grid(True); plt.xlim(0, 10000); plt.ylim(-60, 10); plt.show()

                #     nn=188
                #     fig,ax=plt.subplots(num="pdb188")
                #     listofidxsofresult_lin, pr=find_peaks(10.0 * np.log10(np.nan_to_num(np.square(np.abs(get_full_DS_spectrum(mylpf(df_allrx.iloc[nn,p], fsr, lpf_fc), fsr)[0])))))
                #     plt.ion(); plt.plot( freqs, 10.0 * np.log10(np.nan_to_num(np.square(np.abs(get_full_DS_spectrum(mylpf(df_allrx.iloc[nn,p], fsr, lpf_fc), fsr)[0])))), 'r-', freqs[listofidxsofresult_lin], 10.0 * np.log10(np.nan_to_num(np.square(np.abs(get_full_DS_spectrum(mylpf(df_allrx.iloc[nn,p], fsr, lpf_fc), fsr)[0]))))[listofidxsofresult_lin], 'bx', label=f'{nn}{df_allrx.columns[p][9:12]}\n#Peaks: {len(listofidxsofresult_lin)}'); plt.legend(loc='upper left'); plt.grid(True); plt.xlim(0, 10000); plt.ylim(-60, 10); plt.show()


                #     ######  only peaks,no signal
                #     plt.ion(); plt.plot( freqs[listofidxsofresult_lin], 10.0 * np.log10(np.nan_to_num(np.square(np.abs(get_full_DS_spectrum(mylpf(df_allrx.iloc[nn,p], fsr, lpf_fc), fsr)[0]))))[listofidxsofresult_lin], 'k>', label=f'{nn}{df_allrx.columns[p][9:12]}\n#Peaks: {len(listofidxsofresult_lin)}'); plt.legend(loc='upper left'); plt.grid(True); plt.show()
                #     ### linear!!
                #     plt.ion(); plt.plot( freqs[listofidxsofresult_lin], np.nan_to_num(np.square(np.abs(get_full_DS_spectrum(mylpf(df_allrx.iloc[nn,p], fsr, lpf_fc), fsr)[0])))[listofidxsofresult_lin], 'k>', label=f'{nn}{df_allrx.columns[p][9:12]}\n#Peaks: {len(listofidxsofresult_lin)}'); plt.legend(loc='upper left'); plt.grid(True); plt.show()
                # """


                if cfo_mthd=="new_lin" :
                    """LINEAR """  
                    # print("p is", p)       

                    fpk_lin_height      = 0.02#0.01 #
                    fpk_lin_dist        = 10
                    # fpk_lin_promienence = 0.007
                    # fpk_lin_neighb_thrs = 0.0002

                    listofidxsofresult_lin, _  = find_peaks(result_fft_lin, height=fpk_lin_height, distance=fpk_lin_dist)
                    # print("len of peaks", len(listofidxsofresult_lin))

                    
                    mean_within_spectrum= np.mean(result_fft_lin[(freqs >= -lpf_fc) & ( freqs <= lpf_fc)])

                    # ###### only signal, no peaks!
                    # fig,ax=plt.subplots(num="mine_linear1"); plt.ion(); plt.plot(freqs, result_fft_lin, 'r-', label=f'{n}{df_allrx.columns[p][9:12]}'); plt.axhline(y = np.mean(result_fft_lin), label='full mean', color = 'm', linestyle = '--'); plt.axhline(y = meanwithin_spectrum, color = 'g', linestyle = '--', label='narw mean' ) ; plt.axhline(y = fpk_lin_height, color = 'b', linestyle = '--', label='fpk'); plt.grid(True); plt.xlim(-lpf_fc, lpf_fc); plt.ylim(0, 0.04);  plt.legend(loc='upper left'); plt.show()

                    ###### only signal with peaks!
                    # if n in [139, 143, 144, 188, 189, 192, 254, 276, 289, 290, 291, 299, 365, 366, 414, 421, 426, 498, 499]:
                    
                    # if p==3:# and n in [3, 75, 86, 433, 445]:
                    #     fig,ax=plt.subplots(num="mine_linear16"); plt.plot(freqs, result_fft_lin, 'r-', label=f'{n}{df_allrx.columns[p][9:12]}'); plt.axhline(y = np.mean(result_fft_lin), label='full mean', color = 'm', linestyle = '--'); plt.axhline(y = mean_within_spectrum, color = 'g', linestyle = '--', label='narw mean' ) ; plt.axhline(y = fpk_lin_height, color = 'b', linestyle = '--', label='fpk pwr thrs'); plt.plot( freqs[listofidxsofresult_lin], np.nan_to_num(np.square(np.abs(get_full_DS_spectrum(mylpf(df_allrx.iloc[n,p], fsr, lpf_fc), fsr)[0])))[listofidxsofresult_lin], 'k>', label=f'{n}{df_allrx.columns[p][9:12]}\n#Peaks: {len(listofidxsofresult_lin)}'); plt.grid(True); plt.ylim(0, 1);  plt.legend(loc='upper left'); plt.xlim(-lpf_fc, lpf_fc);plt.show()
                        # pdb.set_trace()  plt.draw(); plt.pause(1); # 
                    
                    # ##### no signal, only peaks!
                    # fig,ax=plt.subplots(num="mine_linear3"); plt.ion(); plt.plot( freqs[listofidxsofresult_lin], np.nan_to_num(np.square(np.abs(get_full_DS_spectrum(mylpf(df_allrx.iloc[nn,p], fsr, lpf_fc), fsr)[0])))[listofidxsofresult_lin], 'k>', label=f'{nn}{df_allrx.columns[p][9:12]}\n#Peaks: {len(listofidxsofresult_lin)}'); plt.legend(loc='upper left'); plt.grid(True); plt.show()

                    if len(listofidxsofresult_lin) ==0:
                        # print("no peak detected!!!")  
                        continue 

                    

                    idx_ofmaxpsd_but_idx_in_list=np.argmax(result_fft_lin[listofidxsofresult_lin])
                    idx_maxpsd_in_result=listofidxsofresult_lin[idx_ofmaxpsd_but_idx_in_list]

                    val_psd_max                 = result_fft_lin[idx_maxpsd_in_result]
                    val_freq_max                = freqs[idx_maxpsd_in_result]  #### THIS IS THE APPROAX OFFSET!!!
                    # print(n,p, val_freq_max, val_psd_max)


                    # too many peaks! skip msrnt but store idx for later use
                    if len(listofidxsofresult_lin) > fpk_lin_dist and val_psd_max >  20*fpk_lin_height: 
                        # high_SNR_idx_dict[df_allrx.columns[p]].append(n) # continue if using dictionary cause needn for every the p values
                        high_SNR_n_list.append(n) # break if using lists cause only need the n value
                        # fig,ax=plt.subplots(num="mine_linear2"); plt.plot(freqs, result_fft_lin, 'r-', label=f'{n}{df_allrx.columns[p][9:12]}'); plt.axhline(y = np.mean(result_fft_lin), label='full mean', color = 'm', linestyle = '--'); plt.axhline(y = mean_within_spectrum, color = 'g', linestyle = '--', label='narw mean' ) ; plt.axhline(y = fpk_lin_height, color = 'b', linestyle = '--', label='fpk pwr thrs'); plt.plot( freqs[listofidxsofresult_lin], np.nan_to_num(np.square(np.abs(get_full_DS_spectrum(mylpf(df_allrx.iloc[n,p], fsr, lpf_fc), fsr)[0])))[listofidxsofresult_lin], 'k>', label=f'{n}{df_allrx.columns[p][9:12]}\n#Peaks: {len(listofidxsofresult_lin)}'); plt.grid(True); plt.ylim(0, 0.04);  plt.legend(loc='upper left');  plt.show(); # plt.xlim(-lpf_fc, lpf_fc); plt.draw(); plt.pause(1); 
                        # pdb.set_trace()
                        break


                    # lowerlimitspectrum = lowerlimit_purple_5perfect_3bads_dict[fn][2] if  fn in lowerlimit_purple_5perfect_3bads_dict else 0
                    # upperlimitspectrum = lowerlimit_purple_5perfect_3bads_dict[fn][1] if  fn in lowerlimit_purple_5perfect_3bads_dict  else lpf_fc
                    
                    # to ensure signal was indeed "seen"
                    if val_psd_max > fpk_lin_height and val_psd_max< 20*fpk_lin_height and val_freq_max < lpf_fc and val_freq_max > 0: 
                        # print(n,p, val_freq_max, val_psd_max)
                        freqoff_dict[df_allrx.columns[p]].append(val_freq_max) # print(n, p, "val_psd_max" , val_psd_max)  if p<2 else ''
                        freqoff_time_dict[df_allrx.columns[p]].append([val_freq_max, this_measr_timeuptoseconds])# freqoff_dist_dict[df_allrx.columns[p]].append([val_freq_max, calcDistLatLong(  all_BS_coords[columns_names_array[p].split('-')[1]] ,  matched_row_ingt.iloc[0][3:5]  )])
                    how_many_zero_vel = how_many_zero_vel+1


                else:
                    # cfo_mthd = "db"
                    """ DB """

                    '''
                    with this db option of this method, though we end up with lesser or but same number of CFOs than the other method get_cfo() 's only method where it is also but always db,
                    but here, we dont have to use the 'lowerlimit_purple_5perfect_3bads_dict' where we had manually set the ranges to drop the outlier CFOs....


                    Besides, even missing only a few CFOs here and there doesnt matter, cause they come in groups and even if just one stays in a group, fitting will still yield the same results!
                    

                    When compared either of the db results with the linear results, the linear results are much much worse, maybe cause threholds are too strict, so very low numberof cfos are collected, so the fit is bad.
                    but the ones that are collected are a subset/are from the db values itself, but they are just low in number......that is lin method isnt catching wrong/new cfos...(except in  d21, 15, d14, d13: note that these had to be manually fixed, so linear method is not even able to stay asway from outliers, that is linear method still keeps the outliers...)


                    SO, if in both the methods, if the db option is btter, then keeping the modified method cause I can get the SNR indexes as high_SNR_n_list 
                    
                    if you wanna work/keep on linear, since the number of CFOS are drastically low, you can do different degree fits 
                    and
                    but must get do new pickles with skipped SNR indexes!
                    
                    '''


                    fpk_lin_dist = 6

                    listofidxsofresult_db, pr  = find_peaks(result_fft_db, height = pwr_threshold, distance=fpk_lin_dist) # prominence =44)#, threshold=5)#, prominence=30)   #Returns:Indices of peaks in x that satisfy all given conditions. # (Pdb) idxs[[0,-1]]# array([     2, 131070])
                    
                    if len(listofidxsofresult_db) ==0: 
                        # print("no peak detected!!!")
                        continue # that is go on to the next p, dont have to 'break' #
                    
                    # print("len of peaks=", len(listofidxsofresult_db)," at n= ", n, " p= ",p)

                    idx_ofmaxpsd_but_idx_in_list=np.argmax(result_fft_db[listofidxsofresult_db])
                    idx_maxpsd_in_result=listofidxsofresult_db[idx_ofmaxpsd_but_idx_in_list]

                    val_psd_max                 = result_fft_db[idx_maxpsd_in_result]
                    val_freq_max                = freqs[idx_maxpsd_in_result]  #### THIS IS THE APPROAX OFFSET!!!
                    # print(n,p, val_freq_max, val_psd_max)

                    # fig,ax=plt.subplots(num="db2"); plt.plot( freqs, 10.0 * np.log10(np.nan_to_num(np.square(np.abs(get_full_DS_spectrum(mylpf(df_allrx.iloc[n,p], fsr, lpf_fc), fsr)[0])))), 'r-', freqs[listofidxsofresult_db], 10.0 * np.log10(np.nan_to_num(np.square(np.abs(get_full_DS_spectrum(mylpf(df_allrx.iloc[n,p], fsr, lpf_fc), fsr)[0]))))[listofidxsofresult_db], 'go', label=f'{n}{df_allrx.columns[p][9:12]}\n#Peaks: {len(listofidxsofresult_db)}'); plt.axhline(y = np.mean(result_fft_db[(freqs >= -lpf_fc) & ( freqs <= lpf_fc)]), color = 'g', linestyle = '--', label='narw mean' ); plt.axhline(y = pwr_threshold, color = 'b', linestyle = '--', label='fpk pwr thrs');  plt.legend(loc='upper left'); plt.grid(True); plt.xlim(0, 10000); plt.ylim(-60, 10); plt.show();
                        
                    ####when too many peaks! skip msrnt but store idx for later use
                    if len(listofidxsofresult_db) > fpk_lin_dist :#and val_psd_max >  20*fpk_lin_height: 
                        # high_SNR_idx_dict[df_allrx.columns[p]].append(n) # continue if using dictionary cause needn for every the p values
                        high_SNR_n_list.append(n) # break if using lists cause only need the n value
                        

                        # fig,ax=plt.subplots(num="db90"); plt.ion(); plt.plot( freqs, 10.0 * np.log10(np.nan_to_num(np.square(np.abs(get_full_DS_spectrum(mylpf(df_allrx.iloc[n,p], fsr, lpf_fc), fsr)[0])))), 'r-', freqs[listofidxsofresult_db], 10.0 * np.log10(np.nan_to_num(np.square(np.abs(get_full_DS_spectrum(mylpf(df_allrx.iloc[n,p], fsr, lpf_fc), fsr)[0]))))[listofidxsofresult_db], 'go', label=f'{n}{df_allrx.columns[p][9:12]}\n#Peaks: {len(listofidxsofresult_db)}'); plt.legend(loc='upper left'); plt.grid(True); plt.xlim(0, 10000); plt.ylim(-60, 10); plt.show();       
                        # fig,ax=plt.subplots(num="db20"); plt.plot(freqs, result_fft_db, 'r-', label=f'{n}{df_allrx.columns[p][9:12]}'); plt.axhline(y = np.mean(result_fft_db[(freqs >= -lpf_fc) & ( freqs <= lpf_fc)]), color = 'g', linestyle = '--', label='narw mean' ); plt.axhline(y = pwr_threshold, color = 'b', linestyle = '--', label='fpk pwr thrs'); plt.plot( freqs[listofidxsofresult_db], 10*np.log10(np.nan_to_num(np.square(np.abs(get_full_DS_spectrum(mylpf(df_allrx.iloc[n,p], fsr, lpf_fc), fsr)[0]))))[listofidxsofresult_db], 'k>', label=f'{n}{df_allrx.columns[p][9:12]}\n#Peaks: {len(listofidxsofresult_db)}'); plt.grid(True); plt.legend(loc='upper left');  plt.show(); plt.xlim(-lpf_fc, lpf_fc); plt.draw(); plt.pause(1); 
                        # fig,ax=plt.subplots(num="db2"); plt.axhline(y = np.mean(result_fft_db), label='full mean', color = 'm', linestyle = '--');

                        fig,ax=plt.subplots(num="high_SNR"); plt.plot(freqs, 10.0 * np.log10(np.nan_to_num(np.square(np.abs(get_full_DS_spectrum(mylpf(df_allrx.iloc[n,p], fsr, lpf_fc), fsr)[0])))), 'r-', freqs[listofidxsofresult_db], 10.0 * np.log10(np.nan_to_num(np.square(np.abs(get_full_DS_spectrum(mylpf(df_allrx.iloc[n,p], fsr, lpf_fc), fsr)[0]))))[listofidxsofresult_db], 'go', label=f'{n}{df_allrx.columns[p][9:12]}\n#Peaks: {len(listofidxsofresult_db)}'); plt.axhline(y = np.mean(result_fft_db[(freqs >= -lpf_fc) & ( freqs <= lpf_fc)]), color = 'g', linestyle = '--', label='narw mean' ); plt.axhline(y = pwr_threshold, color = 'b', linestyle = '--', label='fpk pwr thrs');  plt.legend(loc='upper left'); plt.grid(True); plt.xlim(-lpf_fc, lpf_fc); plt.ylim(-60, 10);  plt.draw();# pdb.set_trace()
                        plt.figure("high_SNR").savefig(f"{overall_plots_dir}" +"/"+f"{fn}"+f'_{n}_broke_out_for_{p}_{cfo_mthd}'+"_highsnr.pdf",format='pdf')
                        
                        break
                    
                    # print(10.0 * np.log10(np.nan_to_num(np.square(np.abs(get_full_DS_spectrum(mylpf(df_allrx.iloc[n,p], fsr, lpf_fc), fsr)[0]))))[listofidxsofresult_db])
                    # print(10.0 * np.log10(np.nan_to_num(np.square(np.abs(get_full_DS_spectrum(mylpf(df_allrx.iloc[n,p], fsr, lpf_fc), fsr)[0]))))[pr['right_bases']]) #(checked this here!) 
                    # print(10.0 * np.log10(np.nan_to_num(np.square(np.abs(get_full_DS_spectrum(mylpf(df_allrx.iloc[n,p], fsr, lpf_fc), fsr)[0]))))[pr['right_bases']] + pr['prominences'])
                    # print(10.0 * np.log10(np.nan_to_num(np.square(np.abs(get_full_DS_spectrum(mylpf(df_allrx.iloc[n,p], fsr, lpf_fc), fsr)[0]))))[pr['right_bases']] + pr['prominences'] == 10.0 * np.log10(np.nan_to_num(np.square(np.abs(get_full_DS_spectrum(mylpf(df_allrx.iloc[n,p], fsr, lpf_fc), fsr)[0]))))[listofidxsofresult_db])                    
                    


                    lowerlimitspectrum = lowerlimit_purple_5perfect_3bads_dict[fn][2] if  fn in lowerlimit_purple_5perfect_3bads_dict else 0
                    upperlimitspectrum = lowerlimit_purple_5perfect_3bads_dict[fn][1] if  fn in lowerlimit_purple_5perfect_3bads_dict  else lpf_fc
                    
                    # if p==0:# and n in [3, 75, 86, 433, 445]:
                    #     fig,ax=plt.subplots(num="db1"); plt.ion(); plt.plot( freqs, 10.0 * np.log10(np.nan_to_num(np.square(np.abs(get_full_DS_spectrum(mylpf(df_allrx.iloc[n,p], fsr, lpf_fc), fsr)[0])))), 'r-', freqs[listofidxsofresult_db], 10.0 * np.log10(np.nan_to_num(np.square(np.abs(get_full_DS_spectrum(mylpf(df_allrx.iloc[n,p], fsr, lpf_fc), fsr)[0]))))[listofidxsofresult_db], 'go', label=f'{n}{df_allrx.columns[p][9:12]}\n#Peaks: {len(listofidxsofresult_db)}'); plt.legend(loc='upper left'); plt.grid(True); plt.xlim(0, 10000); plt.ylim(-60, 10); plt.show();# plt.pause(0.1)
                    #     pdb.set_trace()

                   # to ensure signal was indeed "seen"
                    if val_psd_max > pwr_threshold and val_freq_max < upperlimitspectrum and val_freq_max > lowerlimitspectrum: 
                        freqoff_dict[df_allrx.columns[p]].append(val_freq_max) # print(n, p, "val_psd_max" , val_psd_max)  if p<2 else ''
                        freqoff_time_dict[df_allrx.columns[p]].append([val_freq_max, this_measr_timeuptoseconds])# freqoff_dist_dict[df_allrx.columns[p]].append([val_freq_max, calcDistLatLong(  all_BS_coords[columns_names_array[p].split('-')[1]] ,  matched_row_ingt.iloc[0][3:5]  )])
                    how_many_zero_vel = how_many_zero_vel+1        
            

            else:
                how_many_nonzero_vel = how_many_nonzero_vel+1

    
    plt.ioff()
    plt.close()
                
                # manually fixing for 5 fixables
                # if val_psd_max > pwr_threshold and val_freq_max < 8000 and val_freq_max > 5600:  # D7:  02-03-2023_12-55-47
                # if val_psd_max > pwr_threshold and val_freq_max < 10000 and val_freq_max > 6200: # D13: 02-14-2023_10-45-17
                # if val_psd_max > pwr_threshold and val_freq_max < 10000 and val_freq_max > 5600:  # D14: 02-14-2023_12-48-02
                # if val_psd_max > pwr_threshold and val_freq_max < 10000 and val_freq_max > 8300:  # D15: 02-14-2023_14-49-21
                # if val_psd_max > pwr_threshold and val_freq_max < 10000 and val_freq_max > 5000:  # D21: 02-16-2023_16-59-03
                

    n_stationary_msrmnts = how_many_zero_vel/n_endpoints_is_subplots
    n_moving_msrmnts     = how_many_nonzero_vel/n_endpoints_is_subplots

    print("\nNumber of stationary measurements     ==", n_stationary_msrmnts)
    print("Number of moving measurements         ==", n_moving_msrmnts)

    print("\n\nNumber of >Unique< indexes to be missed are:", len(np.unique(no_measr_time_idx_n + no_gps_mesrnt_idx_n)), "\n\n")
    print("\n\n>new Unique< indexes to be missed are:", np.unique(no_measr_time_idx_n + no_gps_mesrnt_idx_n), "\n\n")

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
            
            '''
            len(to_fit)  == [4, 3, 2]
            deg_of_polyn == [3, 2, 1]
            len(ceoff)   == [4, 3, 2]
            '''

            if len(value_all_tuples) !=0: 
                tts=[]
                ffs=[]
                for each_tuple in value_all_tuples:
                    delta_t_since_start = convert_strptime_to_currUTCtsdelta(each_tuple[1], exp_start_timestampUTC)
                    tts.append(   delta_t_since_start  )
                    ffs.append(each_tuple[0])

                #NOT THIS ANYMORE! fitd_frqoff_perrx_dict.update({one_key: fit_freq_on_time(tts,ffs, degreeforfitting)})

                if len(value_all_tuples)>degreeforfitting: #4>3 
                    degreeforfitting_n = degreeforfitting  # a cubic line for us  
                elif len(value_all_tuples) == degreeforfitting: #3=3  
                    degreeforfitting_n = len(value_all_tuples)-1 # 2 # a quad line for us
                elif len(value_all_tuples) == degreeforfitting-1: #2==3-1
                    degreeforfitting_n = len(value_all_tuples)-1 #1 !!a stright line for us
                # no case for the solo freq value!??
                else: #len(value_all_tuples) ==1
                    degreeforfitting_n = len(value_all_tuples)-1


                fitd_frqoff_perrx_dict.update({one_key: fit_freq_on_time(tts,ffs, degreeforfitting_n)})

            else:
                fitd_frqoff_perrx_dict.update({one_key: None}) 
                print("POLYFIT METHOD: Stationary signal wasnt seen for the RX", one_key, "so, no fitted freq offset obtained! How will you extrapolate foffset then?\
                 \nEither change the powerthreshold or take the avereage method's solutionm(doing latter one for now)")
                # exit()
    
    else:
        print("stationary signal never seen at any BS, check your threshold or bus_frequency_offset_ranges !")
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


    # print("\n\nMethod0: Median freq offset  ==", medn_frqoff_perrx_dict, "\n")
    # print("Method1: Avg freq offset     ==", mean_frqoff_perrx_dict, "\n")
    # print(f"Method2: Polyfit parameters fr degree = {degreeforfitting} are ==\n", fitd_frqoff_perrx_dict, "\n")
    


    summary_cfo_dict = {
    'n_moving_msrmnts' : n_moving_msrmnts,
    'pwr_threshold': pwr_threshold ,
    'degreeforfitting': degreeforfitting,
    'exp_start_timestampUTC': exp_start_timestampUTC,
    'meanmethod': mean_frqoff_perrx_dict, 
    'fitdmethod': fitd_frqoff_perrx_dict, 
    'allcfotime': freqoff_time_dict
    }
 
    print("......done getting the CFO\n\n")
    

    print("high_SNR_n_list", fn, high_SNR_n_list)
    # plot_all_off_dictionaries()
    return summary_cfo_dict, no_measr_time_idx_n, no_gps_mesrnt_idx_n, high_SNR_n_list, n_moving_msrmnts




##########################

def get_cfo(fn, df_allrx, df_allti, gt_loc_df, fsr, lpf_fc, exp_start_timestampUTC, pwr_threshold, degreeforfitting):
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
                # print(f'AGAIN: somehow bus didnt record {n}th GPS measurment! for this row', n)
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
                upperlimitspectrum = lowerlimit_purple_5perfect_3bads_dict[fn][1] if  fn in lowerlimit_purple_5perfect_3bads_dict  else lpf_fc

                # to ensure signal was indeed "seen"
                if val_psd_max > pwr_threshold and val_freq_max < upperlimitspectrum and val_freq_max > lowerlimitspectrum: 
                
                ###### manually fixing for 5 fixables
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
    
    print("\n\nNumber of >Unique< indexes to be missed are:", len(np.unique(no_measr_time_idx_n + no_gps_mesrnt_idx_n)), "\n\n")
    print("\n\n>old Unique< indexes to be missed are:", np.unique(no_measr_time_idx_n + no_gps_mesrnt_idx_n), "\n\n")

    # print("\nMISSED measurements!\n","1. due to missed TIME or waitres", no_measr_time_idx_n, "\n2. due to missed GPS", no_gps_mesrnt_idx_n)    
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
            if len(value_all_tuples) > degreeforfitting: # !=0: 
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
        print("stationary signal never seen at any BS, check your threshold or bus_frequency_offset_ranges !")
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


    # print("\n\nMethod0: Median freq offset  ==", medn_frqoff_perrx_dict, "\n")
    # print("Method1: Avg freq offset     ==", mean_frqoff_perrx_dict, "\n")
    # print(f"Method2: Polyfit parameters fr degree = {degreeforfitting} are ==\n", fitd_frqoff_perrx_dict, "\n")
    

 
    summary_cfo_dict = {
    'pwr_threshold': pwr_threshold ,
    'degreeforfitting': degreeforfitting,
    'exp_start_timestampUTC': exp_start_timestampUTC,
    'meanmethod': mean_frqoff_perrx_dict, 
    'fitdmethod': fitd_frqoff_perrx_dict, 
    'allcfotime': freqoff_time_dict
    }
    print("......done getting the CFO\n\n")
    
    return summary_cfo_dict, no_measr_time_idx_n, no_gps_mesrnt_idx_n 






##########################











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
    polynomia = np.poly1d(coef)
    interpolated_freq = polynomia(np.array([at_this_time]))
    return interpolated_freq

def do_cfo_removal(summary_cfo_dict, degreeforfitting, this_measr_timeuptoseconds, this_bs, this_measurement, rate):
    '''
    '''

    fitd_feqoff_perrx_dict   = summary_cfo_dict['fitdmethod']
    
    mean_frqoff_perrx_dict   = summary_cfo_dict['meanmethod']
    
    frqoff_time_perrx_dict   = summary_cfo_dict['allcfotime']

    exp_start_timestampUTC   = summary_cfo_dict['exp_start_timestampUTC']

    delta_t_since_start      = convert_strptime_to_currUTCtsdelta(this_measr_timeuptoseconds, exp_start_timestampUTC)
    
    if (np.all(fitd_feqoff_perrx_dict[this_bs]!=None)) and (len(frqoff_time_perrx_dict[this_bs]) >= degreeforfitting-1): #This means at least ==2, the stright line fit !   not == degreeforfitting+1 anymore): # If any any 'deg=n' fit happened and was proper, then len(coeff) = deg+1   # print( 'Polyfit', len(frqoff_time_perrx_dict[this_bs]))
        method               = 'Polyfit'   ## Frequency Correction METHOD 1: polyfit!
        
        ### polyfit was done for all thease 3 conditons but with a different degree so sizes of the rerutnred coeff vector will be different, with least in size ==degreeforfitting-1. 
        # if   len(frqoff_time_perrx_dict[this_bs])  > degreeforfitting: # that is, 4 , I.E. if fit hsppened and was proper, that is len(coeff) = deg+1        
        #     foffset_approximated = get_interpolated_foff(fitd_feqoff_perrx_dict[this_bs], delta_t_since_start)[0]
        # elif len(frqoff_time_perrx_dict[this_bs])  == degreeforfitting:  #3=3 # !=0: 
        #     foffset_approximated = get_interpolated_foff(fitd_feqoff_perrx_dict[this_bs], delta_t_since_start)[0]
        # elif len(frqoff_time_perrx_dict[this_bs])  == degreeforfitting-1: #2=3-1 # !=0: 
        
        # but since the fit is already done, the f_off doesnt care about the size of the coeff vector, and calling the get_interpolated_foff can still give you the offset as it was before the 3 conditions!
        foffset_approximated = get_interpolated_foff(fitd_feqoff_perrx_dict[this_bs], delta_t_since_start)[0]  

    else: # for single CFO that is when NONE was stored 
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


