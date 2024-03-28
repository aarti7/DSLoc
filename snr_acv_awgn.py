
def get_avg_power(samps):

    meanofthesignal = np.sum(samps) / len(samps)

    varofthesignal = np.sqrt( np.sum(np.square(samps - meanofthesignal)**2 ) / len(samps) ) 

    allpower = np.sum(np.square(np.abs(samps)))
    avgpwrlinear = allpower /len(samps) # expectecd value of S^2
    avgpwrindb = 10.0 * np.log10(avgpwrlinear)
    return avgpwrindb 


def get_acv1(corrected_signal_withfdANDfresidue, rate, nsamps, ns_freq_res):
    pdb.set_trace()

    meanofthesignal = np.sum(corrected_signal_withfdANDfresidue) / len(corrected_signal_withfdANDfresidue)

    varofthesignal = np.sqrt( np.sum(np.square(corrected_signal_withfdANDfresidue - meanofthesignal)**2 ) / len(corrected_signal_withfdANDfresidue) ) 

    sigma_squared_X = np.var(corrected_signal_withfdANDfresidue) # will be used to rep noise!!

    # Calculate the auto-correlation or auto-covariance of the signal+noise, and the contribution of the white noise is a scaled impulse,  𝜎2𝑋𝛿(𝑡) at the origin. 
    acv_frqcrctdlpfd = np.correlate(corrected_signal_withfdANDfresidue, corrected_signal_withfdANDfresidue, mode='full') / nsamps
    impulse = sigma_squared_X * np.array([1 if i == len(acv_frqcrctdlpfd) // 2 else 0 for i in range(len(acv_frqcrctdlpfd))])  # rep of  noise!!
    
    #The remaining auto-covariance is due to the signal. By removing the impulse and Fourier transforming the auto-covariance, you recover the spectrum of the “cleaned” signal.
    acv_awgn_rmvd_frqcrctdlpfd = acv_frqcrctdlpfd - impulse
    
    freqacv = np.fft.fftshift(np.fft.fftfreq(len(awgn_rmvd_frqcrctdlpfd), 1/rate))
    fdidxacv  = (freqacv >= -ns_freq_res) & ( freqacv <= ns_freq_res)


    return acv_frqcrctdlpfd, acv_awgn_rmvd_frqcrctdlpfd, fdidxacv

def get_acv2(corrected_signal_withfdANDfresidue, rate, nsamps, ns_freq_res):
    pdb.set_trace()

    meanofthesignal = np.sum(corrected_signal_withfdANDfresidue) / len(corrected_signal_withfdANDfresidue)

    varofthesignal = np.sqrt( np.sum(np.square(corrected_signal_withfdANDfresidue - meanofthesignal)**2 ) / len(corrected_signal_withfdANDfresidue) ) 

    sigma_squared_X = np.var(corrected_signal_withfdANDfresidue) # will be used to rep noise!!

    # Calculate the auto-correlation or auto-covariance of the signal+noise, and the contribution of the white noise is a scaled impulse,  𝜎2𝑋𝛿(𝑡) at the origin. 
    acv_frqcrctdlpfd = np.correlate(corrected_signal_withfdANDfresidue, corrected_signal_withfdANDfresidue, mode='full') / nsamps
    impulse = sigma_squared_X * np.array([1 if i == len(acv_frqcrctdlpfd) // 2 else 0 for i in range(len(acv_frqcrctdlpfd))])  # rep of  noise!!
    
    #The remaining auto-covariance is due to the signal. By removing the impulse and Fourier transforming the auto-covariance, you recover the spectrum of the “cleaned” signal.
    acv_awgn_rmvd_frqcrctdlpfd = acv_frqcrctdlpfd - impulse
    
    freqacv = np.fft.fftshift(np.fft.fftfreq(len(awgn_rmvd_frqcrctdlpfd), 1/rate))
    fdidxacv  = (freqacv >= -ns_freq_res) & ( freqacv <= ns_freq_res)


    return acv_frqcrctdlpfd, acv_awgn_rmvd_frqcrctdlpfd, fdidxacv

def get_acv3(sig, fs):
    pdb.set_trace()
   
    """    
    # mean
    meanofthesignal = np.sum(sig) / len(sig)
    
    #var
    v0 = np.sum(np.square(sig - meanofthesignal) )
    v1 = v0 / len(sig)
    varofthesignal  = np.linalg.norm( v1 )

    # var auto
    sigma_squared_X = np.var(sig) # will be used to rep noise!!

    # check if same
    sigma_squared_X == varofthesignal

    allpower = np.sum(np.square(np.abs(samps))) #  were samps are already mean rmeoved?N0 i think cause mean was calculated above and it was nonzero. abs(x_i) = [sqrt(a^2 + b^2)_i]
    avgpwrlinear = allpower /len(samps) # expectecd value of S^2
    # Ans: so for power and var to be same, remove mean from sia before taking the power!
    """
    avg = np.mean(sig)
    var = np.var(sig) # doesn't change when mean is removed from sig

    sig = sig - avg
    allpower = np.sum(np.square(np.abs(sig))) 
    avgpwrlinear = allpower /len(sig) # must be equal to the var
    avgpwrindb = 10.0 * np.log10(avgpwrlinear)

    """
    1. fs samples take   --> 1 sec 
       one sample takes  --> 1/fs seconds --> ts seconds
       so:     t ->>> np.arange(0, 124*ts, ts) ---- > will give a t of lengthof_t=124== durationofsig, that is, 120 smaples with space of ts between them.

       you know fs == 50 samples per second here and you know 124 as such before hand. 

    2.  how to get this same t vector by linspace?

        t --->  np.linspace(0, 1sec, fs) -- > makes sense by defination of frequency being numnr of samples in one second 
        you can think of endtimeofsig = 1second here

        more imrpotantly, this will create a t vecot of length=1sec*fs
        t ---> np.linspace(0, endtime=1sec, 50,endpoint=False) same length of t==1second*50==== >50

        if endtimeofsig = 2.48seconds
        Numberfsamplesbythisendtime = fs * endtimeofsig == length of t vector ==> thus, thrid place shuld be equal to 2.48*fs= 124
        
        t --> np.linspace(0,  endtime= 2.48inseconds, 2.48*fs= 124 samplesbytheendtime) =len(t)is 124

        so more genreally, t = np.linspace(0,  E*1, E*fs) such that lengthof_t ==124

        summary:
        np.arange(0, L*ts, ts) ==t== np.linspace(0,  1*E, , int(E*fs), endpoint=False)
        len(t) = L = 124
        so: E = L*ts = 2.48

        # Calculate the auto-correlation or auto-covariance of the signal+noise, and the contribution of the white noise is a scaled impulse,  𝜎2𝑋𝛿(𝑡) at the origin. 
        sigma_squared_X = np.var(sig)

        acv_frqcrctdlpfd = np.correlate(sig, sig, mode='full') / len(sig)
        impulse = sigma_squared_X * np.array([1 if i == len(acv_frqcrctdlpfd) // 2 else 0 for i in range(len(acv_frqcrctdlpfd))])  # rep of  noise!!
    """




    
    #The remaining auto-covariance is due to the signal. By removing the impulse and Fourier transforming the auto-covariance, you recover the spectrum of the “cleaned” signal.
    clean_rxx = acv_frqcrctdlpfd - impulse
    clean_time = np.fft.ifft(np.fft.fftshift(clean_rxx)) 
    freqacv = np.fft.fftshift(np.fft.fftfreq(len(acv_awgn_rmvd_frqcrctdlpfd), 1/fs))
    fdidxacv  = (freqacv >= -ns_freq_res) & ( freqacv <= ns_freq_res)

    return acv_frqcrctdlpfd, acv_awgn_rmvd_frqcrctdlpfd, fdidxacv

 
def get_SNR():
    plot the SNR of each observation over the 2d xy for each basestation in the 3d plot.
    I think the sNR should show a pattern according to the distance of the observation from the BS


def plot_sig(this_measurement, this_measurement_after_LPF, corrected_signal_withfdANDfresidue, fdidx, acv_frqcrctdlpfd, awgn_rmvd_frqcrctdlpfd, fdidxacv, rate, nsamps):

    # np.random.seed(42)
    # t = np.linspace(0, 1, len(this_measurement), endpoint=False)
    # clean_signal = np.max(this_measurement.real)*np.sin(2 * np.pi * 50 * t)+np.max(this_measurement.imag)* 1j*np.sin(2 * np.pi * 50 * t)

    # # Adaptive Filtering (LMS algorithm)
    # mu = 0.01  # Adaptation step size
    # order = 32  # Filter order
    # # Initialize filter weights
    # W = np.zeros(order)
    # pdb.set_trace()

    # # LMS algorithm
    # for i in range(order, len(this_measurement)):
    #     x = this_measurement[i-order:i]
    #     y_hat = np.dot(W, x)
    #     e = clean_signal[i] - y_hat
    #     W = W + mu * e * x

    # clean_estimate = np.convolve(this_measurement, W, mode='same')

    # plt.figure(figsize=(10, 6))
    # plt.plot(t, clean_signal, label='Clean Signal', alpha=0.7)
    # plt.plot(t, this_measurement, label='Noisy Signal', alpha=0.7)
    # plt.plot(t, clean_estimate, label='Clean Estimate', linestyle='--', linewidth=2)
    # plt.legend()
    # plt.title('Adaptive Filtering for AWGN Denoising (LMS)')
    # plt.show()

    fig, ax = plt.subplots(3,4)
    ax[0][0].plot(this_measurement, label = "raw")
    ax[0][1].plot(this_measurement_after_LPF, label = "lpfd")
    ax[0][2].plot(corrected_signal_withfdANDfresidue, label="frqcrctd+lpfd")
    ax[0][3].plot(np.arange(-len(acv_frqcrctdlpfd)//2, len(acv_frqcrctdlpfd)//2), acv_frqcrctdlpfd, label = "frqcrctd+lpfd acv")

    ax[1][0].plot(np.fft.fftshift(np.fft.fftfreq(nsamps, 1/rate)), np.nan_to_num(10.0*np.log10(np.square(np.abs(np.fft.fftshift(np.fft.fft(this_measurement)))))), label = "raw fft")
    ax[1][1].plot(np.fft.fftshift(np.fft.fftfreq(nsamps, 1/rate)), np.nan_to_num(10.0*np.log10(np.square(np.abs(np.fft.fftshift(np.fft.fft(this_measurement_after_LPF)))))), label = "lpfd fft")
    ax[1][2].plot(np.fft.fftshift(np.fft.fftfreq(nsamps, 1/rate)), np.nan_to_num(10.0*np.log10(np.square(np.abs(np.fft.fftshift(np.fft.fft(corrected_signal_withfdANDfresidue)))))), label="frqcrctd+lpfd fft")
    ax[1][3].plot(np.fft.fftshift(np.fft.fftfreq(len(awgn_rmvd_frqcrctdlpfd), 1/rate)), np.nan_to_num(10.0*np.log10(np.square(np.abs(np.fft.fftshift(np.fft.fft(awgn_rmvd_frqcrctdlpfd)))))), label = "frqcrctd+lpfd awgnremoved fft full")
    
    ax[2][0].plot(np.fft.fftshift(np.fft.fftfreq(nsamps, 1/rate))[fdidx], np.nan_to_num(10.0*np.log10(np.square(np.abs(np.fft.fftshift(np.fft.fft(this_measurement))))))[fdidx], label = "raw fft ns")
    ax[2][1].plot(np.fft.fftshift(np.fft.fftfreq(nsamps, 1/rate))[fdidx], np.nan_to_num(10.0*np.log10(np.square(np.abs(np.fft.fftshift(np.fft.fft(this_measurement_after_LPF))))))[fdidx], label = "lpfd fft ns")
    ax[2][2].plot(np.fft.fftshift(np.fft.fftfreq(nsamps, 1/rate))[fdidx], np.nan_to_num(10.0*np.log10(np.square(np.abs(np.fft.fftshift(np.fft.fft(corrected_signal_withfdANDfresidue))))))[fdidx], label="frqcrctd+lpfd fft ns")
    ax[2][3].plot(np.fft.fftshift(np.fft.fftfreq(len(awgn_rmvd_frqcrctdlpfd), 1/rate))[fdidxacv], np.nan_to_num(10.0*np.log10(np.square(np.abs(np.fft.fftshift(np.fft.fft(awgn_rmvd_frqcrctdlpfd))))))[fdidxacv], label="frqcrctd+lpfd  awgn removed fft ns")


    ax[0][0].legend(loc="lower left"), ax[0][1].legend(loc="lower left"), ax[0][2].legend(loc="lower left"), ax[1][0].legend(loc="lower left")
    ax[1][1].legend(loc="lower left"), ax[1][2].legend(loc="lower left"), ax[2][0].legend(loc="lower left"), ax[2][1].legend(loc="lower left"),
    ax[2][2].legend(loc="lower left"), ax[0][3].legend(loc="lower left"), ax[1][3].legend(loc="lower left"), ax[2][3].legend(loc="lower left"), 
    ax[1][0].set_ylim(-180,20),ax[1][1].set_ylim(-180,20),ax[1][2].set_ylim(-180,20),ax[2][0].set_ylim(-80,20),ax[2][1].set_ylim(-80,20),ax[2][2].set_ylim(-80,20) # ax[3][1].set_ylim(-80,20), ax[3][2].set_ylim(-80,20)
    plt.legend()
    plt.show()



def get_cfo(df_allrx, df_allti, gt_loc_df, fsr, lpf_fc, exp_start_timestampUTC, degreeforfitting, pwr_threshold):
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
    # pdb.set_trace()

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


            fpk_thr  = 0.0007 #0.001 ### the vertical distance to its neighboring samples
            fpk_dst  = 10    #300 # fpk_height = 0.002
            fpk_prom = .001 #3 #30 # the vertical distance between the peak and its lowest contour line


            """
            by limiting the allowed prominence to this values!!!"""#<<<<<<<<<<<<<<#THIS IS WRONG WORDING/Example on https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html!
            # when I set prominence=0.002 result was in following pdb output!
           
            """ (Pdb) pr["prominences"].max()
                0.00428507406004766
                (Pdb) pr["prominences"]
                array([0.00242277, 0.00249285, 0.00223176, 0.00202951, 0.00216404,
                       0.00200453, 0.00428507, 0.00202779])
            
            See here that max is 0.004!!!

            !!!So,

            >>>>fpk_prom is: all peaks that are at least fpk_prom and above
            xxxx    isNOT: at max fpk_prom and below!
            
            pwr_threshold_lin = 0.001 # so pwr threhold and promience value are kinda like the same thing!!!       

            # prominence is like how many samples are maximumally this much away from baseline can one go....kinda like elastic band away from baseline! kinda like sweeping the floor .. not really getting the tops!

            """


            
            ##########################
            if this_speed.values[0] == 0: # to ensure only for cases  when there was no motion. 
                
                this_measurement  = df_allrx.iloc[n,p]
                this_measurement  = mylpf(this_measurement, fsr, lpf_fc)

                result_fft,      freqs     = get_full_DS_spectrum(this_measurement, fsr)
                result_fft_lin             = np.nan_to_num(np.square(np.abs(result_fft))) 

                # both/all condition have to be true at the same time!
                pwr_threshold_is_lin_promienence = 0.004
                listofidxsofresult_lin, _  = find_peaks(result_fft_lin, threshold=fpk_thr, distance=fpk_dst)       # prominence=fpk_prom, distance=300)# threshold=0.001)# prominence=3)# height = 0.002)# threshold=0.001)#, prominence=30)   #Returns:Indices of peaks in x that satisfy all given conditions. # (Pdb) idxs[[0,-1]]# array([     2, 131070])
                
                if len(listofidxsofresult_lin) ==0: # if len(listofidxsofresult_lin) >=200:
                    pdb.set_trace()
                    break


                idx_ofmaxpsd_but_idx_in_list= np.argmax(result_fft_lin[listofidxsofresult_lin])
                idx_maxpsd_in_result=listofidxsofresult_lin[idx_ofmaxpsd_but_idx_in_list]
                
                val_psd_max                 = result_fft_lin[idx_maxpsd_in_result]
                val_freq_max                = freqs[idx_maxpsd_in_result]  #### THIS IS THE APPROAX OFFSET!!!
                
                # result_fft_db               = np.nan_to_num(10.0 * np.log10(result_fft_lin))             
                # sorted_arr                  = np.sort(result_fft_db)[::-1]
                # top_3_values                = sorted_arr[:3]
                # indexes = [i for i in range(len(result_fft_db) - len(top_3_values) + 1) if result_fft_db[i:i+len(top_3_values)] == top_3_values]


                # listofidxsofresult, _      = find_peaks(result_fft_db)                
                # idx_ofmaxpsd_but_idx_in_list=np.argmax(result_fft_db[listofidxsofresult])
                # idx_maxpsd_in_result=listofidxsofresult[idx_ofmaxpsd_but_idx_in_list]
                # val_psd_max                 = result_fft_db[idx_maxpsd_in_result]
                # val_freq_max                = freqs[idx_maxpsd_in_result]  #### THIS IS THE APPROAX OFFSET!!!



                # # ##THIS WAS INCORRECT => 
                # # idx_psd_max               = np.argmax(result_fft_temp_db[idxs])
                # # val_maxpsd_in_result=     np.max(result_fft_db[listofidxsofresult])
                # # mxpsd = np.max(result_fft_db)
                # # frq_mxpsd = freqs[np.argmax(result_fft_db) ]
                
                # # CORRECT!
                # idx_psd_max_in_all_peaks_idxs_not_in_results = idxs[np.argmax(result_fft_temp_db[idxs])] 
                # idx_psd_max = idx_psd_max_in_all_peaks_idxs_not_in_results
                 


                to_test = [139, 143, 144, 188, 189, 192, 254, 276, 289, 290, 291, 299, 365, 366, 414, 421, 426, 498, 499]
                


                # if p==0:# and len(listofidxsofresult_lin) <=50:# and n >=133:# in [12, 28, 64, 262, 263, 270, 273, 356]:
                if p==1:# and n in to_test:
                    # plt.clf()
                    # print(mxpsd, "at", frq_mxpsd) 

                    print(n, "len=", len(listofidxsofresult_lin) )# ,round(val_freq_max,2), round(val_psd_max,2))
                    print(f"#Peaks: {len(listofidxsofresult_lin)}, freq offset is", round(val_freq_max,3), "max power is", round(val_psd_max,3), "n=",n, "p=",p)

                    plt.plot(freqs, result_fft_lin, label=f"{n}{df_allrx.columns[p][9:12]}. frq({val_freq_max})", color = 'r' if this_speed.values[0] ==0 else 'g') #\n mean{np.mean(result_fft_temp_db)} \n max {val_psd_max}  \n
                    # plt.axhline(y = pwr_threshold, color = 'b', linestyle = '-') 
                    plt.plot(freqs[listofidxsofresult_lin], result_fft_lin[listofidxsofresult_lin], \
                        'o', color='g', label=f'number of Peaks: {len(listofidxsofresult_lin)}')
                    # plt.scatter(val_freq_max, val_psd_max, marker='o', s=100,  color='g', label=f'max at {val_freq_max}')
                    # plt.ylim(-60, 10)
                    plt.xlim(0, 10000)
                    plt.ylim(0, 0.04)
                    plt.legend(loc='upper left')
                    plt.grid(True)
                    plt.title('speed = zero')
                    plt.show()



                    # plt.pause(0.1)
                    # plot(x1, y1, 'g^', x2, y2, 'g-')
                    # !import code; code.interact(local=vars())
                    

                    # pdb.set_trace()
                    # # To regerenrate the plot in pdb
                    # fig,ax=plt.subplots(num="mine2")
                    # plt.ion(); plt.plot(freqs, result_fft_lin, 'r-', freqs[listofidxsofresult_lin], result_fft_lin[listofidxsofresult_lin], 'bx', label=f'{n}{df_allrx.columns[p][9:12]}\n#Peaks:{len(listofidxsofresult_lin)}'); plt.legend(loc='upper left'); plt.grid(True); plt.xlim(0, 10000); plt.ylim(0, 0.04); plt.show()
                    # # color black
                    # plt.ion(); plt.plot(freqs, result_fft_lin, 'r-', freqs[listofidxsofresult_lin], result_fft_lin[listofidxsofresult_lin], 'k^', label=f'{n}{df_allrx.columns[p][9:12]}\n#Peaks:{len(listofidxsofresult_lin)}'); plt.legend(loc='upper left'); plt.grid(True); plt.xlim(0, 10000); plt.ylim(0, 0.04); plt.show()
                    # # color yellow
                    # plt.ion(); plt.plot(freqs, result_fft_lin, 'r-', freqs[listofidxsofresult_lin], result_fft_lin[listofidxsofresult_lin], 'y^', label=f'{n}{df_allrx.columns[p][9:12]}\n#Peaks:{len(listofidxsofresult_lin)}'); plt.legend(loc='upper left'); plt.grid(True); plt.xlim(0, 10000); plt.ylim(0, 0.04); plt.show()





                    # # To plot any random nth smapl!
                    # fig,ax=plt.subplots(num="mine3")
                    # nn=to_test[1]
                    # listofidxsofresult_lin, _= find_peaks(np.nan_to_num(np.square(np.abs(get_full_DS_spectrum(mylpf(df_allrx.iloc[nn,p], fsr, lpf_fc), fsr)[0]))), threshold=fpk_thr, distance=fpk_dst)                       
                    # # color blue
                    # plt.ion(); plt.plot(freqs, np.nan_to_num(np.square(np.abs(get_full_DS_spectrum(mylpf(df_allrx.iloc[nn,p], fsr, lpf_fc), fsr)[0]))), 'r-', freqs[listofidxsofresult_lin], np.nan_to_num(np.square(np.abs(get_full_DS_spectrum(mylpf(df_allrx.iloc[nn,p], fsr, lpf_fc), fsr)[0])))[listofidxsofresult_lin], 'bx', label=f'{nn}{df_allrx.columns[p][9:12]}\n#Peaks: {len(listofidxsofresult_lin)}'); plt.legend(loc='upper left'); plt.grid(True); plt.xlim(0, 10000); plt.ylim(0, 0.04); plt.show()
                    # # color green
                    # plt.ion(); plt.plot(freqs, np.nan_to_num(np.square(np.abs(get_full_DS_spectrum(mylpf(df_allrx.iloc[nn,p], fsr, lpf_fc), fsr)[0]))), 'r-', freqs[listofidxsofresult_lin], np.nan_to_num(np.square(np.abs(get_full_DS_spectrum(mylpf(df_allrx.iloc[nn,p], fsr, lpf_fc), fsr)[0])))[listofidxsofresult_lin], 'go', label=f'{nn}{df_allrx.columns[p][9:12]}\n#Peaks: {len(listofidxsofresult_lin)}'); plt.legend(loc='upper left'); plt.grid(True); plt.xlim(0, 10000); plt.ylim(0, 0.04); plt.show()




                # # print("freq offset is", val_freq_max, "max power is" , val_psd_max, n, p)
                # # print("mean of psd is", np.mean(result_fft_temp_db), "std is", np.std(result_fft_temp_db), "3times std of psd is", 3*np.std(result_fft_temp_db))
  
                # threshold = -21 #-23.5 # np.mean(result_fft_temp) + 3*np.std(result_fft_temp)
                # if val_psd_max > threshold and val_freq_max < bus_frequency_offset_ranges[1] and val_freq_max > bus_frequency_offset_ranges[0]: # to ensure signal was indeed "seen"





                # manually fixing for 5 fixables
                # if val_psd_max > pwr_threshold and val_freq_max < 8000 and val_freq_max > 5600:  # D7:  02-03-2023_12-55-47
                # if val_psd_max > pwr_threshold and val_freq_max < 10000 and val_freq_max > 6200: # D13: 02-14-2023_10-45-17
                # if val_psd_max > pwr_threshold and val_freq_max < 10000 and val_freq_max > 5600:  # D14: 02-14-2023_12-48-02
                # if val_psd_max > pwr_threshold and val_freq_max < 10000 and val_freq_max > 8300:  # D15: 02-14-2023_14-49-21<<
                # if val_psd_max > pwr_threshold and val_freq_max < 10000 and val_freq_max > 5000:  # D21: 02-16-2023_16-59-03

                # to ensure signal was indeed "seen"
                
                if  val_psd_max > pwr_threshold_is_lin_promienence and val_freq_max < lpf_fc and val_freq_max > 0: 
                    # print("meanPSD:", round(np.mean(result_fft_db),3), "stdPSD:", round(np.std(result_fft_db),3), "3timesstdPSD:", round(3*np.std(result_fft_db),3))
                    # print(f"#Peaks: {len(listofidxsofresult_lin)}, freq offset is", round(val_freq_max,3), "max power is", round(val_psd_max,3), "n=",n, "p=",p)

                    # print(n, p, "val_psd_max" , val_psd_max)  if p<2 else ''
                    freqoff_dict[df_allrx.columns[p]].append(val_freq_max)
                    freqoff_time_dict[df_allrx.columns[p]].append([val_freq_max, this_measr_timeuptoseconds])
                    # freqoff_dist_dict[df_allrx.columns[p]].append([val_freq_max, calcDistLatLong(  all_BS_coords[columns_names_array[p].split('-')[1]] ,  matched_row_ingt.iloc[0][3:5]  )])

                how_many_zero_vel = how_many_zero_vel+1
        
            else:
                how_many_nonzero_vel = how_many_nonzero_vel+1

    # plt.ioff()
    # plt.close()
    # pdb.set_trace()
    
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

