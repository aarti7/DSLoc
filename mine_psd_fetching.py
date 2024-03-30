from libs import * 

CFO_REMOVE_FLAG  = 1
PWR_THRESHOLD     = -125 #-19 #in dB
DEGREE_OF_POLYFIT = 3


from functions_cforemoval import *


# plt.rcParams['grid.color'] = (0.5, 0.5, 0.5, 0.1)
# plt.rcParams["figure.figsize"] = (7,7)

# plt.rcParams['lines.linewidth'] = 5
# plt.rcParams['font.size']=20
# plt.rc('xtick', labelsize=20)
# plt.rc('ytick', labelsize=20)

# plt.rcParams['lines.linewidth'] = 2
# plt.rcParams['font.size']=16
# plt.rc('xtick', labelsize=16)
# plt.rc('ytick', labelsize=16)

plt.rcParams['lines.linewidth'] = 2
plt.rcParams['font.size']=8
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)




names_to_ds_dict = {
"01-30-2023_15-40-55":"D1",
"01-30-2023_17-47-39":"D2",
"01-30-2023_19-53-09":"D3",
"01-30-2023_20-33-48":"D4",
"01-30-2023_21-29-33":"D5",
"02-03-2023_10-39-20":"D6",
"02-03-2023_12-55-47":"D7",
"02-06-2023_14-29-39":"D8",
"02-09-2023_13-05-31":"D9",
"02-09-2023_15-08-10":"D10",
"02-09-2023_17-12-28":"D11",
"02-14-2023_08-37-53":"D12",
"02-14-2023_10-45-17":"D13",
"02-14-2023_12-48-02":"D14",
"02-14-2023_14-49-21":"D15",
"02-14-2023_16-57-55":"D16",
"02-14-2023_18-53-20":"D17",
"02-16-2023_10-04-45":"D18",
"02-16-2023_12-23-48":"D19",
"02-16-2023_14-55-52":"D20",
"02-16-2023_16-59-03":"D21",
"02-16-2023_19-01-43":"D22",
"02-16-2023_19-40-05":"D23",
}



######################################### from GPS file ######
###############################################################
###############################################################  ###############################################################
###############################################################  ###############################################################
###############################################################  ###############################################################
###############################################################  ###############################################################
###############################################################  ###############################################################
###############################################################  

all_BS_coords = {
  'hospital': (40.77105, -111.83712),  
  'ustar': (40.76899, -111.84167),
  'bes': (40.76134, -111.84629),
  'honors': (40.76440, -111.83695),
  'smt': (40.76740, -111.83118),}

def latlon_to_utm(lat_lon):
    lat, lon = lat_lon
    utm_coords = utm.from_latlon(lat, lon)
    return utm_coords[1], utm_coords[0]  # Northing, Easting


def quickplot_labels_non_utmed_filtering(i, fn, df, lat_lim, lon_lim, llat, llon): # filenname is passed
   
    labels  = df["speed_postuple"]    
    bsnames = df.columns.values  
    
    fig_ori, ax_ori = plt.subplots(figsize=(6, 6))
    ax_ori.axvline(x = lon_lim, ls=':', c= '#FFB6C1', alpha=0.5) # pink
    ax_ori.axhline(y = lat_lim, ls=':', c= '#FFB6C1', alpha=0.5)
    ax_ori.axvline(x = llon,    ls=':', c= '#FFB6C1', alpha=0.5)
    ax_ori.axhline(y = llat,    ls=':', c= '#FFB6C1', alpha=0.5)
    
    for ii in range(len(labels)):
        lat_y, long_x =  labels.iloc[ii][2][0], labels.iloc[ii][2][1]
        ax_ori.scatter( long_x, lat_y, c='C{}'.format(i), marker='o', s=2)
    ax_ori.scatter( labels.iloc[0][2][1], labels.iloc[0][2][0] , marker='*', s=50, label = 'first loc', c = 'y')

    [ax_ori.scatter(all_BS_coords[bs.split('-')[1]][1], all_BS_coords[bs.split('-')[1]][0], marker='1',
                    s=50, label= bs.split('-')[1] ) for b, bs in enumerate(bsnames) if b<5 ]    
    
    # ax_ori.legend(loc='lower right')
    ax_ori.grid(True)
    plt.title(f"Filtered tracjectory of Exp: {fn} \n Number of filtered mrmnts plotted: {len(labels)}") 
    plt.show()  

    
####################################################


def green_detour(fn, this_exp_df, plotflag):
      
    lat_up =   40.7711
    lon_up = -111.8431   
    c1 = (this_exp_df.iloc[:,-1].apply(lambda x: x[2][0])) <= lat_up   #anything more southwards is to be dropped
    c2 = (this_exp_df.iloc[:,-1].apply(lambda x: x[2][1])) <= lon_up  # anything more westwards is to be dropped    
    detour_df = this_exp_df[ c1 & c2]
    this_exp_df.drop(detour_df.index, inplace=True)     
    
    lat_bottom = 40.7655 # not plotting this line
    c1 = (this_exp_df.iloc[:,-1].apply(lambda x: x[2][0])) <= lat_bottom   #anything more southwards is to be dropped
    detour_df = this_exp_df[ c1]
    this_exp_df.drop(detour_df.index, inplace=True)    
       
    lat_down =  40.7675
    lon_down = -111.8359    
    c1 = (this_exp_df.iloc[:,-1].apply(lambda x: x[2][0])) <= lat_down   #anything more southwards is to be dropped
    c2 = (this_exp_df.iloc[:,-1].apply(lambda x: x[2][1])) >= lon_down  # anything more eastwards is to be dropped       
    detour_df = this_exp_df[ c1 & c2]   
    this_exp_df.drop(detour_df.index, inplace=True)    
    if plotflag:
        quickplot_labels_non_utmed_filtering(random.randint(1,100), fn, this_exp_df, lat_up, lon_up, lat_down, lon_down)
    
    
    this_exp_df.reset_index(inplace=True, drop=True)
    
    return this_exp_df


####################################################

def orange_detour(fn, this_exp_df, plotflag):
      
    lat_limit = 40.770
    lon_limit = -111.8428    
    c1 = (this_exp_df.iloc[:,-1].apply(lambda x: x[2][0])) >= lat_limit   #anything more southwards is to be dropped
    c2 = (this_exp_df.iloc[:,-1].apply(lambda x: x[2][1])) <= lon_limit  # anything more westwards is to be dropped    
    detour_df = this_exp_df[ c1 & c2] 
    this_exp_df.drop(detour_df.index, inplace=True)
    if plotflag:
        quickplot_labels_non_utmed_filtering(random.randint(1,100), fn, this_exp_df, lat_limit, lon_limit, lat_limit, lon_limit)
    
    this_exp_df.reset_index(inplace=True, drop=True)
    return this_exp_df


####################################################

def get_filtered_df_and_plot(name, df, plotflag =False):
    
    condition = (df.iloc[:,-1].apply(lambda x: x[2][0]).max()) >= 40.772   # indexofdetour = this_exp_df.shape[0] if fn != 'meas_02-14-2023_18-53-20' else 50 ## detour temp fix
    
    if condition: 
        print(". Bus route was green.", end="")
        filtered_df = green_detour(name, df, plotflag)  
    else: 
        print(". Bus route was orange.", end="")
        filtered_df = orange_detour(name, df, plotflag)
    
    print(" The filtered df's shape:", filtered_df.shape)
    return filtered_df        




############# 5 declares from mine_plots_rms_cfo_spectrum_loc
###############################################################  ###############################################################
###############################################################  ###############################################################
###############################################################  ###############################################################
###############################################################  ###############################################################
###############################################################  ###############################################################
###############################################################  



def calcDistLatLong(coord1, coord2):
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

    total_power = np.sum(dspd)
    fractional_power = dspd/total_power
    bar_frqs = np.sum(fractional_power*frqs) #== mean of Dshift
    var_frqs = np.sum(fractional_power*(frqs-bar_frqs)**2) 
    rmsFreqval = np.sqrt(var_frqs) #== std=sqrt(variance) of Dshift 
    return round(rmsFreqval, 2)

def rmsfunction(psd_db_ns, psd_linear_ns, freqs_ns, PWR_THRESHOLD):#, FD_MAXPOSSIBLE):
    # print("PWR_THRESHOLD is", PWR_THRESHOLD)
    max_noise_power_db = PWR_THRESHOLD # np.max([np.max(psd_db_ns[noise_window_left]), np.max(psd_db_ns[ noise_window_rght ])])    
    max_noise_power_linear = 10**(max_noise_power_db/10)

    rmsis_entire = getrmsFreq(psd_linear_ns, freqs_ns)
    rmsis_thrsld = getrmsFreq(psd_linear_ns[psd_linear_ns > max_noise_power_linear], freqs_ns[psd_linear_ns > max_noise_power_linear])
    return rmsis_thrsld #, max_noise_power_db

def my_plot_rms_dicts(fnm, rmsdict, overall_plots_dir, runtime):
    print(" you cant comapre the rms for two fifferent bs cause they arent calibrated")
    
    fig_rmsall, axs_rmsall = plt.subplots(figsize=(6,4), num = "rmsall")
    for uu, kvp in enumerate(rmsdict.items()):
        if len(kvp[1]) !=0 and kvp[0].split('-')[1] =='ustar':
            rms_df_d = pd.DataFrame()
            rms_df_d = pd.DataFrame(kvp[1]).apply(lambda x: pd.Series(x) )  
    
            # plt.scatter(rms_df_d.iloc[:,1], rms_df_d.iloc[:,0], label = f'{kvp[0]}'.split('-')[1] )
            
            # zero speed mask!!
            df_onlynonzero = rms_df_d[rms_df_d[0] != 0] 
            # print("before", len(df_onlynonzero))
            maxx = min(df_onlynonzero.iloc[:,0].nlargest(2))
            df_onlynonzero = df_onlynonzero[df_onlynonzero[0] < maxx] # zero speed mask!!
            # print("after",len(df_onlynonzero), f'{kvp[0]}'.split('-')[1])
            axs_rmsall.scatter(df_onlynonzero.iloc[:,1], df_onlynonzero.iloc[:,0], c='C{}'.format(uu), label = f'{kvp[0]}'.split('-')[1] )
    plt.xlabel('Speed(mps)')
    plt.grid(True)
    plt.legend(loc="upper left")
    plt.ylabel('RMS Doppler Spread (Hz)')
    plt.tight_layout()
    print("RMS plots saved!")
    plt.figure("rmsall").savefig(f"{overall_plots_dir}" +"/"+f"{runtime}_{fnm}"+"_rmsall.pdf",format='pdf')
    plt.close("rmsall")    




    """
    RMS shows larger values at zero velocity as they are a function of environment that is Unmonitored/ Unmeasured.

    """

    # fig_rmsper, axs_rmsper = plt.subplots(1, len(rmsdict.keys()),figsize=(12, 3), num = "rmsperbssep")  
    # for uu, kvp in enumerate(rmsdict.items()):
    #     if len(kvp[1]) !=0:
    #         rms_df_d = pd.DataFrame()
    #         rms_df_d = pd.DataFrame(kvp[1]).apply(lambda x: pd.Series(x) )  
    
    #         # axs_rmsper[uu].scatter(rms_df_d.iloc[:,1], rms_df_d.iloc[:,0], label = f'{kvp[0]}'.split('-')[1] )
            
    #         # zero speed mask!!
    #         df_onlynonzero = rms_df_d[rms_df_d[0] != 0] 
    #         # print("before", len(df_onlynonzero))
    #         maxx = min(df_onlynonzero.iloc[:,0].nlargest(2))
    #         df_onlynonzero = df_onlynonzero[df_onlynonzero[0] < maxx] # zero speed mask!!
    #         # print("after",len(df_onlynonzero), f'{kvp[0]}'.split('-')[1])
    #         axs_rmsper[uu].scatter(df_onlynonzero.iloc[:,1], df_onlynonzero.iloc[:,0], c='C{}'.format(uu), label = f'{kvp[0]}'.split('-')[1] )

    #         axs_rmsper[uu].set_xlabel('Speed(mps)')
    #         axs_rmsper[uu].grid(True)
    #         axs_rmsper[uu].legend(loc="upper left")
    # axs_rmsper[0].set_ylabel('RMS Doppler Spread (Hz)')
    # plt.tight_layout()
    # plt.figure("rmsperbssep").savefig(f"{overall_plots_dir}" +"/"+f"{runtime}_{fnm}"+"_rmsperbssep.pdf",format='pdf')
    # plt.close("rmsperbssep")



def plot_all_off_dictionaries(ff, fn, summary_cfo_dict, overall_plots_dir, runtime, degreeforfitting, cfo_mthd):           # freqoff_time_dict, mean_frqoff_perrx_dict, exp_start_timestampUTC): #freqoff_dict, freqoff_dist_dict,
    marker_list=["o","s","P","^","*","x"]

    mean_frqoff_perrx_dict      = summary_cfo_dict['meanmethod']
    freqoff_time_dict           = summary_cfo_dict['allcfotime']
    exp_start_timestampUTC      = summary_cfo_dict['exp_start_timestampUTC']


    #### ALL BS in one offset vs time 
    # print("fn is ====", fn)
    ds_numbr_is = names_to_ds_dict[fn]

    fig, ax = plt.subplots(figsize=(10,5), num = "ZeroSpeedOffset_overtime_allBSin1")   
    ax.set_rasterized(True)

    times = []
    lenn = []
    
    for i, kvp in enumerate(freqoff_time_dict.items()):
        vv=[]
        tt=[]
        tt_unaltered=[]
        vals = kvp[1]
        if len(vals) >= degreeforfitting+1: # len(vals) != 0:
            for j in range(len(vals)):
                vv.append(vals[j][0])
                
                delta_t_since_start = convert_strptime_to_currUTCtsdelta(vals[j][1], exp_start_timestampUTC)
                tt.append(   round(delta_t_since_start/3600 , 2) ) 
                tt_unaltered.append(   delta_t_since_start  )

            xfit = np.linspace(min(tt_unaltered), max(tt_unaltered), 100)
            polynomia = np.poly1d(fit_freq_on_time(tt_unaltered, vv, degreeforfitting)) 
            plt.plot(tt, vv, ls=':', c='C{}'.format(i), alpha=0.9-(.1*i), marker=marker_list[i], markersize=5+(3*i), markeredgecolor='k', label=f"{kvp[0].split('-')[1]}: {len(vals)} CFOs")            
            plt.plot( np.round(xfit/3600 , 2), polynomia(xfit), ls='-', c='C{}'.format(i), alpha=0.9-(.1*i), label=f'{degreeforfitting}rd deg. polynomial fit') 
            times.append(np.array(tt).max())
            # lenn.append(len(vals))
        
        else:
            plt.plot([], [], ls=':', c='C{}'.format(i), alpha=0.7, marker=marker_list[i], markersize=5+(3*i), markeredgecolor='k', label=f"{kvp[0].split('-')[1]}: {len(vals)} CFOs")
            plt.plot([], [], label=f'No fitting performed') 
        


    bb = np.round(np.linspace(0, max(times), 10 ), 2)
    plt.xticks(bb, bb,  rotation=90)
    plt.legend()
    plt.xlabel(f"Elapsed time (hours) during dataset {ds_numbr_is} collected on date {fn.split('_')[0]}.")#+f' D{ff}') 
    plt.ylabel("Frequency offset (Hz)")
    plt.grid(alpha=0.7)
    plt.tight_layout()
    print("\nCFO plots saved!\n")
    plt.figure("ZeroSpeedOffset_overtime_allBSin1").savefig(f"{overall_plots_dir}" +"/"+f"{ds_numbr_is}_{fn}"+f{cfo_mthd}+f'{int(time.time())}'+"_ZeroSpeedOffset_overtime_allBSin1.pdf",format='pdf')
    # plt.figure("ZeroSpeedOffset_overtime_allBSin1").savefig(f"{overall_plots_dir}" +"/"+f"{ds_numbr_is}_{fn}"+"_ZeroSpeedOffset_overtime_allBSin1.svg",format='svg', dpi=1200) 
    # plt.figure("ZeroSpeedOffset_overtime_allBSin1").savefig(f"{overall_plots_dir}" +"/"+f"{ds_numbr_is}_{fn}"+"_ZeroSpeedOffset_overtime_allBSin1.eps", dpi=1000)
    # plt.figure("ZeroSpeedOffset_overtime_allBSin1").savefig(f"{overall_plots_dir}" +"/"+f"{ds_numbr_is}_{fn}"+"_ZeroSpeedOffset_overtime_allBSin1.eps",format='eps', dpi=1200)
    # plt.figure("ZeroSpeedOffset_overtime_allBSin1").savefig(f"{overall_plots_dir}" +"/"+f"{ds_numbr_is}_{fn}"+"_ZeroSpeedOffset_overtime_allBSin1.eps", dpi=800)
    plt.close("ZeroSpeedOffset_overtime_allBSin1")
    

###############################################################
###############################################################  ###############################################################
###############################################################  ###############################################################
###############################################################  ###############################################################
###############################################################  ###############################################################
###############################################################  ###############################################################
###############################################################  

# def plot_spectrum_full_5x1(df, this_exp_mtdata):

#     ns        = this_exp_mtdata['narrow_spectrum']

#     nsamps    = this_exp_mtdata['nsamps'] #2**17
#     samp_freq = this_exp_mtdata['rate']  #220000
#     freq_full = np.fft.fftshift(np.fft.fftfreq(nsamps, 1/samp_freq))

#     colnames_list  = df.columns.values[:-1]    
    
#     fig, ax = plt.subplots(len(colnames_list), 1,  figsize=(16, 7)) 
#     plt.ion()

#     for r in range(0, len(df)):
#         this_speed = df['speed_postuple'].iloc[r][0]              
        
#         for p in range(0, len(colnames_list)):
            
#             ############################# x,y axis!  ########################################

#             # psdln_ns = np.nan_to_num(np.square(np.abs(df.iloc[r, p])))
#             psdln_ns = df.iloc[r, p]
#             psdDB_ns = np.nan_to_num(10.0 *np.log10(psdln_ns))  # phase_ns   = np.angle(df.iloc[r,p])


#             f_ns     = freq_full[(freq_full >= -ns) & ( freq_full <= ns) ]        
            
#             ax[p].clear()
#             ax[p].plot(f_ns, psdDB_ns, '--', color = 'r' if this_speed==0 else 'g', label = f"{r}/{df.shape[0]}: Spec at {colnames_list[p].split('-')[1]}")
#             ax[p].legend(loc='upper left')
#             ax[p].set_ylim(-80,20)
#             ax[p].set_ylabel("Magnitude^2")
#             ax[p].set_xlabel("Freq(hz)")
#             ax[p].grid(True)

#         plt.suptitle(f'{r}/{df.shape[0]}, Speed = {this_speed} mps')                               
#         plt.tight_layout()
#         plt.draw()
#         plt.pause(0.01)

#     plt.ioff()
#     plt.close()

# def plot_spectrum_per_rx_2x2(fn, df, this_exp_mtdata, summary_cfo_dict, PWR_THRESHOLD, degreeforfitting):#, FD_MAXPOSSIBLE):
#     fnm = fn.split('meas_')[1].split('.pickle')[0]

#     nsamps    = this_exp_mtdata['nsamps'] #2**17
#     samp_freq = this_exp_mtdata['rate']  #220000
    
#     freq_res  = samp_freq / nsamps
#     freq_full = np.fft.fftshift(np.fft.fftfreq(nsamps, 1/samp_freq))
    
#     fitd_feqoff_perrx_dict      = summary_cfo_dict['fitdmethod']
#     mean_frqoff_perrx_dict      = summary_cfo_dict['meanmethod']
#     frqoff_time_perrx_dict      = summary_cfo_dict['allcfotime']
#     this_exp_start_timestampUTC = summary_cfo_dict['exp_start_timestampUTC']
    
    
    
#     colnames_list  = df.columns.values[:-1]    
#     rmsdict   = {key: [] for key in colnames_list}

#     print("sample psd value is",df.iloc[0,1][0], "len =", len(df.iloc[0,1]))
    

#     fig,ax = plt.subplots(2,2, figsize=(16, 7))
#     # [ax[0][1].scatter(all_BS_coords[bs.split('-')[1]][1], all_BS_coords[bs.split('-')[1]][0], marker='1', s=100, label= bs.split('-')[1] ) for bs in colnames_list ]    
#     plt.ion()

#     for r in range(0, len(df)):
#         this_speed = df['speed_postuple'].iloc[r][0]         # fig, axx = plt.subplots(1, 5, figsize=(15, 3), num="narrowspectrum", sharey= True)            
        
#         for p in range(0, len(colnames_list)-1):
            

#             ############################# x,y axis!  ########################################

#             # psdln_ns = np.nan_to_num(np.square(np.abs(df.iloc[r, p])))
#             psdln_ns = df.iloc[r, p]
#             psdDB_ns = np.nan_to_num(10.0 *np.log10(psdln_ns))  # phase_ns   = np.angle(df.iloc[r,p])


#             aa_psd_max = psdDB_ns[np.argmax(psdDB_ns)]
            
#             lenofpsd = len(psdDB_ns)
#             ns_frq_res = freq_res* np.floor(lenofpsd/2) #freq_res* np.ceil(lenofpsd/2)
#             f_ns = freq_full[(freq_full >= -ns_frq_res) & ( freq_full <= ns_frq_res) ]        
            
#             ############################# cfo re-est for polting ########################### 

#             this_measr_timeuptoseconds = df['speed_postuple'].iloc[r][-1] 
#             this_delta_t_since_start   = convert_strptime_to_currUTCtsdelta(this_measr_timeuptoseconds, this_exp_start_timestampUTC)

#             if (np.all(fitd_feqoff_perrx_dict[colnames_list[p]]!=None)) and (len(frqoff_time_perrx_dict[colnames_list[p]]) >= degreeforfitting+1): # that is, if fit hsppened and was proper, that is len(coeff) = deg+1
#                 method               = 'Polyfit'   ## Frequency Correction METHOD 1: polyfit!
#                 foffset_approximated = get_interpolated_foff(fitd_feqoff_perrx_dict[colnames_list[p]], this_delta_t_since_start)[0]
#             else:
#                 method               = 'Mean'      ## Frequency Correction METHOD 3 or 2: averaging
#                 foffset_approximated = mean_frqoff_perrx_dict[colnames_list[p]]                                            
        
#             foffset_approximated     = int(np.ceil(foffset_approximated/freq_res))*freq_res # fix as per the resolutionv # It, though, was average of freqs that came from fft_res, but averaging removed the fft_res effect, so need to do that again in the next line
           
#             ############################# plot and store rms cfo ########################################
            
#             ax[0][0].clear()
#             ax[0][0].plot(f_ns, psdDB_ns, '-o',color = 'r' if this_speed==0 else 'g', label = f"{r}/{df.shape[0]}: Narrow Spec at {colnames_list[p].split('-')[1]}")
#             if aa_psd_max > PWR_THRESHOLD:
                
#                 rmsis = rmsfunction(psdDB_ns, psdln_ns, f_ns, PWR_THRESHOLD)    
                
#                 rmsdict[colnames_list[p]].append([rmsis, this_speed, this_measr_timeuptoseconds, foffset_approximated, df['speed_postuple'].iloc[r][2], df['speed_postuple'].iloc[r][3] , calcDistLatLong( all_BS_coords[colnames_list[p].split('-')[1]], df['speed_postuple'].iloc[r][2:3][0] )])
                
#                 rms_at = AnchoredText(f"RMS:{round(rmsis,4)} Hz \nMaxm noise:{round(PWR_THRESHOLD,2)}", prop=dict(size=10), frameon=True,pad=0.1, loc='upper left')  
#                 rms_at.patch.set_boxstyle("round, pad=0.,rounding_size=0.2")
#                 ax[0][0].add_artist(rms_at)
#             ax[0][0].axhline(y = PWR_THRESHOLD, lw=2, ls='--', c='k', label= f"Power Threshold{PWR_THRESHOLD}")
#             ax[0][0].legend(loc="lower left")
#             # ax[0][0].set_ylim(-80,20)            
#             ax[0][0].set_ylim(-180,-100)

#             ax[0][0].set_xlabel("Freq (hz)")
#             ax[0][0].set_ylabel("Power (dB)")
            
#             lat_y, long_x =  df['speed_postuple'].iloc[r][2][0],  df['speed_postuple'].iloc[r][2][1]
            
#             ax[0][1].scatter( long_x, lat_y, c='k', marker='o', s=10, label='current loc') #c='C{}'.format(), # sns.scatterplot(data=data, x='x', y='y', hue='colors', palette='rainbow')
#             ax[0][1].scatter(all_BS_coords[colnames_list[p].split('-')[1]][1], all_BS_coords[colnames_list[p].split('-')[1]][0], marker='1', s=100, label= colnames_list[p].split('-')[1] ) 
#             # ax[0][1].set_xlim(-111.84629-100, -111.83118+100)
#             # ax[0][1].set_ylim(40.760-100, 40.770+100)
                    
#             ax[1][1].clear()
#             tts = []
#             ffs = []
#             if (np.all( fitd_feqoff_perrx_dict[colnames_list[p]] !=None)) and (len(frqoff_time_perrx_dict[colnames_list[p]]) >= degreeforfitting+1): # that is, if fit hsppened and was proper, that is len(coeff) = deg+1
#                 for eachfrq, eachtime in frqoff_time_perrx_dict[colnames_list[p]]:
#                     delta_t_since_start = convert_strptime_to_currUTCtsdelta(eachtime, this_exp_start_timestampUTC)
#                     tts.append(   delta_t_since_start  )
#                     ffs.append(eachfrq)                
                
#                 # print('yes', r, colnames_list[p].split('-')[1], len(tts), len(ffs))
                
#                 ax[1][1].plot(tts, ffs, '-o', c='#FFB6C1', label=f'all {len(ffs)} offsets at { colnames_list[p]}')
#                 xfit = np.linspace(min(tts), max(tts), 100)               
#                 polynomia = np.poly1d(fit_freq_on_time(tts, ffs, degreeforfitting))  
#                 ax[1][1].plot(xfit, polynomia(xfit), c='b', label=f'deg {degreeforfitting} polynomial')  
#                 ax[1][1].scatter(convert_strptime_to_currUTCtsdelta(this_measr_timeuptoseconds, this_exp_start_timestampUTC), foffset_approximated,  marker='*', s=50 ,  color='r' if this_speed==0 else 'g', label = f"Current CFO: {round(foffset_approximated,2)} {method}")             
#                 ax[1][1].legend(loc='upper right')
#                 ax[1][1].set_xlim(min(tts), max(tts))
#                 ax[1][1].set_ylim(min(ffs)-100, max(ffs)+100)
        
#             ax[1][1].set_xlabel("Time spent (s)")
#             ax[1][1].set_ylabel("Freq offset (Hz)")

#             [ea.grid(True) for ea in ax.flatten()]
#             plt.suptitle(f'{r}/{df.shape[0]}, Est CFO by {method}: {round(foffset_approximated,2)} Hz, \n Speed: {this_speed} mps,  {fnm}')                               
#             plt.tight_layout()
#             plt.draw()
#             plt.pause(0.001)

#     plt.ioff()
#     plt.close()

#     my_plot_rms_dicts(rmsdict)



############################################################
############################################################ 


def plot_spectrum_per_rx_1x1(fn, df, this_exp_mtdata, summary_cfo_dict, PWR_THRESHOLD, degreeforfitting):#, FD_MAXPOSSIBLE):
    fnm = fn.split('meas_')[1].split('.pickle')[0]
    

    nsamps    = this_exp_mtdata['nsamps'] #2**17
    samp_freq = this_exp_mtdata['rate']  #220000
    
    freq_res  = samp_freq / nsamps
    freq_full = np.fft.fftshift(np.fft.fftfreq(nsamps, 1/samp_freq))

    ns        = this_exp_mtdata['narrow_spectrum']
    f_ns     = freq_full[(freq_full >= -ns) & ( freq_full <= ns) ]     

    fitd_feqoff_perrx_dict      = summary_cfo_dict['fitdmethod']
    mean_frqoff_perrx_dict      = summary_cfo_dict['meanmethod']
    frqoff_time_perrx_dict      = summary_cfo_dict['allcfotime']
    this_exp_start_timestampUTC = summary_cfo_dict['exp_start_timestampUTC']
    
    # print("fnm is", fnm, this_exp_start_timestampUTC)
    plot_all_off_dictionaries(fnm, summary_cfo_dict)



    # colnames_list  = df.columns.values[:-1]    
    # fig,ax = plt.subplots(1,1, figsize=(10, 5))
    # rmsdict   = {key: [] for key in colnames_list}

    # plt.ion()
    # print("sample psd value is",df.iloc[0,1][0], "len =", len(df.iloc[0,1]))

    # for r in range(0, len(df)):
    #     this_speed = df['speed_postuple'].iloc[r][0]         # fig, axx = plt.subplots(1, 5, figsize=(15, 3), num="narrowspectrum", sharey= True)            
        
    #     for p in range(0, len(colnames_list)-1):
            

    #         ############################# x,y axis!  ########################################

    #         psdln_ns = df.iloc[r, p]

    #         psdDB_ns = np.nan_to_num(10.0 *np.log10(psdln_ns))  # phase_ns   = np.angle(df.iloc[r,p])
    #         aa_psd_max = psdDB_ns[np.argmax(psdDB_ns)]

    #         ax.clear()
    #         ax.plot(f_ns, psdDB_ns, '-o',color = 'r' if this_speed==0 else 'g', label = f"{r}/{df.shape[0]}: Narrow Spec at {colnames_list[p].split('-')[1]}")
    #         ax.axhline(y = PWR_THRESHOLD, lw=2, ls='--', c='k', label= f"Power Threshold{PWR_THRESHOLD}")

            
    #         ############################# cfo re-est for polting ########################### 

    #         this_measr_timeuptoseconds = df['speed_postuple'].iloc[r][-1] 
    #         this_delta_t_since_start   = convert_strptime_to_currUTCtsdelta(this_measr_timeuptoseconds, this_exp_start_timestampUTC)

    #         if (np.all(fitd_feqoff_perrx_dict[colnames_list[p]]!=None)) and (len(frqoff_time_perrx_dict[colnames_list[p]]) >= degreeforfitting+1): # that is, if fit hsppened and was proper, that is len(coeff) = deg+1
    #             method               = 'Poly'   ## Frequency Correction METHOD 1: polyfit!
    #             foffset_approximated = get_interpolated_foff(fitd_feqoff_perrx_dict[colnames_list[p]], this_delta_t_since_start)[0]
    #         else:
    #             method               = 'Mean'      ## Frequency Correction METHOD 3 or 2: averaging
    #             foffset_approximated = mean_frqoff_perrx_dict[colnames_list[p]]                                            
        
    #         foffset_approximated     = int(np.ceil(foffset_approximated/freq_res))*freq_res # fix as per the resolutionv # It, though, was average of freqs that came from fft_res, but averaging removed the fft_res effect, so need to do that again in the next line
           

    #         if aa_psd_max > PWR_THRESHOLD:
                
    #             rmsis = rmsfunction(psdDB_ns, psdln_ns, f_ns, PWR_THRESHOLD)    
                
    #             rmsdict[colnames_list[p]].append([rmsis, this_speed, this_measr_timeuptoseconds, foffset_approximated, df['speed_postuple'].iloc[r][2], df['speed_postuple'].iloc[r][3] , calcDistLatLong( all_BS_coords[colnames_list[p].split('-')[1]], df['speed_postuple'].iloc[r][2:3][0] )])
                
    #             rms_at = AnchoredText(f"RMS:{round(rmsis,4)} Hz \nMaxm noise:{round(PWR_THRESHOLD,2)}", prop=dict(size=10), frameon=True,pad=0.1, loc='upper left')  
    #             rms_at.patch.set_boxstyle("round, pad=0.,rounding_size=0.2")
    #             ax.add_artist(rms_at)

    #         # ax[0].set_ylim(-80,20)
    #         ax.set_ylim(-180,-100)
    #         ax.grid(True)
    #         ax.set_xlabel("Freq (Hz)")
    #         ax.set_ylabel("Power (dB)/Hz") 
    #         ax.legend(loc="lower left")
    #         plt.suptitle(f'{r}/{df.shape[0]}, Est CFO by {method}: {round(foffset_approximated,2)} Hz, \n Speed: {this_speed} mps,  {fnm}')                               
    #         plt.tight_layout()
    #         plt.draw()
    #         plt.pause(0.001)

    # plt.ioff()
    # plt.close()
    # my_plot_rms_dicts(rmsdict)
    # plot_all_off_dictionaries(summary_cfo_dict)





############################################################
############################################################    

if __name__ == "__main__":

    ############################################################
    #### mine! that is psd! after storing processing and result generation!
    ############################################################
   
    print("post processing Time right now  = ", datetime.now().strftime("%H:%M:%S"))
    PSDDATADIR = Path.cwd()
    print('Your current working directory that has all pickle files is:', Path.cwd(), "\n")

    all_pickled_psd_tagged_files = list(sorted(Path(PSDDATADIR).rglob('*.pickle')))
    # print("Files are = "), [print(ff) for ff in all_pickled_psd_tagged_files], print("\nTotal # of files in folder= ", len(all_pickled_psd_tagged_files), "\n")


    totalrows = 0
    one_totaldataset_df= pd.DataFrame()


    if not CFO_REMOVE_FLAG:
        print("no cfo removed flag set")
        

    for i, filename in enumerate(all_pickled_psd_tagged_files):        
        fn = '_'.join( str(filename).split('Shout_')[1].split('_')[0:3] )
       
        loaded_data = pkl.load(open(filename, 'rb') )
        print("----------------------------------------------------------------------\n",i+1, fn, pd.DataFrame(loaded_data[0]).shape, end="")

        ## Metadata of each experiment
        this_exp_metadata = loaded_data[1]

        ## Data for each experiment
        this_exp_data = loaded_data[0]
        this_exp_df   = pd.DataFrame(this_exp_data)


        ## detour temp fix
        indexofdetour = this_exp_df.shape[0] #if fn != 'meas_02-14-2023_18-53-20' else 50
        print("length of this experiment data:", this_exp_df.iloc[:indexofdetour,:].shape[0])
        this_exp_cfos = loaded_data[2] 
        plot_spectrum_per_rx_1x1 (fn, this_exp_df.iloc[:indexofdetour,:], this_exp_metadata, this_exp_cfos, PWR_THRESHOLD, DEGREE_OF_POLYFIT)#, FD_MAXPOSSIBLE)  # quickplot_labels_non_utmed_per_df(i, fn, this_exp_df.iloc[:indexofdetour]) 
        
        # ## plots for each experiment
        # ##from: mine_GPSlocs_plotting_function
        # get_filtered_df_and_plot(fn, this_exp_df, plotflag =True) 


        
        # ##### from: mine_plots_rms_cfo_spectrum_loc
        # if CFO_REMOVE_FLAG: 
        #     ## cfo of each experiment
        #     this_exp_cfos= loaded_data[2] 

        #     # plot_spectrum_per_rx_2x2 (fn, this_exp_df.iloc[:indexofdetour,:], this_exp_metadata, this_exp_cfos, PWR_THRESHOLD, DEGREE_OF_POLYFIT)#, FD_MAXPOSSIBLE)  # quickplot_labels_non_utmed_per_df(i, fn, this_exp_df.iloc[:indexofdetour]) 
        #     plot_spectrum_per_rx_1x1 (fn, this_exp_df.iloc[:indexofdetour,:], this_exp_metadata, this_exp_cfos, PWR_THRESHOLD, DEGREE_OF_POLYFIT)#, FD_MAXPOSSIBLE)  # quickplot_labels_non_utmed_per_df(i, fn, this_exp_df.iloc[:indexofdetour]) 

        # else: 
        #     plot_spectrum_full_5x1(this_exp_df.iloc[:indexofdetour,:], this_exp_metadata) 
        


        ## concatenate
        one_totaldataset_df = pd.concat([one_totaldataset_df, this_exp_df.iloc[:indexofdetour, :]])
        totalrows += this_exp_df.shape[0]    
        
        print("\nCurrently, the dataframe stacked with all data ====>", totalrows)

    print("\nFinally, the dataframe stacked with all data ====>",
          "\nfull_data_df shape:", one_totaldataset_df.shape, 
          "\nfull_data_df column names:", one_totaldataset_df.columns, 
          "\nfull_data_df example data label:", one_totaldataset_df.iloc[95,-1], 
          "\nfull_data_df each data length:", one_totaldataset_df.iloc[95][0].shape)  # print("length should be", 500*11+200+100+100+1500+499+499+365+495+475+50+156+162, 10101)
    
    


    print("Time now  = ", datetime.now().strftime("%H:%M:%S"))
    print("\n\n\nDONE!!!!\n\n")