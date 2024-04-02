#
# LICENSE:
# Copyright (C) 2024  Aarti Singh
#
# Open source license, please see LICENSE.md
# 
# Author: Aarti Singh, aartisingh@wustl.edu
#
# Version History:

# Version 1.0:  Initial Release. 20 Jan 2024
# Version 2.0:  Initial Release. 30 Mar 2024

#

from libs import * 
from GPSlocs_plotting_functions import *


def getrmsFreq(dspd,frqs):
    total_power = np.sum(dspd)
    fractional_power = dspd/total_power
    bar_frqs = np.sum(fractional_power*frqs) #== mean of Dshift
    var_frqs = np.sum(fractional_power*(frqs-bar_frqs)**2) 
    rmsFreqval = np.sqrt(var_frqs) #== std=sqrt(variance) of Dshift 
    return round(rmsFreqval, 2)

def rmsfunction(psd_db_ns, psd_linear_ns, freqs_ns, PWR_THRESHOLD):  
    max_noise_power_db = PWR_THRESHOLD # np.max([np.max(psd_db_ns[noise_window_left]), np.max(psd_db_ns[ noise_window_rght ])])    
    max_noise_power_linear = 10**(max_noise_power_db/10)

    rmsis_entire = getrmsFreq(psd_linear_ns, freqs_ns)
    rmsis_thrsld = getrmsFreq(psd_linear_ns[psd_linear_ns > max_noise_power_linear], freqs_ns[psd_linear_ns > max_noise_power_linear])
    return rmsis_thrsld  

def my_plot_rms_dicts(fnm, rmsdict, overall_plots_dir, runtime, saveplots_flag):
    
    fig_rmsall, axs_rmsall = plt.subplots(figsize=(6,4), num = "rmsall")
    for uu, kvp in enumerate(rmsdict.items()):
        if len(kvp[1]) !=0 and kvp[0].split('-')[1] =='ustar':
            rms_df_d = pd.DataFrame()
            rms_df_d = pd.DataFrame(kvp[1]).apply(lambda x: pd.Series(x) )  
    
            # plt.scatter(rms_df_d.iloc[:,1], rms_df_d.iloc[:,0], label = f'{kvp[0]}'.split('-')[1] )
            
            # zero speed mask!!
            df_onlynonzero = rms_df_d[rms_df_d[0] != 0] 
            maxx = min(df_onlynonzero.iloc[:,0].nlargest(2))
            df_onlynonzero = df_onlynonzero[df_onlynonzero[0] < maxx] # zero speed mask!!
            axs_rmsall.scatter(df_onlynonzero.iloc[:,1], df_onlynonzero.iloc[:,0], c='C{}'.format(uu), label = f'{kvp[0]}'.split('-')[1] )
    
    plt.xlabel('Speed(mps)')
    plt.grid(True)
    plt.legend(loc="upper left")
    plt.ylabel('RMS Doppler Spread (Hz)')
    plt.tight_layout()
    if not saveplots_flag:
        plt.show()
    else:
        plt.figure("rmsall").savefig(f"{overall_plots_dir}" +"/"+f"{runtime}_{fnm}"+"_rmsall.svg",format='svg', dpi=1200) #.pdf",format='pdf')
        print(f"\n\nRMS plots saved at loc: {overall_plots_dir}!")
        plt.close("rmsall")    



######################################## Plot per RX spectrum and MT location   #######################
#########################################################################################################
def plot_everyRXspectrum_VS_MTloc_1x2(fnm, df, routewas, this_exp_mtdata, summary_cfo_dict , overall_plots_dir='./', runtim=0, saveplots_flag=0):
    
    print(f"\nCurrent filename: {fnm} with number of observations stored:{df.shape[0]}" )
    print("\n\nPlotting on screen with plt.ion() mode .....")

    if saveplots_flag:
        print("Storing in these subdirectories")
        overall_plots_dir.mkdir(parents=True, exist_ok=True)

        runtime_plots_dir = Path(str(overall_plots_dir)+"/"+f'plots_at_runtime_{runtime}')
        runtime_plots_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nStoring in these subdirectories{runtime_plots_dir}")

    pwr_threshold               = summary_cfo_dict['pwr_threshold']
    # degreeforfitting            = summary_cfo_dict['degreeforfitting'] # no need to extract for now
    fitd_feqoff_perrx_dict      = summary_cfo_dict['fitdmethod']
    mean_frqoff_perrx_dict      = summary_cfo_dict['meanmethod']
    frqoff_time_perrx_dict      = summary_cfo_dict['allcfotime']
    this_exp_start_timestampUTC = summary_cfo_dict['exp_start_timestampUTC']


    ns        = this_exp_mtdata['narrow_spectrum']
    nsamps    = this_exp_mtdata['nsamps'] #2**17
    samp_freq = this_exp_mtdata['rate']  #220000
    freq_res  = samp_freq / nsamps
    freq_full = np.fft.fftshift(np.fft.fftfreq(nsamps, 1/samp_freq))
    f_ns      = freq_full[(freq_full >= -ns) & ( freq_full <= ns) ]

    if routewas =='orange':
        routeclr = '#ffb16d'
        routename = 'O' #B
    else:
        routeclr = '#33b864' #'#2a7e19' #
        routename = 'G' #A

    # apply UTM on BS locs
    all_BS_coords_df = pd.DataFrame(all_BS_coords.values(), columns=['Latitude', 'Longitude'], index=all_BS_coords.keys())
    all_BS_coords_df = all_BS_coords_df.apply(lambda row: pd.Series(latlon_to_utm(row)), axis=1)  #c0=north and c1=east

    # apply UTM on MT locs
    allMTlocs_df = df.iloc[:,-1].apply(lambda row: latlon_to_utm(row[2])).apply(pd.Series)

    colnames_list  = df.columns.values[:-1] 
    rmsdict =   {key: [] for key in colnames_list}
    
    frst, last = 1, 2
    fig, ax = plt.subplots(frst, last, figsize=(8, 3), num = "psdVsloc", sharey = False, sharex = False) #sharey = True, sharex = True)
    l=last-1  

    
    plt.ion()
    # [ax[l].scatter( utm.from_latlon(all_BS_coords[bs.split('-')[1]][0], all_BS_coords[bs.split('-')[1]][1])[0:2][0], utm.from_latlon(all_BS_coords[bs.split('-')[1]][0], all_BS_coords[bs.split('-')[1]][1])[0:2][1], marker='1', c='C{}'.format(uu+103), s=100, label= bs.split('-')[1][:3] ) for uu, bs in enumerate(colnames_list)]  #FFB6C1, #ffb16d
            
    for r in range(0, len(df)):
        this_speed = df['speed_postuple'].iloc[r][0]         # fig, axx = plt.subplots(1, 5, figsize=(15, 3), num="narrowspectrum", sharey= True)            
        
        for p in range(0, len(colnames_list)):#-1): this is annoying!

            ############################# x,y axis!  ########################################

            psdln_ns = np.nan_to_num(np.square(np.abs(df.iloc[r, p])))
            psdDB_ns = np.nan_to_num(10.0 *np.log10(psdln_ns))  # phase_ns   = np.angle(df.iloc[r,p])    
            aa_psd_max        = psdDB_ns[np.argmax(psdDB_ns)]
            
            ax[0].clear()
            ax[0].plot(f_ns, psdDB_ns, '-o',color = 'r' if this_speed==0 else 'g', label = f"{r}/{df.shape[0]}: Narrow Spec at {colnames_list[p].split('-')[1]}")
            
            if aa_psd_max > pwr_threshold:
                rmsis = rmsfunction(psdDB_ns, psdln_ns, f_ns, pwr_threshold)
                
                rmsdict[df.columns[p]].append([rmsis, this_speed, None, None, None])
                
                rms_at = AnchoredText(f"RMS:{round(rmsis,2)} Hz\nSpeed:{round(this_speed,1)} mps", prop=dict(size=8), frameon=True,pad=0.1, loc='upper left') 
                rms_at.patch.set_boxstyle("round, pad=0.,rounding_size=0.2")
               
                ax[0].add_artist(rms_at)

            ax[0].axhline(y = pwr_threshold, lw=1, ls='--', c='k', label= f"Noise Threshold: {pwr_threshold}") #\nEst. CFO: {round(cfo_approx,2)} Hz
            ax[0].legend(loc="lower left")
            ax[0].set_ylim(-80,20)
            
            # ax[0].set_xlim(f_ns[0],f_ns[-1])
            ax[0].set_xlabel("Freq (Hz)")  
            ax[0].set_ylabel("Power (dB)")


            ax[l].clear()

            # All MT locs scattered!
            # pdb.set_trace() 42_y 45_x
            allMTlocs_df.plot.scatter(x=1, y=0, c=routeclr, marker='.', s=2, ax=ax[l], label='route')

            #All BS locs scattered!
            [ax[l].scatter( utm.from_latlon(all_BS_coords[bs.split('-')[1]][0], all_BS_coords[bs.split('-')[1]][1])[0:2][0], utm.from_latlon(all_BS_coords[bs.split('-')[1]][0], all_BS_coords[bs.split('-')[1]][1])[0:2][1], marker='1', c='C{}'.format(uu+103), s=100, label= bs.split('-')[1][:3] ) for uu, bs in enumerate(colnames_list)]  #FFB6C1, #ffb16d
            
            lat_y_40s, long_x_11s =  df['speed_postuple'].iloc[r][2][0],  df['speed_postuple'].iloc[r][2][1]
            easting_lngs_x, northing_lats_y = utm.from_latlon(lat_y_40s, long_x_11s)[0:2] # print("easting_lngs_x, northing_lats_y", easting_lngs_x, northing_lats_y, "\n")             
            ax[l].scatter(easting_lngs_x, northing_lats_y, c='k', marker='*', s=12, label="loc" ) #, label= f"|v|:{round(this_speed.values[0],1)} mps"  # [ax[l].scatter( utm.from_latlon(all_BS_coords[bs.split('-')[1]][0], all_BS_coords[bs.split('-')[1]][1])[0:2][0], utm.from_latlon(all_BS_coords[bs.split('-')[1]][0], all_BS_coords[bs.split('-')[1]][1])[0:2][1] , marker='1', s=100, label= bs.split('-')[1] )    for bs in df_allrx.columns.values]

            [ea.grid(True) for ea in ax.flatten()]            
            
            # xl = all_BS_coords_df.loc['bes'][0]-100
            # xh = all_BS_coords_df.loc['smt'][0]+600
            
            # yl = all_BS_coords_df.loc['bes'][1]-500
            # yh = all_BS_coords_df.loc['hospital'][1]+500
            
            # ax[l].set_xlim(xl, xh)
            # ax[l].set_ylim(yl, yh)
            
            xlbs = [i- ax[l].get_xticks()[1] for i in ax[l].get_xticks() ]
            ax[l].set_xticklabels(xlbs)
            ylbs = [i- ax[l].get_yticks()[1] for i in ax[l].get_yticks() ]
            ax[l].set_yticklabels(ylbs)
            
            ax[l].yaxis.tick_right()
            ax[l].yaxis.set_label_position("right")
            ax[l].set_ylabel("UTM Northing (m)")
            ax[l].set_xlabel("UTM Easting (m)")
            ax[l].legend(loc='lower right')
            plt.tight_layout()
            


            ### plt.suptitle(f'{r}/{df.shape[0]}, Speed = {this_speed} mps, Est CFO: {round(foffset_approximated,2)} Hz')                               


            # ############################# cfo re-est for polting ########################### 

            # this_measr_timeuptoseconds = df['speed_postuple'].iloc[r][-1] 
            # this_delta_t_since_start   = convert_strptime_to_currUTCtsdelta(this_measr_timeuptoseconds, this_exp_start_timestampUTC)

            # if (np.all(fitd_feqoff_perrx_dict[colnames_list[p]]!=None)) and (len(frqoff_time_perrx_dict[colnames_list[p]]) >= degreeforfitting+1): # that is, if fit hsppened and was proper, that is len(coeff) = deg+1
            #     method               = 'Polyfit'   ## Frequency Correction METHOD 1: polyfit!
            #     foffset_approximated = get_interpolated_foff(fitd_feqoff_perrx_dict[colnames_list[p]], this_delta_t_since_start)[0]
            # else:
            #     method               = 'Mean'      ## Frequency Correction METHOD 3 or 2: averaging
            #     foffset_approximated = mean_frqoff_perrx_dict[colnames_list[p]]                                            
        
            # foffset_approximated     = int(np.ceil(foffset_approximated/freq_res))*freq_res # fix as per the resolutionv # It, though, was average of freqs that came from fft_res, but averaging removed the fft_res effect, so need to do that again in the next line
           
                    
            # ax[2].clear()
            # tts = []
            # ffs = []
            # if (np.all( fitd_feqoff_perrx_dict[colnames_list[p]] !=None)) and (len(frqoff_time_perrx_dict[colnames_list[p]]) >= degreeforfitting+1): # that is, if fit hsppened and was proper, that is len(coeff) = deg+1
            #     for eachfrq, eachtime in frqoff_time_perrx_dict[colnames_list[p]]:
            #         delta_t_since_start = convert_strptime_to_currUTCtsdelta(eachtime, this_exp_start_timestampUTC)
            #         tts.append(   delta_t_since_start  )
            #         ffs.append(eachfrq)                

            #     ax[2].plot(tts, ffs, '-o', c='#FFB6C1', label=f'all {len(ffs)} offsets at { colnames_list[p]}')
            #     xfit = np.linspace(min(tts), max(tts), 100)               
            #     polynomia = np.poly1d(fit_freq_on_time(tts, ffs, degreeforfitting))  
            #     ax[2].plot(xfit, polynomia(xfit), c='b', label=f'deg {degreeforfitting} polynomial')  
            #     ax[2].scatter(convert_strptime_to_currUTCtsdelta(this_measr_timeuptoseconds, this_exp_start_timestampUTC), \
            #         foffset_approximated,  marker='*', s=50 ,  color='r' if this_speed==0 else 'g', 
            #         label = f"Current CFO: {round(foffset_approximated,2)} by {method} method")             
            #     ax[2].legend(loc='upper right')
            #     # ax[2].set_xlim(0, 9000)
            #     ax[2].set_xlim(min(tts), max(tts))
            #     ax[2].set_ylim(min(ffs)-100, max(ffs)+100)
        
            # ax[2].set_xlabel("Time spent (s)")
            # ax[2].set_ylabel("Freq offset (Hz)")

            if p==4:
                if saveplots_flag:
                    plt.figure("psdVsloc").savefig(f"{runtime_plots_dir}" +"/"+f"{r}_{p}_{fnm}_"+"psdVsloc.svg",format='svg', dpi=1200)  #.pdf",format='pdf')
                else:
                    plt.draw()
                    plt.pause(0.01) 

    plt.ioff()
    plt.close()
    my_plot_rms_dicts(fnm, rmsdict, runtime_plots_dir, runtime, saveplots_flag)

if __name__ == "__main__":

    ############################################################
    ####  After storing! processing and result generation!
    ############################################################
    print('Your current working directory that has all pickle files is:', Path.cwd(), "\n")
    print("Time right now  = ", datetime.now().strftime("%H:%M:%S"))

    PSDDATADIR = str(Path.cwd())+'/IQ_pickles_forGit_v2_10152_oldcfofunc_373highsnrKept_detournotyetfiltered'
    print('The directory of pickle files is:', PSDDATADIR, "\n")

    psd_tagged_files = list(sorted(Path(PSDDATADIR).rglob('*.pickle')))
    print("Files are = "), [print(ff) for ff in psd_tagged_files], print("\nTotal # of files in folder= ", len(psd_tagged_files), "\n")


    saveplots_flag = 0

    overall_plots_dir = Path(PSDDATADIR+'/all_psds_plots')
    runtime = f'{int(time.time())}'
    print("current runtime is:", runtime)


    totalrows = 0  
    one_totaldataset_df    = pd.DataFrame()
    
    for i, filename in enumerate(psd_tagged_files):        
        fn = '_'.join( str(filename).split('Shout_')[1].split('_')[0:3] )
       
        loaded_data = pkl.load(open(filename, 'rb') )
        
        print("----------------------------------------------------------------------",i+1, fn, pd.DataFrame(loaded_data[0]).shape, end="")
        ## Cfo of each experiment
        this_exp_cfos = loaded_data[2] 

        ## Metadata of each experiment
        this_exp_metadata = loaded_data[1]

        ## Data for each experiment
        this_exp_data = loaded_data[0]
        this_exp_df   =  pd.DataFrame(this_exp_data)
        # print(f"\nOriginal length of this experiment data: {this_exp_df.shape[0]}")
        ## Detour fix: Will reduce the size of the df! **AND** plot if plotflag=True
        routewas = get_filtered_df_and_plot(fn, this_exp_df, plotflag =False) # from: mine_GPSlocs_plotting_function
        # print(f"Filtered length of this experiment data:", this_exp_df.shape[0])

        plot_everyRXspectrum_VS_MTloc_1x2(fn, this_exp_df, routewas, this_exp_metadata, this_exp_cfos, overall_plots_dir, runtime, saveplots_flag)
        ## concatenate
        one_totaldataset_df = pd.concat([one_totaldataset_df, this_exp_df])
        totalrows += this_exp_df.shape[0]  
        # break  
        
        print("\nCurrently, the dataframe stacked with all data ====>", totalrows)

    print("\nFinally, the dataframe stacked with all data ====>",
          "\nfull_data_df shape:", one_totaldataset_df.shape, 
          "\nfull_data_df column names:", one_totaldataset_df.columns.values, 
          "\nExample sample of IQ data with label at 95th row:", one_totaldataset_df.iloc[95,-1], 
          "\nEach IQ data is of vector size:", one_totaldataset_df.iloc[95][0].shape) # print("over number of rows should be", 500*11+200+100+100+1500+499+499+365+495+475+50+156+162????)
    
    print("Time now  = ", datetime.now().strftime("%H:%M:%S"))
    print("\n\n\nDONE!!!!\n\n")