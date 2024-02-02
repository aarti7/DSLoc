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
#

from libs import * 
from functions_cforemoval import *

######################################## BaseStation coordinates ########################################
#########################################################################################################


all_BS_coords = {
  'hospital': (40.77105, -111.83712),  
  'ustar': (40.76899, -111.84167),
  'bes': (40.76134, -111.84629),
  'honors': (40.76440, -111.83695),
  'smt': (40.76740, -111.83118),
}

######################################## Plot per BS spectrum and Calculated CFO #######################
#########################################################################################################

def plot_spectrum_per_rx_1x3(df, this_exp_mtdata, summary_cfo_dict, PWR_THRESHOLD, degreeforfitting):#, FD_MAXPOSSIBLE):

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
    
    colnames_list  = df.columns.values[:-1]    
    fig,ax = plt.subplots(1,3, figsize=(16, 5))
    [ax[2].scatter(all_BS_coords[bs.split('-')[1]][1], all_BS_coords[bs.split('-')[1]][0], marker='1', s=100, label= bs.split('-')[1] ) for bs in colnames_list ]    

    plt.ion()

    for r in range(0, len(df)):
        this_speed = df['speed_postuple'].iloc[r][0]         # fig, axx = plt.subplots(1, 5, figsize=(15, 3), num="narrowspectrum", sharey= True)            
        
        for p in range(0, len(colnames_list)-1):
            

            ############################# x,y axis!  ########################################

            psdln_ns = np.nan_to_num(np.square(np.abs(df.iloc[r, p])))
            psdDB_ns = np.nan_to_num(10.0 *np.log10(psdln_ns))  # phase_ns   = np.angle(df.iloc[r,p])    
            
            ax[0].clear()
            ax[0].plot(f_ns, psdDB_ns, '-o',color = 'r' if this_speed==0 else 'g', label = f"{r}/{df.shape[0]}: Narrow Spec at {colnames_list[p].split('-')[1]}")

            ax[0].axhline(y = PWR_THRESHOLD, lw=2, ls='--', c='k', label= f"Power Threshold{PWR_THRESHOLD}")
            ax[0].legend(loc="lower left")
            ax[0].set_ylim(-80,20)
            ax[0].set_xlabel("Freq (hz)")
            ax[0].set_ylabel("Power (dB)")            
            ############################# cfo re-est for polting ########################### 

            this_measr_timeuptoseconds = df['speed_postuple'].iloc[r][-1] 
            this_delta_t_since_start   = convert_strptime_to_currUTCtsdelta(this_measr_timeuptoseconds, this_exp_start_timestampUTC)

            if (np.all(fitd_feqoff_perrx_dict[colnames_list[p]]!=None)) and (len(frqoff_time_perrx_dict[colnames_list[p]]) >= degreeforfitting+1): # that is, if fit hsppened and was proper, that is len(coeff) = deg+1
                method               = 'Polyfit'   ## Frequency Correction METHOD 1: polyfit!
                foffset_approximated = get_interpolated_foff(fitd_feqoff_perrx_dict[colnames_list[p]], this_delta_t_since_start)[0]
            else:
                method               = 'Mean'      ## Frequency Correction METHOD 3 or 2: averaging
                foffset_approximated = mean_frqoff_perrx_dict[colnames_list[p]]                                            
        
            foffset_approximated     = int(np.ceil(foffset_approximated/freq_res))*freq_res # fix as per the resolutionv # It, though, was average of freqs that came from fft_res, but averaging removed the fft_res effect, so need to do that again in the next line
           
                    
            ax[1].clear()
            tts = []
            ffs = []
            if (np.all( fitd_feqoff_perrx_dict[colnames_list[p]] !=None)) and (len(frqoff_time_perrx_dict[colnames_list[p]]) >= degreeforfitting+1): # that is, if fit hsppened and was proper, that is len(coeff) = deg+1
                for eachfrq, eachtime in frqoff_time_perrx_dict[colnames_list[p]]:
                    delta_t_since_start = convert_strptime_to_currUTCtsdelta(eachtime, this_exp_start_timestampUTC)
                    tts.append(   delta_t_since_start  )
                    ffs.append(eachfrq)                

                ax[1].plot(tts, ffs, '-o', c='#FFB6C1', label=f'all {len(ffs)} offsets at { colnames_list[p]}')
                xfit = np.linspace(min(tts), max(tts), 100)               
                polynomia = np.poly1d(fit_freq_on_time(tts, ffs, degreeforfitting))  
                ax[1].plot(xfit, polynomia(xfit), c='b', label=f'deg {degreeforfitting} polynomial')  
                ax[1].scatter(convert_strptime_to_currUTCtsdelta(this_measr_timeuptoseconds, this_exp_start_timestampUTC), \
                    foffset_approximated,  marker='*', s=50 ,  color='r' if this_speed==0 else 'g', 
                    label = f"Current CFO: {round(foffset_approximated,2)} by {method} method")             
                ax[1].legend(loc='upper right')
                # ax[1].set_xlim(0, 9000)
                ax[1].set_xlim(min(tts), max(tts))
                ax[1].set_ylim(min(ffs)-100, max(ffs)+100)
        
            ax[1].set_xlabel("Time spent (s)")
            ax[1].set_ylabel("Freq offset (Hz)")


            lat_y, long_x =  df['speed_postuple'].iloc[r][2][0],  df['speed_postuple'].iloc[r][2][1]
            ax[2].scatter( long_x, lat_y, c='k', marker='o', s=10) 
            ax[2].legend(loc='upper right')

            [ea.grid(True) for ea in ax.flatten()]
            plt.suptitle(f'{r}/{df.shape[0]}, Speed = {this_speed} mps, Est CFO: {round(foffset_approximated,2)} Hz')                               
            plt.tight_layout()
            plt.draw()
            plt.pause(0.05)

    plt.ioff()
    plt.close()

######################################## Plot All BS spectrums at a time #######################
#########################################################################################################

def plot_spectrum_all_rx_5x1(df, this_exp_mtdata):

    ns        = this_exp_mtdata['narrow_spectrum']

    nsamps    = this_exp_mtdata['nsamps'] #2**17
    samp_freq = this_exp_mtdata['rate']  #220000
    freq_full = np.fft.fftshift(np.fft.fftfreq(nsamps, 1/samp_freq))

    colnames_list  = df.columns.values[:-1]    
    
    fig, ax = plt.subplots(len(colnames_list), 1,  figsize=(16, 7)) 
    plt.ion()

    for r in range(0, len(df)):
        this_speed = df['speed_postuple'].iloc[r][0]              
        
        for p in range(0, len(colnames_list)):
            
            ############################# x,y axis!  ########################################

            psdln_ns = np.nan_to_num(np.square(np.abs(df.iloc[r, p])))
            psdDB_ns = np.nan_to_num(10.0 *np.log10(psdln_ns))  
            # phase_ns   = np.angle(df.iloc[r,p])            
            f_ns     = freq_full[(freq_full >= -ns) & ( freq_full <= ns) ]        
            
            ax[p].clear()
            ax[p].plot(f_ns, psdDB_ns, '--', color = 'r' if this_speed==0 else 'g', label = f"{r}/{df.shape[0]}: Spec at {colnames_list[p].split('-')[1]}")
            ax[p].legend(loc='upper left')
            ax[p].set_ylim(-80,20)
            ax[p].set_ylabel("Magnitude^2")
            ax[p].set_xlabel("Freq(hz)")
            ax[p].grid(True)

        plt.suptitle(f'{r}/{df.shape[0]}, Speed = {this_speed} mps')                               
        plt.tight_layout()
        plt.draw()
        plt.pause(0.05)

    plt.ioff()
    plt.close()

if __name__ == "__main__":

    ############################################################
    ####  After storing! processing and result generation!
    ############################################################
   
    print("Time right now  = ", datetime.now().strftime("%H:%M:%S"))
    PSDDATADIR = Path.cwd()
    print('Your current working directory that has all pickle files is:', Path.cwd(), "\n")

    psd_tagged_files = list(sorted(Path(PSDDATADIR).rglob('*.pickle')))
    print("Files are = "), [print(ff) for ff in psd_tagged_files], print("\nTotal # of files in folder= ", len(psd_tagged_files), "\n")

    totalrows = 0
    one_totaldataset_df    = pd.DataFrame()
    
    CFO_REMOVE_FLAG = args.ofstremove

    for i, filename in enumerate(psd_tagged_files):        
        fn = '_'.join( str(filename).split('Shout_')[1].split('_')[0:3] )
       
        loaded_data = pkl.load(open(filename, 'rb') )
        
        print("----------------------------------------------------------------------\n",i+1, fn, pd.DataFrame(loaded_data[0]).shape)

        ## Metadata of each experiment
        this_exp_metadata = loaded_data[1]

        ## Data for each experiment
        this_exp_data = loaded_data[0]
        this_exp_df   =  pd.DataFrame(this_exp_data)

        ## detour temp fix
        indexofdetour = this_exp_df.shape[0]
        print("length of this experiment data:", this_exp_df.iloc[:indexofdetour,:].shape[0])
        
        if CFO_REMOVE_FLAG:
            ## cfo of each experiment
            this_exp_cfos = loaded_data[2] 
            plot_spectrum_per_rx_1x3(this_exp_df.iloc[:indexofdetour,:], this_exp_metadata, this_exp_cfos, PWR_THRESHOLD, DEGREE_OF_POLYFIT)#, FD_MAXPOSSIBLE)  # quickplot_labels_non_utmed_per_df(i, fn, this_exp_df.iloc[:indexofdetour]) 
        else:
            plot_spectrum_all_rx_5x1(this_exp_df.iloc[:indexofdetour,:], this_exp_metadata) 
        
        ## concatenate
        one_totaldataset_df = pd.concat([one_totaldataset_df, this_exp_df.iloc[:indexofdetour, :]])
        totalrows += this_exp_df.shape[0]    
        
        print("\nCurrently, the dataframe stacked with all data ====>", totalrows)

    print("\nFinally, the dataframe stacked with all data ====>",
          "\nfull_data_df shape:", one_totaldataset_df.shape, 
          "\nfull_data_df column names:", one_totaldataset_df.columns, 
          "\nfull_data_df example data label:", one_totaldataset_df.iloc[95,-1], 
          "\nfull_data_df each data length:", one_totaldataset_df.iloc[95][0].shape) # print("length should be", 500*11+200+100+100+1500+499+499+365+495+475+50+156+162)
    
    print("Time now  = ", datetime.now().strftime("%H:%M:%S"))
    print("\n\n\nDONE!!!!\n\n")