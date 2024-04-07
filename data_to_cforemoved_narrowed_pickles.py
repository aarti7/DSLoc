from libs import * 
from functions_cforemoval import *

from mine_psd_fetching import *
from matplotlib.mlab import psd

###############################################################
###############################################################    

## Default Directories

DEF_DIRDATA    = "./" 
DEF_HDF5NAME   = "measurements.hdf5"
DEF_HDF5BRANCH = "wtx"
MEAS_ROOT      = "saveiq_w_tx"

## HARDCODED VALUE
FRQ_CUTOFF     = 1e4 #in hz

###############################################################
###############################################################  


CFO_REMOVE_FLAG   = 1
MPLOT_FLAG        = 0
PWR_THRESHOLD     = -19 #in dB
DEGREE_OF_POLYFIT = 3

BUFFER_WINDOW     = 1e2 #in hz
MAXIMUM_BUS_SPEED = 21  #in mps
    

######################################## Read hdf5 tree ########
################################################################
def read_leaves_to_DF(leaves, allsampsandtime):
    print('Reading hdf5 tree......',end='')

    endpoints=[]
    df_allrx= pd.DataFrame()
    df_allti = pd.DataFrame()
  
    no_measr_time_idx_n_new = []

    exp_start_timestampUTC = leaves[0].split('/')[0] # is same for all the k leaves!

    for k in range(0, len(leaves), 2): 
        
        tx =  leaves[k].split('/')[1]
        gtx = leaves[k].split('/')[2] if args.treebranch == 'wtx' else 'noTXhappened'
        grx = leaves[k].split('/')[3] if args.treebranch == 'wtx' else leaves[k].split('/')[2]
        rx =  leaves[k].split('/')[4] if args.treebranch == 'wtx' else leaves[k].split('/')[3]
        samp = leaves[k].split('/')[5] if args.treebranch == 'wtx' else leaves[k].split('/')[4]
        time =  leaves[k+1].split('/')[5] if args.treebranch == 'wtx' else leaves[k+1].split('/')[4]

        endpoints.append(rx)

        if args.treebranch == 'wtx':
            ds_samples = allsampsandtime[tx][gtx][grx][rx][samp]
            ds_times = allsampsandtime[tx][gtx][grx][rx][time]
        else:
            print("in WO_TX leaf.. ******************************************** SHOULDNT HAPPEN! ")
            ds_samples = allsampsandtime[tx][grx][rx][samp]
            ds_times = allsampsandtime[tx][grx][rx][time]


        smlist = list( ds_samples[()]  )

        tmlist = list( ds_times[()]    )
        empty_time_idxs = [index for index, value in enumerate(tmlist) if not value]
        no_measr_time_idx_n_new.extend(empty_time_idxs)

        sm = { rx : smlist}
        ti = { f'{rx}_ti' : tmlist}

        df_samples = pd.DataFrame(sm)
        df_time = pd.DataFrame(ti)

        df_allrx = pd.concat([df_allrx, df_samples], axis=1)
        df_allti = pd.concat([df_allti, df_time], axis=1)

    print('Done reading hdf5 tree!\n')
    return tx, np.unique(no_measr_time_idx_n_new), df_allrx, df_allti, exp_start_timestampUTC #, endpoints

##############################################################
############################################################### 


def do_data_storing(ff, attrs, allsampsandtime, leaves):
    """
    B. Two important flags to check before running this script. These flags are hardcoded into the script so can only be changed first reading within the script.
        1.  to_filter_tothemax_or_not:  Flag to select what to store: PSD or IQ 
        
        2. to_plott = 1: Flag to save the PSD/CFO/RMS plots 
            2a)

            2b)RMS dict and plots can be created iff this flag=True. 
            >RMSdict has this format: list of 6 values [nth_rms, nth_speed, nth_timstamp, nth_lat, nth_long, nth_distto_pthBS ] for each nth_obs at each pth_BS.
            >RMSdict is not saved on as pkl files.
            >RMSvalues are plotted on psd plots as a patch.
            >RMS functions are not shared for others.
        3. Meh~~~ to_store_flag  # flag to (re)save the plots and dicts!
            > Running out of disk space and existential patience to resave the plots! so put a flag for now!

    C. This functions takes:

    D. This functions stores:

    D. This function returns:

    """

    ############################################
    runtime = f'{int(time.time())}'  # print("runtime is:", runtime)

    fnm=f"{args.dirdata}".split('meas_')[1]
    ############################################
    
    #### GT reading #############################
    
    loc_files = list(Path(args.dirdata).rglob('*.csv') ) # pick the csv outta cwd
    print('\nLocation gps csv file to read is:', loc_files, "\n")
    
    if len(loc_files) == 0:
        print("No BUS Location csv file present!!!!!!!")
        exit(1)
    else:
        gt_loc_df = pd.read_csv(loc_files[0], header=0)
        gt_loc_df['Time (UTC)'] = pd.to_datetime(gt_loc_df['Time (UTC)']).dt.tz_convert('US/Mountain').dt.strftime("%Y-%m-%d %H:%M:%S")
        
        print('\nOverall length of CSV', gt_loc_df.shape, "Max bus speed in this experiment was:", gt_loc_df['Speed (meters/sec)'].max())
    


    ############################################
    ### For doing modified cfo, and get snr indexes, and/or filtering the GPS CSV gt_loc_df
    ############################################
    
    to_filter_tothemax_or_not = 1 # Flag to select what to store: PSD or IQ 
    
    ############################################
    
    to_plott = 1 # Flag to plot and render/show the PSD/CFO/RMS plots 

    to_store_flag = 1 # flag to (re)save the plots and dicts!

    # if to_store_flag:Cant do this cause plt_dict_function asks for dir always!! unless you wanna plot_dicts after the psd_pkl iterations, along with rmsplots!

    nameofplotsfolder = "formyML_psdplots " if to_filter_tothemax_or_not else "forGIT_IQ_plots"     #"finalplots"  #"overall_plots"
    overall_plots_dir = Path(args.dirdata+'/'+f'{nameofplotsfolder}'+'/'+f"{runtime}")
    overall_plots_dir.mkdir(parents=True, exist_ok=False)
    



    ############################################
    # # #### Not filtering the GPS CSV gt_loc_df anymore, but still doing modified cfo with snr indexes as to_filter_tothemax_or_not is set to 1 above
    # if to_filter_tothemax_or_not: 
    #     ##### to_filter_df_new_format_4in1list = gt_loc_df.assign(latlontuple=gt_loc_df.apply(lambda row: [row['Speed (meters/sec)'], row['Track'],  row['Lat'], row['Lon'] ]   , axis=1)).drop(gt_loc_df.columns.tolist(), axis=1)
    #     to_filter_df_old_fomrat_3in1list = gt_loc_df.assign(latlontuple=gt_loc_df.apply(lambda row: [row['Speed (meters/sec)'], row['Track'],  (row['Lat'], row['Lon']) ], axis=1)).drop(gt_loc_df.columns.tolist(), axis=1)
    #     routewas = get_filtered_df_and_plot(fnm, to_filter_df_old_fomrat_3in1list) # inplace is set to true! so old df to_filter_df_old_fomrat_3in1list is lost, not gt_loc_df!
    #     detour_idxs = gt_loc_df.index.difference(to_filter_df_old_fomrat_3in1list.index)
    #     print("detour indexes found")
    #     if len(detour_idxs) !=0: gt_loc_df.drop(detour_idxs, inplace=True) # drop all the indexes from the ground truth gps df with inplace =true and only when there were some detours returned!
    #     print("detour indexes removed", len(detour_idxs))


    #### ATTRS reading #############################

    tx_CENTER_FREQUENCY  = attrs['txfreq']
    rx_CENTER_FREQUENCY  = attrs['rxfreq'] 
    WAVELENGTH           = scipy.constants.c / tx_CENTER_FREQUENCY 
      
    rate = attrs['rxrate']
    nsamps = attrs['nsamps']
    
    ############################################
    #### calling the function for reading hdf5 into df #############################
    txis, no_measr_time_idx_n_from_hdf5, df_allrx, df_allti, exp_start_timestampUTC = read_leaves_to_DF(leaves, allsampsandtime)
    
    print("This experiment's bus is       == ", txis,)
    print('Name of the Base Stations  are   ==',  df_allrx.columns.values)
    
    n_endpoints_is_subplots = len(df_allrx.columns.values)
    n_total_measurements    = len(df_allrx)

    print('Number of RX Base Stations are   ==', n_endpoints_is_subplots)
    print('Number of total measurements     ==', n_total_measurements, "\n\n") 
        
    ############################################
    ## 1 Hardcoded narrowed spectrum 
    ns = args.frqcutoff # in hz
    print("Hardcoded value:","Narrowed Spectrum freq_span ==", ns)
    

    ## 2 power threshold value
    pwr_threshold = args.pwrthrshld # in dB
    print("Hardcoded value:","threshold ==", pwr_threshold, "\n")
    
    
    ############################################
    cfo_summary_dict = {}
    CFO_REMOVE_FLAG=args.ofstremove

    ######################################################################################################################################
    ######################################################################################################################################
    if CFO_REMOVE_FLAG:
        print("\n\nCFO removal flag set to true. so will remove approax CFO from spectrum!\n\n")
        ## Hardcoded values ##########################    
        
        ## 1 Cutoff frequency for LPF 
        lpf_fc = args.frqcutoff # 1e4 # in hz 
        print("Hardcoded value:","Narrowed LPF freq_span ==", lpf_fc )


        ## 3 degree of polynomial fit value
        degreeforfitting = args.degoffit #3
        print("Hardcoded value:","degree of fit ==", degreeforfitting )
        
        ## 4 max speed value
        MAXIMUM_BUS_SPEED_POSSIBLE = args.maxspeed #21 # in meters_per_second
        print("Hardcoded value:","MAXIMUM_BUS_SPEED_POSSIBLE ==", MAXIMUM_BUS_SPEED_POSSIBLE)
        # 
        FD_MAXPOSSIBLE             = MAXIMUM_BUS_SPEED_POSSIBLE / WAVELENGTH
        print("FD_MAXPOSSIBLE is          == " , FD_MAXPOSSIBLE)
        
        ## 5 narrowed spectrum 
        bufferwindw = args.window    # 1e2 # is a little extra window as a buffer on both sides.  in hz
        
        ## Overall narrow spectrum = buffer + expected max doppler spread value. in hz
        ns = bufferwindw + FD_MAXPOSSIBLE 
        print("Hardcoded value:","Narrowed Spectrum freq_span when CFO removed ==", ns)

        ##lengthy df traversal 1 to get cfo ###################################################################
        
        ### calling the function for cfo calculation #############################
        
        ### NEW! keeping! using db option, almost same number of rows as old! but getting the high_SNR_n_list so keeping this one! 
        if to_filter_tothemax_or_not: 
            cfo_mthd = 'new_db' # new_lin
            cfo_summary_dict, no_measr_time_idx_n_from_cfo, no_gps_mesrnt_idx_n_from_cfo, high_SNR_n_list, n_moving_msrmnts = get_cfo_either_lin_or_db_pwr(fnm, df_allrx, df_allti, gt_loc_df, rate, lpf_fc, exp_start_timestampUTC, pwr_threshold, degreeforfitting, cfo_mthd, overall_plots_dir)
            plot_all_off_dictionaries(ff, fnm, cfo_summary_dict, f'{runtime}', degreeforfitting , cfo_mthd, overall_plots_dir, to_store_flag)

        else:
            ### OLD!
            cfo_mthd = 'old_db'
            cfo_summary_dict, no_measr_time_idx_n_from_cfo, no_gps_mesrnt_idx_n_from_cfo = get_cfo(fnm, df_allrx, df_allti, gt_loc_df, rate, lpf_fc, exp_start_timestampUTC, pwr_threshold, degreeforfitting)
            plot_all_off_dictionaries(ff, fnm, cfo_summary_dict,  f'{runtime}', degreeforfitting , cfo_mthd, overall_plots_dir, to_store_flag)
    
    ######################################################################################################################################
    ######################################################################################################################################


    #### storing all the ATTRS in metadata_dict #############################
    metadata_dict = {
    "tx" : txis,
    "nsamps" : nsamps,
    "rate" : rate, 
    "tx_CENTER_FREQUENCY" : tx_CENTER_FREQUENCY,
    "narrow_spectrum": ns,
    "exp_datetime" : datetime.fromtimestamp(int(exp_start_timestampUTC)).astimezone( pytz.timezone("America/Denver")) #.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    ######################################################################################################################################
    ######################################################################################################################################

    ## data dicts ################################
    data_and_label_dict = {key: [] for key in df_allrx.columns.values}
    data_and_label_dict["speed_postuple"]   = []
    
    nogpsgt=0

    ######################################################################################################################################
    ######################################################################################################################################

    if to_plott:

        colnames_list  = df_allrx.columns.values 
        rmsdict =   {key: [] for key in colnames_list}
        

        frst, last = 1, 2
        fig, ax = plt.subplots(frst, last, figsize=(8, 3), num = "psdVsloc", sharey = False, sharex = False) #sharey = True, sharex = True)
        l=last-1  
        plt.ion()

        #### apply UTM on MT locs
        alllocs_df = gt_loc_df.apply( lambda row: pd.Series(utm.from_latlon( row['Lat'], row['Lon'])[0:2]), axis=1)
        
        #### apply UTM on BS locs
        all_BS_coords_df = pd.DataFrame(all_BS_coords.values(), columns=['Latitude', 'Longitude'], index=all_BS_coords.keys())
        all_BS_coords_df[['Northing', 'Easting']] = all_BS_coords_df.apply(lambda row: pd.Series(latlon_to_utm(row)), axis=1) 

    ######################################################################################################################################
    ######################################################################################################################################

    ########### iterating dataframe to make pickles with (spectrum, label) 
    for n in range(0, n_total_measurements ):

        if n in no_measr_time_idx_n_from_hdf5 or n in no_gps_mesrnt_idx_n_from_cfo:
            print(f"{n}th measurment skipped cause its empty\n") # as this pth /all p msrmnts is missing! Neither should store or do labelling for those who are present")
            continue # but not exiting(breaking) the loop. Such that, will go to the next n
        
        for p in range(0, n_endpoints_is_subplots):   

            this_measr_time = df_allti.iloc[n,p]  

            ### LABELS! #################          
            ## Time - will be pickling!
            this_measr_timeuptoseconds = this_measr_time[0:-7] # NOTE: ground truth is per second but has no resolution for 'subseconds'
            
            if not isinstance(this_measr_timeuptoseconds, str):
                this_measr_timeuptoseconds = this_measr_timeuptoseconds.decode('utf-8')      

            ## Get respective row from GT_LOC_CSV            
            matched_row_ingt = gt_loc_df[gt_loc_df['Time (UTC)'].str.contains(this_measr_timeuptoseconds)]           
            
            # ##########################
            # if len(matched_row_ingt) == 0:
            #     nogpsgt+=1
            #     print('Somehoww bus didnt record {n}th GPS measurment ground truth for this row', n, "so not storing!\n")
            #     break # shouldnt go to the next p so no continue!
            #     ##########################
            
            # Location - will be pickling!
            current_bus_pos_tuple = (matched_row_ingt['Lat'].values[0],matched_row_ingt['Lon'].values[0])

            # Speed  - will be pickling!
            this_speed = matched_row_ingt['Speed (meters/sec)']
            
            # Direction  - will be pickling!
            direction_w_TN = matched_row_ingt['Track'].values[0] if any(colname == 'Track' for colname in matched_row_ingt.columns.values) else 0
            ####################################

            ## IQ DATA! ##################
            this_measurement  = df_allrx.iloc[n,p]
            
            if CFO_REMOVE_FLAG:
                ##remove cfo            
                this_measurement, cfo_approx, cfomthd_forthismsrmnt = do_cfo_removal(cfo_summary_dict, degreeforfitting, this_measr_timeuptoseconds, df_allrx.columns[p], this_measurement, rate)

            ####################################
            
            if to_filter_tothemax_or_not: # with new cfo method 
                # get psd, not spectrum, only for my ML

                if n in high_SNR_n_list:
                    # taking advantage of new getcfo function to filter out high SNR altogether
                    break
                
                #####full psd
                # fig_plt = plt.figure("fig_pltfunction")
                # [pxxc_linear,   freq_full]   = plt.psd(this_measurement, NFFT= nsamps, Fs=rate)          ## pxxc_DB, pxxc_linear, fxxc = psdcalc(corrected_signal, rate)
                # plt.close("fig_pltfunction")
                # print(pxxc_linear[0], pxxc_linear[-1], freq_full[0], freq_full[-1])

                
                pxxc_linear, freq_full = psd(this_measurement, NFFT=nsamps, Fs=rate)
                # print(10.0 *np.log10(np.nan_to_num(np.square(np.abs(pxxc_linear))))[[0,-1]]) 


                full_spectrum, freq_full  = get_full_DS_spectrum(this_measurement, rate)

                # print(10.0 *np.log10(np.nan_to_num(np.square(np.abs(full_spectrum))))[[0,-1]])                
                # psdln          = np.nan_to_num(np.square(np.abs(full_spectrum)))
                # psdDB          = 10.0 *np.log10(psdln)                


                # narrowed frequency indexes 
                fdidx  = (freq_full >= -ns) & ( freq_full <= ns) 
                # narrowed psd
                data_and_label_dict[df_allrx.columns[p]].append(pxxc_linear[fdidx])  # pickling!


            else: # old cfo method for uploading on git
                ##### full spectrum 
                [full_spectrum, freq_full]   = get_full_DS_spectrum(this_measurement, rate)
                # print(full_spectrum[0], full_spectrum[-1])
                
                ##### narrowed frequency indexes 
                fdidx  = (freq_full >= -ns) & ( freq_full <= ns)             
                # narrowed spectrum
                data_and_label_dict[df_allrx.columns[p]].append(full_spectrum[fdidx]) # pickling!

            ####################################

            ## labelling here! Labels are to be stored as the last column
            if p==n_endpoints_is_subplots-1: # when at the last Rx's index, that is only after the last BS has been picked, we store loc, speed, track, and time in their respective columns
                data_and_label_dict["speed_postuple"].append([this_speed.values[0], direction_w_TN, current_bus_pos_tuple, this_measr_timeuptoseconds])
                        
            #####################################
            
            if to_plott and p==4:# plotting, saving only for ustar

                f_ns              = freq_full[fdidx]      
                psdln_ns          = np.nan_to_num(np.square(np.abs(full_spectrum[fdidx])))
                psdDB_ns          = np.nan_to_num(10.0 *np.log10(np.nan_to_num(np.square(np.abs(full_spectrum[fdidx])))))  
                aa_psd_max        = psdDB_ns[np.argmax(psdDB_ns)]

                ax[0].clear()
                ax[0].plot(f_ns, psdDB_ns, '-o',color = 'r' if this_speed.values[0]==0 else 'g', label = f"{n}th FP at {df_allrx.columns[p].split('-')[1]}")
                
                if aa_psd_max > pwr_threshold:
                    rmsis = rmsfunction(psdDB_ns, psdln_ns, f_ns, pwr_threshold)
                    
                    rmsdict[df_allrx.columns[p]].append([rmsis, this_speed.values[0], this_measr_timeuptoseconds, matched_row_ingt.iloc[0][3:5][0], matched_row_ingt.iloc[0][3:5][1] , calcDistLatLong( all_BS_coords[df_allrx.columns[p].split('-')[1]], matched_row_ingt.iloc[0][3:5] )])
                    
                    
                    rms_at = AnchoredText(f"RMS:{round(rmsis,2)} Hz\nSpeed:{round(this_speed.values[0],1)} mps", prop=dict(size=8), frameon=True,pad=0.1, loc='upper left')  #\nMax. noise:{round(pwr_threshold,2)}
                    rms_at.patch.set_boxstyle("round, pad=0.,rounding_size=0.2")
                   
                    ax[0].add_artist(rms_at)

                ax[0].axhline(y = pwr_threshold, lw=1, ls='--', c='k', label= f"Noise Threshold: {pwr_threshold}") #\nEst. CFO: {round(cfo_approx,2)} Hz
                ax[0].legend(loc="lower left")
                ax[0].set_ylim(-80,20)
                
                ax[0].set_xlim(f_ns[0],f_ns[-1])
                ax[0].set_xlabel("Freq (Hz)")  
                ax[0].set_ylabel("Power (dB)")
                


                ax[l].clear()
                
                # if routewas == 'orange': #ff in #[6,7,18,19,20]:
                if ff in [6,7,18,19]:
                    routeclr = '#ffb16d'
                    routename = 'O' #B
                else:
                    routeclr = '#33b864' #'#2a7e19' #
                    routename = 'G' #A

                alllocs_df.iloc[:2000].plot.scatter(x=0, y=1, c=routeclr, marker='.', s=2, ax=ax[l], label='route ')#+f"{routename}") ########[ax[l].scatter( utm.from_latlon(all_BS_coords[bs.split('-')[1]][0], all_BS_coords[bs.split('-')[1]][1])[0:2][0],utm.from_latlon(all_BS_coords[bs.split('-')[1]][0], all_BS_coords[bs.split('-')[1]][1])[0:2][1], marker='1', s=100, label= bs.split('-')[1][:3] ) for bs in df_allrx.columns.values] 
                [ax[l].scatter( utm.from_latlon(all_BS_coords[bs.split('-')[1]][0], all_BS_coords[bs.split('-')[1]][1])[0:2][0], utm.from_latlon(all_BS_coords[bs.split('-')[1]][0], all_BS_coords[bs.split('-')[1]][1])[0:2][1], marker='1', c='C{}'.format(uu+103), s=100, label= bs.split('-')[1][:3] ) for uu, bs in enumerate(df_allrx.columns.values)]  #FFB6C1, #ffb16d
                
                lat_y_40s, long_x_11s =  matched_row_ingt.iloc[0][3], matched_row_ingt.iloc[0][4] # ax[1].scatter( long_x_11s, lat_y_40s, c='k', marker='o', s=10)
                easting_lngs_x, northing_lats_y = utm.from_latlon(lat_y_40s, long_x_11s)[0:2] # print("easting_lngs_x, northing_lats_y", easting_lngs_x, northing_lats_y, "\n")
                         
                ax[l].scatter(easting_lngs_x, northing_lats_y, c='k', marker='*', s=12, label="loc" ) #, label= f"|v|:{round(this_speed.values[0],1)} mps"  # [ax[l].scatter( utm.from_latlon(all_BS_coords[bs.split('-')[1]][0], all_BS_coords[bs.split('-')[1]][1])[0:2][0], utm.from_latlon(all_BS_coords[bs.split('-')[1]][0], all_BS_coords[bs.split('-')[1]][1])[0:2][1] , marker='1', s=100, label= bs.split('-')[1] )    for bs in df_allrx.columns.values]


                [ea.grid(True) for ea in ax.flatten()]

                xl = all_BS_coords_df.loc['bes'][3]-100
                xh = all_BS_coords_df.loc['smt'][3]+600

                yl = all_BS_coords_df.loc['bes'][2]-500
                yh = all_BS_coords_df.loc['hospital'][2]+500


                ax[l].set_xlim(xl, xh)
                ax[l].set_ylim(yl, yh)

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
                
                # plt.draw()
                # plt.pause(0.01) 
                
                # if to_store_flag:
                    # plt.figure("psdVsloc").savefig(f"{overall_plots_dir}" +"/"+f"{runtime}_{n}_{fnm}_"+"psdVsloc.svg",format='svg', dpi=1200)  #.pdf",format='pdf')
                         

        print("Rows traversed: %i/%i" %(n+1, n_total_measurements), end='\r')



    if to_plott:
        plt.ioff()
        plt.close("psdVsloc")
        
        if to_store_flag:
            my_plot_rms_dicts(fnm,rmsdict,overall_plots_dir, runtime)
            my_3d_plot_rms_dicts(fnm,rmsdict,overall_plots_dir, runtime)
    


    # print(f'\nSize of a single spectrum : {len(freq_full[fdidx])}')

    final_totallength = [len(val) for k, val in enumerate(data_and_label_dict.values()) ]
    print("How many rows we got in this pickled data file", final_totallength)
    print("Count of rows skipped",  len(no_measr_time_idx_n_from_hdf5) + len(no_gps_mesrnt_idx_n_from_cfo) if not to_filter_tothemax_or_not else  len(no_measr_time_idx_n_from_hdf5) + len(no_gps_mesrnt_idx_n_from_cfo) + len(high_SNR_n_list) )
    
    same_numb_rows = all(element == final_totallength[0] for element in final_totallength)   
    if same_numb_rows:
        print("Storing only iff each BS/columns got same number of rows.")
        nofrows = np.unique(final_totallength)
        
        #### Store one pickle file for one hdf5 data file in the common directory
        fn = f"{args.dircommon}"+"/"+ f"{args.dirdata}".split('/')[-1]+'.pickle' #+f'runtime_{runtime}'
        if to_store_flag:
            pkl.dump((data_and_label_dict, metadata_dict, cfo_summary_dict), open(fn, 'wb' ) )
            print("\n\nPickled!", end="")
        
        print("\n\n\n\n-----------------------------------------------------------\n\n\n\n")
        return metadata_dict, cfo_summary_dict, data_and_label_dict, nofrows

    else:
        print("\n\nColumns got different number of rows.")
        exit()  
    ############################################################################################################
    ############################################################################################################
    """
    5 fixables!

    Shout_meas_02-03-2023_12-55-47 bes!
    
    
    Shout_meas_02-14-2023_10-45-17 ustar!=fixed   (LO leak) 
    Shout_meas_02-14-2023_12-48-02 utsar!=fixed bes! not fixed: two mirror peaks around 0, 1khz of frequncy.. looks like lol), 
    Shout_meas_02-14-2023_14-49-21 smt! big noise/snr/ inteference)<<
    
    Shout_meas_02-16-2023_16-59-03 bes! (big noise/snr/ inteference, numerous peaks)
                

    # if val_psd_max > pwr_threshold and val_freq_max < 8000 and val_freq_max > 5600:  # D7:  02-03-2023_12-55-47
    # if val_psd_max > pwr_threshold and val_freq_max < 10000 and val_freq_max > 6200: # D13: 02-14-2023_10-45-17
    # if val_psd_max > pwr_threshold and val_freq_max < 10000 and val_freq_max > 6100:  # D14: 02-14-2023_12-48-02
    # if val_psd_max > pwr_threshold and val_freq_max < 10000 and val_freq_max > 8300:  # D15: 02-14-2023_14-49-21
    if val_psd_max > pwr_threshold and val_freq_max < 10000 and val_freq_max > 5000:  # D21: 02-16-2023_16-59-03



    2 unfixables! Tried the linear methods.. detail in snr_acv_awgn.py get_cfo function
    Shout_meas_02-09-2023_15-08-10, Shout_meas_02-09-2023_17-12-28
    Shout_meas_01-30-2023_15-40-55 (multiple, first picked which was smaller)

    The CFO, that is the offest between the mobile transmitter's carrier frequency and the frequency of a receiver base-station,
    shows a trend in its values over the course of an experiement.
    We show this trend using our longest data $D6$ collected in one experiment in figure \ref{}, with the 


    ##################  All possible mep offsets ##################
    ###############################################################

    mep_ppm = {
    '4407': [8500,9500],  #feb16
    '4603': [6000,7000],  #feb3
    '4734': [8500,10000], #feb16, feb14
    '6181': [7500,8500],  #jan30
    # '6183': [6500,7500],  #jan30(3)
    # '6183': [1900,5000], #feb9(3), feb6(1), 
    }

    mep_ppm = {'6180': [7000,8000],
    '6181': [7500,8500],
    '6182': [7000,8000],
    '6185': [7000,8000],
    '6183': [1900,5000],
    '4555': [3000,4000],
    '4603': [6000,7000],
    '4604': [6000,7000],
    '4734': [8500,9500],
    '4407': [8500,9500]
    }
    #'6183': [6500,7500]:
    

    """

  
###############################################################
############################################################### 

def quickplot_labels_non_utmed(i, fn, df): # filenname is passed
   
    labels  = df["speed_postuple"]    
    bsnames = df.columns.values  
    
    fig_ori, ax_ori = plt.subplots(figsize=(8, 8))
    
    for ii in range(len(labels)):
        lat_y, long_x =  labels.iloc[ii][2][0], labels.iloc[ii][2][1]
        ax_ori.scatter( long_x, lat_y, c='C{}'.format(i), marker='o', s=2)
    ax_ori.scatter( labels.iloc[0][2][1], labels.iloc[0][2][0] , marker='*', s=100, label = 'first loc', c = 'y')

    [ax_ori.scatter(all_BS_coords[bs.split('-')[1]][1], all_BS_coords[bs.split('-')[1]][0], marker='1',
                    s=100, label= bs.split('-')[1] ) for b, bs in enumerate(bsnames) if b<5 ]    
    
    ax_ori.legend()
    ax_ori.grid(True)
    plt.title(f"Tracjectory of Exp: {fn} \n Number of collected measurements: {len(labels)}") 
    plt.show()  


def get_dataset_keys(f):
    keys = []
    f.visit(lambda key : keys.append(key) if isinstance(f[key], h5py.Dataset) else None)
    return keys

def parse_args_definition():

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--dircommon",  type=str, default=DEF_DIRDATA,   help="The directory in which resides one or more folders that each have one data file. Default: %s" % DEF_DIRDATA)
    parser.add_argument("-d", "--dirdata",    type=str, default=DEF_DIRDATA,   help="The name of the per experiment directory. Default: %s" % DEF_DIRDATA)
    parser.add_argument("-f", "--hdf5name",   type=str, default=DEF_HDF5NAME,  help="The name of the HDF5 format data file. Default: %s" % DEF_HDF5NAME)
    parser.add_argument("-t", "--treebranch", type=str, default=DEF_HDF5BRANCH,help="Must specify which branch (wtx or wotx) on the hdf5 tree to traverse. Default: %s" % DEF_HDF5BRANCH)

    parser.add_argument("-l", "--frqcutoff",  type=int, default=FRQ_CUTOFF,    help="The name of the per experiment directory. Default: %s" % FRQ_CUTOFF)
    parser.add_argument("-o", "--ofstremove", type=int, default=CFO_REMOVE_FLAG,    help="Must specify which branch (wtx or wotx) on the hdf5 tree to traverse. Default: %s" % CFO_REMOVE_FLAG)
    
    parser.add_argument("-p", "--pwrthrshld", type=int, default=PWR_THRESHOLD,    help="The name of the HDF5 format data file. Default: %s" % PWR_THRESHOLD)
    parser.add_argument("-s", "--maxspeed",   type=int, default=MAXIMUM_BUS_SPEED,help="Must specify which branch (wtx or wotx) on the hdf5 tree to traverse. Default: %s" % MAXIMUM_BUS_SPEED)
    parser.add_argument("-e", "--degoffit",   type=int, default=DEGREE_OF_POLYFIT,help="The name of the HDF5 format data file. Default: %s" % DEGREE_OF_POLYFIT)
    parser.add_argument("-w", "--window",     type=int, default=BUFFER_WINDOW,    help="Must specify which branch (wtx or wotx) on the hdf5 tree to traverse. Default: %s" % BUFFER_WINDOW)

    
    return parser #not parsing them here, so not doing parser.parse_args()!

if __name__ == "__main__":


    
    """
    A. This script is run by this command:
    python ~/Documents/DopplerSpreadLocalization/data_to_cforemoved_narrowed_pickles.py -l 10000 -o 1

    """
    

    print("Start time = ", datetime.now().strftime("%H:%M:%S"))

    print('Your current working directory that has all HDF5 data files is:', Path.cwd(), "\n")

    parser_itself = parse_args_definition()
    args_old      = parser_itself.parse_args() # parsing them here!
    
    totalrows           = 0
    NumberofDatafolders = sum( 1 for f in sorted(Path(args_old.dircommon).iterdir()) if f.is_dir() and str(f).startswith('Shout_') )
    print("\nTotal number of data folders to read:", NumberofDatafolders, "\n\n")

    ff=0

    for uu, f in enumerate(sorted(Path(args_old.dircommon).iterdir())):
        
        if f.is_dir() and str(f).startswith('Shout_'):
            ff+=1
            args = parser_itself.parse_args(['--dirdata', str(f) ]) #update the argument here!            
            
            # args = parser_itself.parse_args(['--frqcutoff', 'args_old.frqcutoff]) #update the argument here!            
            
            args.frqcutoff  = args_old.frqcutoff
            args.ofstremove = args_old.ofstremove            
            args.pwrthrshld = args_old.pwrthrshld
            args.maxspeed   = args_old.maxspeed            
            args.degoffit   = args_old.degoffit
            args.window     = args_old.window # print(args)
            
            ### print("uu is", uu, "\nff is", ff)
            print("ff is", ff)
            
            # if ff!=16:
            #     continue

            print(f"\n\nProcessing the data in {ff}th {args.dirdata} directory\n")           

            dsfile = h5py.File("%s/%s" % (args.dirdata, args.hdf5name), "r")
            dsfile_with_meas_root = dsfile[MEAS_ROOT]
            all_leaves_2 = get_dataset_keys(dsfile_with_meas_root)
            
            num_leaves = len(all_leaves_2)
            timestamp_leaf = all_leaves_2[0].split('/')[0]
            if all_leaves_2[0].split('/')[1] == 'wo_tx':
                print("\nOnly wo_TX data collection happened. There is no branch for w_tx\n You should confirm what you are looking for!!")
                leaves_withoutTX = all_leaves_2
            else:
                leaves_withTX = all_leaves_2[0:int(num_leaves/2)]
                leaves_withoutTX = all_leaves_2[int(num_leaves/2):]

            commonANDdeepest_root_foralltxrx  = dsfile_with_meas_root[timestamp_leaf]
            needed_attrs_obj = commonANDdeepest_root_foralltxrx.attrs

            metadata_dict, cfo_summary_dict, datalabel_dict, nofrws = do_data_storing(ff, needed_attrs_obj, commonANDdeepest_root_foralltxrx, leaves_withTX if args.treebranch =='wtx' else leaves_withoutTX) #all_leaves_2) # leaves_withTX)
            
            totalrows += nofrws 
            # break   

    print(f"\n\nAll {NumberofDatafolders} data folders in the common data directory {args.dircommon} done! Total valid rows collected = {totalrows}\n\n")
    
    print("Time now  = ", datetime.now().strftime("%H:%M:%S"))

    # exec(open("/Users/aartisingh/Documents/DopplerSpreadLocalization/pickes_to_psdplots.py").read()) 
    # exec(open("/Users/aartisingh/Documents/DopplerSpreadLocalization/pickes_to_IQ_plots.py").read()) 


