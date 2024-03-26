from libs import * 
from functions_cforemoval import *

from mine_psd_fetching import *


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
    '''
    '''
    print('Reading hdf5 tree......',end='')

    endpoints=[]
    df_allrx= pd.DataFrame()
    df_allti = pd.DataFrame()
  
    no_measr_time_idx_n2 = []

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
        no_measr_time_idx_n2.extend(empty_time_idxs)

        sm = { rx : smlist}
        ti = { f'{rx}_ti' : tmlist}

        df_samples = pd.DataFrame(sm)
        df_time = pd.DataFrame(ti)

        df_allrx = pd.concat([df_allrx, df_samples], axis=1)
        df_allti = pd.concat([df_allti, df_time], axis=1)

    print('Done reading hdf5 tree!\n')
    return tx, np.unique(no_measr_time_idx_n2), df_allrx, df_allti, exp_start_timestampUTC #, endpoints

##############################################################
############################################################### 


def do_data_storing(ff, attrs, allsampsandtime, leaves):

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
    

    #### ATTRS reading #############################

    tx_CENTER_FREQUENCY  = attrs['txfreq']
    rx_CENTER_FREQUENCY  = attrs['rxfreq'] 
    WAVELENGTH           = scipy.constants.c / tx_CENTER_FREQUENCY 
      
    rate = attrs['rxrate']
    nsamps = attrs['nsamps']

    #### calling the function for reading hdf5 into df #############################
    txis, no_measr_time_idx_n, df_allrx, df_allti, exp_start_timestampUTC = read_leaves_to_DF(leaves, allsampsandtime)
    
    print("This experiment's bus is       == ", txis,)
    print('Name of the Base Stations  are   ==',  df_allrx.columns.values)
    
    n_endpoints_is_subplots = len(df_allrx.columns.values)
    n_total_measurements    = len(df_allrx)

    print('Number of RX Base Stations are   ==', n_endpoints_is_subplots)
    print('Number of total measurements     ==', n_total_measurements, "\n\n") 
    

    ## Hardcoded narrowed spectrum 
    ns = args.frqcutoff # in hz
    print("Hardcoded value:","Narrowed Spectrum freq_span ==", ns)
    cfo_summary_dict = {}


    ## 2 power threshold value
    pwr_threshold = args.pwrthrshld # -19# in dB
    print("Hardcoded value:","threshold ==", pwr_threshold, "\n")
    
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
        
        FD_MAXPOSSIBLE             = MAXIMUM_BUS_SPEED_POSSIBLE / WAVELENGTH
        print("FD_MAXPOSSIBLE is          == " , FD_MAXPOSSIBLE)
        
        ## 5 narrowed spectrum 
        bufferwindw = args.window    # 1e2 # is a little extra window as a buffer on both sides.  in hz
        
        ## Overall narrow spectrum = buffer + expected max doppler spread value. in hz
        ns = bufferwindw + FD_MAXPOSSIBLE 
        print("Hardcoded value:","Narrowed Spectrum freq_span when CFO removed ==", ns)

        ##lengthy df traversal 1 to get cfo ###################################################################
        
        ### calling the function for cfo calculation #############################
        cfo_summary_dict, no_measr_time_idx_n2, no_gps_mesrnt_idx_n2 = get_cfo(df_allrx, df_allti, gt_loc_df, rate, lpf_fc, exp_start_timestampUTC, degreeforfitting, pwr_threshold)

        # plot_all_off_dictionaries(cfo_summary_dict)
    
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
    fnm = f"{args.dirdata}".split('meas_')[1] 
    plott = 1

    if plott:
        
        alllocs_df = gt_loc_df.apply( lambda row: pd.Series(utm.from_latlon( row['Lat'], row['Lon'])[0:2]), axis=1)
        
        colnames_list  = df_allrx.columns.values 
        rmsdict =   {key: [] for key in colnames_list}
        
        """        # fig, ax = plt.subplots(2,2, figsize=(16, 7))
                # [ax[0][1].scatter( all_BS_coords[bs.split('-')[1]][1], all_BS_coords[bs.split('-')[1]][0], marker='1', s=100, label= bs.split('-')[1] ) for bs in df_allrx.columns.values ]    
        """
        
        # frst=1
        # last=len(colnames_list)+1
        # fig, ax = plt.subplots(frst, last, figsize=(14, 3), num = "psdVsloc", sharey = True, sharex = True)
        # l=last-1

        runtime = f'{int(time.time())}'
        print("runtime is:", runtime)
        overall_plots_dir = Path(args.dirdata+'/overall_plots/'+f"{runtime}")
        overall_plots_dir.mkdir(parents=True, exist_ok=True)
        
        frst=1
        last=2
        # fig, ax = plt.subplots(frst, last, figsize=(8, 3), num = "psdVsloc", sharey = True, sharex = True)
        fig, ax = plt.subplots(frst, last, figsize=(8, 3), num = "psdVsloc", sharey = False, sharex = False)
        l=last-1  



        all_BS_coords_df = pd.DataFrame(all_BS_coords.values(), columns=['Latitude', 'Longitude'], index=all_BS_coords.keys())
        all_BS_coords_df[['Northing', 'Easting']] = all_BS_coords_df.apply(lambda row: pd.Series(latlon_to_utm(row)), axis=1) 


    
        """        # # shax = ax[l].get_shared_x_axes()
                # # shay = ax[l].get_shared_y_axes()
                
                # # shax.remove(ax[l])
                # # shay.remove(ax[l])


                # # ax[l].xaxis.major = matplotlib.axis.Ticker()
                # # xloc = matplotlib.ticker.AutoLocator()
                # # xfmt = matplotlib.ticker.ScalarFormatter()
                # # ax[l].xaxis.set_major_locator(xloc)
                # # ax[l].xaxis.set_major_formatter(xfmt)

                # # ax[l].yaxis.major = matplotlib.axis.Ticker()
                # # yloc = matplotlib.ticker.AutoLocator()
                # # yfmt = matplotlib.ticker.ScalarFormatter()
                # # ax[l].yaxis.set_major_locator(yloc)
                # # ax[l].yaxis.set_major_formatter(yfmt)
                
                # # [ax[l].scatter( utm.from_latlon(all_BS_coords[bs.split('-')[1]][0], all_BS_coords[bs.split('-')[1]][1])[0:2][0],utm.from_latlon(all_BS_coords[bs.split('-')[1]][0], all_BS_coords[bs.split('-')[1]][1])[0:2][1], marker='1', s=100, label= bs.split('-')[1][:3] ) for bs in df_allrx.columns.values] 
                # # alllocs_df.iloc[:2500].plot.scatter(x=0, y=1, color='b', marker='.', s=2, ax=ax[l])
                
                # # [ax[l].scatter( px[0].values[0],  px[1].values[0], marker='.', s=2, c='b') for px in alllocs_df.iloc[:25]] 
                        
                # # # xxs=[]
                # # # yys=[]
                # # # for bs in df_allrx.columns.values:
                # # #     xx = utm.from_latlon(all_BS_coords[bs.split('-')[1]][0], all_BS_coords[bs.split('-')[1]][1])[0:2][0]
                # # #     yy = utm.from_latlon(all_BS_coords[bs.split('-')[1]][0], all_BS_coords[bs.split('-')[1]][1])[0:2][1]
                # # #     xxs.append(xx)
                # # #     yys.append(yy)
                # # #     ax[l].scatter( xx,yy, marker='1', s=100, label= bs.split('-')[1][:3] )        

                
                # # # xxs=[]
                # # # yys=[]
                # # # for bs in df_allrx.columns.values:
                # # #     xx = utm.from_latlon(all_BS_coords[bs.split('-')[1]][0], all_BS_coords[bs.split('-')[1]][1])[0:2][0]
                # # #     yy = utm.from_latlon(all_BS_coords[bs.split('-')[1]][0], all_BS_coords[bs.split('-')[1]][1])[0:2][1]
                # # #     xxs.append(xx)
                # # #     yys.append(yy)
                # # #     ax[l].scatter( xx,yy, marker='1', s=100, label= bs.split('-')[1][:3] )  
                        
                # # [ax[l].scatter( utm.from_latlon(all_BS_coords[bs.split('-')[1]][0], all_BS_coords[bs.split('-')[1]][1])[0:2][0],utm.from_latlon(all_BS_coords[bs.split('-')[1]][0], all_BS_coords[bs.split('-')[1]][1])[0:2][1], marker='1', s=100, label= bs.split('-')[1][:3] ) for bs in df_allrx.columns.values] 
                # # # [ax[1].scatter( all_BS_coords[bs.split('-')[1]][1], all_BS_coords[bs.split('-')[1]][0] , marker='1', s=100, label= bs.split('-')[1] )    for bs in df_allrx.columns.values]

                # # [ax[l].scatter( px[0], px[1], marker='.', s=2, c='b') for px in alllocs[:2000]] 
        """

         
    
    ######################################################################################################################################
    ######################################################################################################################################
    
    ########### iterating dataframe to make pickles with (spectrum, label) 
    for n in range(0, n_total_measurements ):

        if n in no_measr_time_idx_n:
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
            
            ##########################
            if len(matched_row_ingt) == 0:
                nogpsgt+=1
                print(f'Somehow bus didnt record {n}th GPS measurment ground truth for this row', n, "so not storing!\n")
                break # shouldnt go to the next p so no continue!
                ##########################
            
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
            # full spectrum 
            [full_spectrum, freq_full]   = get_full_DS_spectrum(this_measurement, rate)
            
            # narrowed spectrum indexes 
            fdidx  = (freq_full >= -ns) & ( freq_full <= ns) 

            # narrowed spectrum - will be pickling!
            data_and_label_dict[df_allrx.columns[p]].append(full_spectrum[fdidx])


            ####################################
            # fig_plt = plt.figure("fig_pltfunction")
            # [pxxc_linear, freq_full] = plt.psd(this_measurement, NFFT= nsamps, Fs = rate)          ## pxxc_DB, pxxc_linear, fxxc = psdcalc(corrected_signal, rate)
            # plt.close("fig_pltfunction")  

            # ## narrowed spectrum indexes 
            # fdidx  = (freq_full >= -ns) & ( freq_full <= ns) 

            # data_and_label_dict[df_allrx.columns[p]].append(pxxc_linear[fdidx])


            ####################################

            ## labelling here! Labels are to be stored as the last column
            if p==n_endpoints_is_subplots-1: # when at the last Rx's index, that is only after the last BS has been picked, we store loc, speed, track, and time in their respective columns
                data_and_label_dict["speed_postuple"].append([this_speed.values[0], direction_w_TN, current_bus_pos_tuple, this_measr_timeuptoseconds])
                        
            #####################################
            
            if plott and p==4:
                f=0 #p

                f_ns              = freq_full[fdidx]      
                psdln_ns          = np.nan_to_num(np.square(np.abs(full_spectrum[fdidx])))
                psdDB_ns          = np.nan_to_num(10.0 *np.log10(psdln_ns))  
                aa_psd_max        = psdDB_ns[np.argmax(psdDB_ns)]

                ax[f].clear()
                ax[f].plot(f_ns, psdDB_ns, '-o',color = 'r' if this_speed.values[0]==0 else 'g', label = f"{n}th FP at {df_allrx.columns[p].split('-')[1]}")
                
                if aa_psd_max > pwr_threshold:
                    rmsis = rmsfunction(psdDB_ns, psdln_ns, f_ns, pwr_threshold)
                    
                    rmsdict[df_allrx.columns[p]].append([rmsis, this_speed.values[0], this_measr_timeuptoseconds, matched_row_ingt.iloc[0][3:5][0], matched_row_ingt.iloc[0][3:5][1] , calcDistLatLong( all_BS_coords[df_allrx.columns[p].split('-')[1]], matched_row_ingt.iloc[0][3:5] )])
                    
                    rms_at = AnchoredText(f"RMS:{round(rmsis,2)} Hz\nSpeed:{round(this_speed.values[0],1)} mps", prop=dict(size=8), frameon=True,pad=0.1, loc='upper left')  #\nMax. noise:{round(pwr_threshold,2)}
                    rms_at.patch.set_boxstyle("round, pad=0.,rounding_size=0.2")
                   
                    ax[f].add_artist(rms_at)
        


                ax[f].axhline(y = pwr_threshold, lw=1, ls='--', c='k', label= f"Noise Threshold: {pwr_threshold}") #\nEst. CFO: {round(cfo_approx,2)} Hz
                ax[f].legend(loc="lower left")
                ax[f].set_ylim(-80,20)
                
                ax[f].set_xlim(f_ns[0],f_ns[-1])
                ax[f].set_xlabel("Freq (Hz)")  
                ax[f].set_ylabel("Power (dB)")
                

                """                
                # lat_y_40s, long_x_11s =  matched_row_ingt.iloc[0][3], matched_row_ingt.iloc[0][4]
                # # ax[1].scatter( long_x_11s, lat_y_40s, c='k', marker='o', s=10)
                # # pdb.set_trace()
                # easting_lngs_x, northing_lats_y = utm.from_latlon(lat_y_40s, long_x_11s)[0:2]
                # print( easting_lngs_x, northing_lats_y)
                # ax[5].scatter(easting_lngs_x, northing_lats_y, c='k', marker='o', s=10, label= f"Curent Speed: {round(this_speed.values[0],1)} mps")

                # ax[5].legend(loc='upper left')

                # ax[1][0].clear()
                # [full_spectrum, freq_full]   = get_full_DS_spectrum(df_allrx.iloc[n,p], rate)
                
                # if CFO_REMOVE_FLAG:
                #     ax[1][0].plot(freq_full[(freq_full >= -lpf_fc) & ( freq_full <= lpf_fc) ], np.nan_to_num(10.0 *np.log10((np.square(np.abs(full_spectrum[(freq_full >= -lpf_fc) & ( freq_full <=lpf_fc ) ]))))),   color = 'r' if this_speed.values[0]==0 else 'g', label="full spec after cfo removed")         
                # else:
                #     ax[1][0].plot(freq_full[(freq_full >= -ns) & ( freq_full <=ns ) ], np.nan_to_num(10.0 *np.log10((np.square(np.abs(full_spectrum[(freq_full >= -ns) & ( freq_full <=ns) ]))))),  color = 'r' if this_speed.values[0]==0 else 'g', label="full spec after cfo removed")

                #     ax[1][0].legend(loc='upper left')
                # ax[1][0].set_ylim(-80,20)
                # ax[1][0].set_xlabel("Freq (hz)")
                # ax[1][0].set_ylabel("Power (dB)")


                # ax[1][1].clear() # [ax[1][1].scatter(convert_strptime_to_currUTCtsdelta(v[1], cfo_summary_dict['exp_start_timestampUTC']), v[0], color='y')  for i, v in enumerate( cfo_summary_dict['allcfotime'][df_allrx.columns[p]] ) ]    
                
                # if CFO_REMOVE_FLAG: 
                #     tts = []
                #     ffs = []     

                #     if len(cfo_summary_dict['allcfotime'][df_allrx.columns[p]])!=0:
                #         for eachfrq, eachtime in cfo_summary_dict['allcfotime'][df_allrx.columns[p]]:
                #             delta_t_since_start = convert_strptime_to_currUTCtsdelta(eachtime, cfo_summary_dict['exp_start_timestampUTC'] )
                #             tts.append(   delta_t_since_start  )
                #             ffs.append(eachfrq)
                        
                #         ax[1][1].plot(tts, ffs, '-o', c='#FFB6C1', label=f'all {len(ffs)} offsets')
                        
                #         xfit = np.linspace(min(tts), max(tts), 100)
                        
                #         polynomia = np.poly1d(fit_freq_on_time(tts, ffs, degreeforfitting))                    
                #         ax[1][1].plot(xfit, polynomia(xfit), c='b', label=f' {degreeforfitting}th deg polynomial')  
                #         ax[1][1].scatter(convert_strptime_to_currUTCtsdelta(this_measr_timeuptoseconds, cfo_summary_dict['exp_start_timestampUTC']) , cfo_approx ,  marker='*', s=50 ,  color='r' if this_speed.values[0]==0 else 'g', label = f"CFO: {round(cfo_approx,2)}at {df_allrx.columns[p].split('-')[1]}")             
                #         ax[1][1].legend(loc='upper right')
                #         ax[1][1].set_ylim(6600, 6999)
                #         ax[1][1].set_xlabel("Time spent (s)")
                #         ax[1][1].set_ylabel("Freq offset (Hz)")
                """

                ax[l].clear()

                # shax = ax[l].get_shared_x_axes()
                # shay = ax[l].get_shared_y_axes()
                
                # shax.remove(ax[l])
                # shay.remove(ax[l])


                # ax[l].xaxis.major = matplotlib.axis.Ticker()
                # xloc = matplotlib.ticker.AutoLocator()
                # xfmt = matplotlib.ticker.ScalarFormatter()
                # ax[l].xaxis.set_major_locator(xloc)
                # ax[l].xaxis.set_major_formatter(xfmt)

                # ax[l].yaxis.major = matplotlib.axis.Ticker()
                # yloc = matplotlib.ticker.AutoLocator()
                # yfmt = matplotlib.ticker.ScalarFormatter()
                # ax[l].yaxis.set_major_locator(yloc)
                # ax[l].yaxis.set_major_formatter(yfmt)
                
                print("ff is", ff)
                if ff in [6,7,18,19,20]:
                    routeclr = '#ffb16d'
                    routename = 'B'
                else:
                    routeclr = '#33b864' #'#2a7e19' #
                    routename = 'A'

                alllocs_df.iloc[:2000].plot.scatter(x=0, y=1, c=routeclr, marker='.', s=2, ax=ax[l], label='route ')#+f"{routename}") ########[ax[l].scatter( utm.from_latlon(all_BS_coords[bs.split('-')[1]][0], all_BS_coords[bs.split('-')[1]][1])[0:2][0],utm.from_latlon(all_BS_coords[bs.split('-')[1]][0], all_BS_coords[bs.split('-')[1]][1])[0:2][1], marker='1', s=100, label= bs.split('-')[1][:3] ) for bs in df_allrx.columns.values] 
                [ax[l].scatter( utm.from_latlon(all_BS_coords[bs.split('-')[1]][0], all_BS_coords[bs.split('-')[1]][1])[0:2][0], utm.from_latlon(all_BS_coords[bs.split('-')[1]][0], all_BS_coords[bs.split('-')[1]][1])[0:2][1], marker='1', c='C{}'.format(uu+103), s=100, label= bs.split('-')[1][:3] ) for uu, bs in enumerate(df_allrx.columns.values)]  #FFB6C1, #ffb16d
                
                lat_y_40s, long_x_11s =  matched_row_ingt.iloc[0][3], matched_row_ingt.iloc[0][4] # ax[1].scatter( long_x_11s, lat_y_40s, c='k', marker='o', s=10)
                easting_lngs_x, northing_lats_y = utm.from_latlon(lat_y_40s, long_x_11s)[0:2] # print("easting_lngs_x, northing_lats_y", easting_lngs_x, northing_lats_y, "\n")
                         
                ax[l].scatter(easting_lngs_x, northing_lats_y, c='k', marker='*', s=12, label="loc" ) #, label= f"|v|:{round(this_speed.values[0],1)} mps"  # [ax[l].scatter( utm.from_latlon(all_BS_coords[bs.split('-')[1]][0], all_BS_coords[bs.split('-')[1]][1])[0:2][0], utm.from_latlon(all_BS_coords[bs.split('-')[1]][0], all_BS_coords[bs.split('-')[1]][1])[0:2][1] , marker='1', s=100, label= bs.split('-')[1] )    for bs in df_allrx.columns.values]

                
                sdf=900

                """       
                         # # # # ylim_u = max(np.array(yys))+sdf
                        # # # # ylim_d = min(np.array(yys))-sdf
                        # # # # xlim_r = max(np.array(xxs))+sdf
                        # # # # xlim_l = min(np.array(xxs))-sdf
                        # # # # print(ylim_u, ylim_d, xlim_r, xlim_l)
                        # # # ax[1].yaxis.set_ticks([])
                        # # # ax[2].yaxis.set_ticks([])
                        # # # ax[3].yaxis.set_ticks([])
                        # # # ax[4].yaxis.set_ticks([])
                        # # # plt.suptitle(f'{n}/{n_total_measurements} Speed = {this_speed.values[0]} mps')                               
                        # # # plt.suptitle(f'{n}/{n_total_measurements}, Est CFO by {cfomthd_forthismsrmnt}: {round(cfo_approx,2)} Hz, \n Speed: {this_speed.values[0]} mps,  {fnm}')                               
                        # # # plt.figure("psdVsloc").savefig("./"+f"{n}_{fnm}"+"psdVsloc.pdf",format='pdf')
                """

                [ea.grid(True) for ea in ax.flatten()]

                xl = all_BS_coords_df.loc['bes'][3]-100
                xh = all_BS_coords_df.loc['smt'][3]+600

                yl = all_BS_coords_df.loc['bes'][2]-500
                yh = all_BS_coords_df.loc['hospital'][2]+500

                # ax[l].set_xlim(429383-sdf, 429383+sdf-100)
                # ax[l].set_ylim(4513165-sdf+100, 4513165+sdf)

                ax[l].set_xlim(xl, xh)
                ax[l].set_ylim(yl, yh)

                # new_xticks = ax[l].get_xticks() - ax[l].get_xticks()[0]
                # ax[l].set_xticks(new_xticks)
                # ax[l].tick_params(axis='x', rotation=90)

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
                plt.draw()
                plt.pause(0.01)

                
                # plt.figure("psdVsloc").savefig(f"{overall_plots_dir}" +"/"+f"{n}_{fnm}"+"psdVsloc.pdf",format='pdf')
                
                # fnplot = f"{overall_plots_dir}" +"/"+ f"{n}.pdf"
                # plt.savefig(fnplot,format='pdf')


            

        print("Rows traversed: %i/%i" %(n+1, n_total_measurements), end='\r')



    if plott:
        plt.ioff()
        plt.close("psdVsloc")
        my_plot_rms_dicts(fnm,rmsdict,overall_plots_dir)
        plot_all_off_dictionaries(ff, fnm, cfo_summary_dict, overall_plots_dir)
    

    print(f'\nSize of a single spectrum : {len(freq_full[fdidx])}')

    final_totallength = [len(val) for k, val in enumerate(data_and_label_dict.values()) ]
    print("How many rows we got in this pickled data file", final_totallength, f"\nTotal {len(no_measr_time_idx_n) + nogpsgt} rows skipped" )
    
    same_numb_rows = all(element == final_totallength[0] for element in final_totallength)   
    if same_numb_rows:
        print("Storing only iff each BS/columns got same number of rows.")
        nofrows = np.unique(final_totallength)
        
        ## Store one pickle file for one hdf5 data file in the common directory
        fn = f"{args.dircommon}"+"/"+ f"{args.dirdata}".split('/')[-1]+'.pickle'

        # pkl.dump((data_and_label_dict, metadata_dict, cfo_summary_dict), open(fn, 'wb' ) )
        # print("\n\nPickled!\n\n\n\n-----------------------------------------------------------\n\n\n\n")
        
        return metadata_dict, cfo_summary_dict, data_and_label_dict, nofrows

    else:
        print("\n\nColumns got different number of rows.")
        exit()  
    ############################################################################################################
    ############################################################################################################

  
###############################################################
############################################################### 


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

            print(f"\n\nProcessing the data in {uu}th {args.dirdata} directory\n")
            
            # if uu<5:
            #     continue

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
            
            totalrows += nofrws # pdb.set_trace() # break   

    print(f"\n\nAll {NumberofDatafolders} data folders in the common data directory {args.dircommon} done! Total valid rows collected = {totalrows}\n\n")
    
    print("Time now  = ", datetime.now().strftime("%H:%M:%S"))

    # exec(open("/Users/aartisingh/Documents/DopplerSpreadLocalization/mine_psd_fetching.py").read()) 

    # pdb.set_trace()

