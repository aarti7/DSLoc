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

CFO_REMOVE_FLAG  = 0
MPLOT_FLAG       = 0 
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


def do_data_storing(attrs, allsampsandtime, leaves):
    
    """
    For each data_dir, this function stores one pickle file that has two dictionaries.

    1. Dict#1 has 
        a. Under one key for one basestation (column_name = name of BS station), complex-valued spectrum of the received signal at the BS is stored, but narrowed dow  to max_expected_Doppler_frequency_plus_CFofffset_plus_buffer.
        b. The last key, named 'speed_postuple', respective labels are stored as one tuple (that is, transmitter speed, transmitter track, (transmitter latitude, transmitter laongitude), for the matching timestamp from CSV_GPS to (the common to all BS) timstamp .

    

    Dict#1 format:

    {                   basestation_1              basestation_2                 .....            basestation_M                             speed_postuple        
    
    value_1 :       [complex_value_spectrum11]   [complex_value_spectrum12]                [complex_value_spectrum1M]       ( speed_r1, track_r1, (lat_r1, lon_r1), timestamp_r1)
      
    .                      .                               .                                           .                                     .                              
    .                      .                               .                                           .                                     .                               
    .                      .                               .                                           .                                     .                               
    
    value_N :              .                               .                                           .                                     .                                      


    }                 



    2. Dict#2 has the metadata of this one experiment's hdf5 file that was in the data_dir. ( name of bus transmitter, number of samples collected, sampling rate, center frequency, limit set for max_plus_buffer_expected_Doppler_frequency, timestamp of the start of the experiment,) 

     
    Dict#1 format:
    {
    "tx" : txis,
    "nsamps" : nsamps,
    "rate" : rate, 
    "tx_CENTER_FREQUENCY" : tx_CENTER_FREQUENCY,
    "narrow_spectrum":ns,
    "exp_datetime" : datetime.datetime.fromtimestamp(int(exp_timestampUTC)).astimezone( pytz.timezone("America/Denver")) #.strftime("%Y-%m-%d %H:%M:%S")
    }
    

    3. Optional: If offset_removal_flag that is the CFO_REMOVE_FLAG, was set to true (python hdf5_to_pickles.py -o 1). If set to False, empty summary dictionary returned.
    
    Dict#3 has the summary of the calculated CFOs gathered from all the instances at each BS, whenever the mobile node was stationary (|v|=0) during this one experiment.
    (Firstmost time instance, mean of all the stationary CFOs, coeffs of a polynomial of a user-specified degree that was fitted on collected stationary CFOs, all time and stationary CFO for per BS) 

    Dict#3 format:
    {
    'exp_start_timestampUTC': exp_start_timestampUTC,
    'meanmethod': mean_frqoff_perrx_dict, 
    'fitdmethod': fitd_frqoff_perrx_dict, 
    'allcfotime': freqoff_time_dict
    }


    """

    #### GT reading #############################
    
    loc_files = list(Path(args.dirdata).rglob('*.csv') ) # pick the csv outta cwd
    print('\nLocation gps csv file to read is:', loc_files, "\n")
    
    if len(loc_files) == 0:
        print(" No BUS Location csv file present!!!!!!!")
        exit(1)
    else:
        gt_loc_df = pd.read_csv(loc_files[0], header=0)
        gt_loc_df['Time (UTC)'] = pd.to_datetime(gt_loc_df['Time (UTC)']).dt.tz_convert('US/Mountain').dt.strftime("%Y-%m-%d %H:%M:%S")
        
        print('\nOverall length of CSV', gt_loc_df.shape, "\n\nMax bus speed in this experiment was:", gt_loc_df['Speed (meters/sec)'].max(),"\n")
    

    #### ATTRS reading #############################

    tx_CENTER_FREQUENCY  = attrs['txfreq']
    rx_CENTER_FREQUENCY  = attrs['rxfreq'] 
    WAVELENGTH           = scipy.constants.c / tx_CENTER_FREQUENCY 
      
    rate = attrs['rxrate']
    nsamps = attrs['nsamps']

    #### calling the function for reading hdf5 into df #############################
    txis, no_measr_time_idx_n, df_allrx, df_allti, exp_start_timestampUTC = read_leaves_to_DF(leaves, allsampsandtime)
    
    print("\nThis experiment's bus is       == ", txis, "\n\n")
    print('Name of the Base Stations  are   ==',  df_allrx.columns.values, "\n")
    
    n_endpoints_is_subplots = len(df_allrx.columns.values)
    n_total_measurements    = len(df_allrx)

    print('Number of RX Base Stations are   ==', n_endpoints_is_subplots, "\n")
    print('Number of total measurements     ==', n_total_measurements, "\n\n") 
    

    ## Hardcoded narrowed spectrum 
    ns = args.frqcutoff # in hz
    print("Hardcoded value:","Narrowed Spectrum freq_span ==", ns, "hz")
    
    ## 2 power threshold value
    pwr_threshold = args.pwrthrshld # -19# in dB
    print("Hardcoded value:","threshold ==", pwr_threshold, "\n")
    
    CFO_REMOVE_FLAG = args.ofstremove

    cfo_summary_dict = {}

    ######################################################################################################################################
    ######################################################################################################################################
    if CFO_REMOVE_FLAG:
        print("\n\nCFO removal flag set to True. So, removing approax CFO from spectrum!")
        ## Hardcoded values ##########################    
        
        ## 1 Cutoff frequency for LPF 
        lpf_fc = args.frqcutoff # 1e4 # in hz 
        # print("Hardcoded value:","Narrowed LPF freq_span ==", )
        
        ## 3 degree of polynomial fit value
        degreeforfitting = args.degoffit #3
        # print("Hardcoded value:","degree of fit ==", degreeforfitting, "\n")
        
        ## 4 max speed value
        MAXIMUM_BUS_SPEED_POSSIBLE = args.maxspeed #21 # in meters_per_second
        # print("Hardcoded value:","MAXIMUM_BUS_SPEED_POSSIBLE ==", MAXIMUM_BUS_SPEED_POSSIBLE)
        
        FD_MAXPOSSIBLE             = MAXIMUM_BUS_SPEED_POSSIBLE / WAVELENGTH
        # print("\nFD_MAXPOSSIBLE is          == " , FD_MAXPOSSIBLE, "\n" )
        
        ## 5 narrowed spectrum 
        bufferwindw = args.window    # 1e2 # is a little extra window as a buffer on both sides.  in hz
        
        ## Overall narrow spectrum = buffer + expected max doppler spread value. in hz
        ns = bufferwindw + FD_MAXPOSSIBLE 
        print("And new Hardcoded value:","Narrowed Spectrum freq_span when CFO removed ==", round(ns,4), "hz")

        ##lengthy df traversal 1 to get cfo ###################################################################
        
        ### calling the function for cfo calculation #############################
        cfo_summary_dict, no_measr_time_idx_n2, no_gps_mesrnt_idx_n2 = get_cfo(df_allrx, df_allti, gt_loc_df, rate, lpf_fc, exp_start_timestampUTC, degreeforfitting, pwr_threshold)

    
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
    
    ########### iterating dataframe to make pickles with (spectrum, label) 
    for n in range(0, n_total_measurements ):

        if n in no_measr_time_idx_n:
            print(f"{n}th measurment skipped cause its empty") # as this pth /all p msrmnts is missing! Neither should store or do labelling for those who are present")
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
                this_measurement, cfo_approx = do_cfo_removal(cfo_summary_dict, degreeforfitting, this_measr_timeuptoseconds, df_allrx.columns[p], this_measurement, rate)

            ## full spectrum 
            [full_spectrum, freq_full]   = get_full_DS_spectrum(this_measurement, rate)

            ## narrowed spectrum indexes 
            fdidx  = (freq_full >= -ns) & ( freq_full <= ns)     

            ## narrowed spectrum - will be pickling!
            data_and_label_dict[df_allrx.columns[p]].append(full_spectrum[fdidx])
            ####################################

            ## labelling here! Labels are to be stored as the last column
            if p==n_endpoints_is_subplots-1: # when at the last Rx's index, that is only after the last BS has been picked, we store loc, speed, track, and time in their respective columns
                data_and_label_dict["speed_postuple"].append([this_speed.values[0], direction_w_TN, current_bus_pos_tuple, this_measr_timeuptoseconds])
                        

        print("Rows traversed: %i/%i" %(n+1, n_total_measurements), end='\r')
    
    print(f'\n\nSize of a single spectrum : {len(full_spectrum[fdidx])}\n\n')

    final_totallength = [len(val) for k, val in enumerate(data_and_label_dict.values()) ]
    print("How many rows we got in this pickled data file", final_totallength, f"\n\n\nTotal {len(no_measr_time_idx_n) + nogpsgt} rows skipped" )
    
    same_numb_rows = all(element == final_totallength[0] for element in final_totallength)   
    if same_numb_rows:
        print("\n\nStoring only iff each BS/columns got same number of rows.")
        nofrows = np.unique(final_totallength)
        
        ## Store one pickle file for one hdf5 data file in the common directory
        fn = f"{args.dircommon}"+"/"+ f"{args.dirdata}".split('/')[-1]+'.pickle'

        pkl.dump((data_and_label_dict, metadata_dict, cfo_summary_dict), open(fn, 'wb' ) )
        print("\n\n\nPickled!\n\n\n\n-----------------------------------------------------------\n\n\n\n")
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
    parser.add_argument("-m", "--mpltflag",   type=int, default=MPLOT_FLAG,    help="The name of the per experiment directory. Default: %s" % MPLOT_FLAG)
    
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


    for uu, f in enumerate(sorted(Path(args_old.dircommon).iterdir())):
        if f.is_dir() and str(f).startswith('Shout_'):

            args = parser_itself.parse_args(['--dirdata', str(f) ]) #update the argument here!            
            args.frqcutoff = args_old.frqcutoff
            args.ofstremove = args_old.ofstremove            
            args.mpltflag = args_old.mpltflag

            args.pwrthrshld = args_old.pwrthrshld
            args.maxspeed = args_old.maxspeed            
            args.degoffit = args_old.degoffit
            args.window = args_old.window

            print(f"\n\n-----------------------------------------------------------\nProcessing the data in the {args.dirdata} directory\n")
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

            metadata_dict, cfo_summary_dict, datalabel_dict, nofrws = do_data_storing(needed_attrs_obj, commonANDdeepest_root_foralltxrx, leaves_withTX if args.treebranch =='wtx' else leaves_withoutTX) #all_leaves_2) # leaves_withTX)
            
            totalrows += nofrws 
            # break  
            

    print(f"\n\nAll {NumberofDatafolders} data folders in the common data directory {args.dircommon} done! Total valid rows collected = {totalrows}\n\n")
    

    print("Time now  = ", datetime.now().strftime("%H:%M:%S"))
    if args.mpltflag:
        exec(open("pickles_to_spectrum_postprocessing.py").read()) 


