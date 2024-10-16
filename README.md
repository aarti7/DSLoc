# Doppler spread based localization
We implement the following strategy for generating the fingerprint database $F$. 

We perform a series of large-scale, over-the- air experiments on the POWDER platform \cite{}. In total $N=5$ base-stations ( namely Behavioral, Honors, Hospital, SMT, UStar) are utilized.  

### Experiments

A mobile node continuously transmits an unmodulated, complex sinusoidal wave at a frequency $f_c = 3515$ MHz. The bandwidth of operation is $10$ MHz. The B210 radios that are used for transmission, itself do not have a fixed power level of transmission, as the power level is determined by the RF front-end and the antenna used with the SDR. The AD9361 RFIC transceiver used in the B210 provides an adjustable and programmable output power ranging from -4 dBm to 6 dBm.

 Multiple base-stations simultaneously record the IQ values during the duration of each of the experiments. Each base-station has a USRP X310 receiver. The power of operation is and the receive gain is maintained as 30 dB. All the receiver X310 have an external clock of base-stations are time and frequency synchronized with the White Rabbit Synchronization system \cite{}.




<!--Measuring Doppler Spread using POWDER-->
<!--frequenices of operation-->
<!--calibration or synchorinization-->



<!--
SHOUT: 
piping issues: measurements dropped.
More robust signal reception.
-->

### Install the required packages:

1. Following python packages needs to be installed: 
	* numpy 
	* pandas 
	* scipy
	* h5py
	* pytz
	* matplotlib
	* utm
	
2.  To install the above packages, open the terminal and run: `pip install -r requirements.txt`.



###  List of scripts:
4. `hdf5_to_pickles.py`: 
5. `functions_cforemoval.py`: 
6. `pickles_to_spectrum_postprocessing.py`:

* Additional scripts: `libs.py`: For the `imports` needed. 


  


### Hdf5 to pandas dataframe conversion
*  Go to the common data directory that has all the experiments folders generated by SHOUT. The naming convention used is `Shout_meas_<date>_<mountaintime>` For example: `Shout_meas_01-30-2023_15-40-55`. Each folder must have these files:  Run the  `hdf5_to_pickles.py` script as : `python hdf5_to_pickles.py`. 

	* Flags to alter:
            <!--args.frqcutoff = args_old.frqcutoff
            args.ofstremove = args_old.ofstremove            
            args.mpltflag = args_old.mpltflag
            args.pwrthrshld = args_old.pwrthrshld
            args.maxspeed = args_old.maxspeed            
            args.degoffit = args_old.degoffit
            args.window = args_old.window-->
   

* List of functions in the `hdf5_to_pickles.py` script:
<!-- * Currently used functions:	
-->   
	  1. `parse_args_def` 
     6. `leaves_to_DF`
     8. `do_data_storing`
     9. `get_dataset_keys`
     11. `parse_args_def` 
   
 <!-- 1. `convert_strptime_to_currUTCtsdelta`-->
     <!--2. `calcDistLatLong`-->
    <!-- 2. `get_avg_power`-->
     <!--3. `get_full_DS_spectrum`-->
     <!--4. `psdcalc`-->
    <!-- 5. `match`-->
     <!--7. `freq_off_averaged_for_full_df`-->

<!-- * Dormant functions
     1. `meridian_convergence`
     2. `meridian_convergence2`
     3. `fancy`
     4. `getrmsFreq`
     5. `offset_estimation_souden`
     6. `freq_off_alternate`
     7. `fit_freq_on_time`
     8. `fit_freq_on_time`
     9. `get_interpolated_foff`
     10. `plot_all_off_dictionaries`-->

* List of functions in the `functions_cforemoval.py` script:

     12. `get_full_DS_spectrum`
     13. `fit_freq_on_time`
     14. `mylpf`
     15. `get_cfo`
     16. `convert_strptime_to_currUTCtsdelta`
     17. `get_interpolated_foff`
     18. `do_cfo_removal`

   
* For each experiment, one pickle file gets stored in the common data directory
<!--*  and not in the git directory to avoid branch updates. --> Naming convention maintained is `Shout_meas_<date>_<mountaintime>.pickle`. For example: `Shout_meas_01-30-2023_15-40-55.pickle`

* List of functions in the `pickles_to_spectrum_postprocessing.py` script. Plots the psds. It can be seemlessly run after all the pickle files are done getting generated iff the flag -m is set to 1. (Default is 0)




### Data analysis

## Acknowledgements
This material is based upon work supported by the National Science Foundation under Grant Numbers 2232464, 1827940, and 1622741.
