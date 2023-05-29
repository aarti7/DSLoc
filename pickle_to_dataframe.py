#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('reset', '-f')
from libraries_FIELDS import * 


# In[7]:


pickle_dir = './pickle_files'
pickled_files = list(sorted(Path(pickle_dir).rglob('*.pickle')))
colorplate = plt.cm.get_cmap('viridis', len(pickled_files))
pickled_files, len(pickled_files)


# In[17]:


full_data_df = pd.DataFrame()
cforeacheachile = ['r','b','g','k','c','m'] 

for i, filename in enumerate(pickled_files):

    
    fn = '_'.join(str(filename).split('_')[3:5])
    print(fn)
    loaded_dicts = pkl.load(open(filename, 'rb') )
    
    ## Metadata of each experiment
    this_exp_metadata = loaded_dicts[1]
    timstampofExpMT = datetime.datetime.strptime(this_exp_metadata['exp_datetime'], "%Y-%m-%d %H:%M:%S").astimezone( pytz.timezone("America/Denver"))
    this_exp_metadata['exp_datetime'] = timstampofExpMT
    
    ## Data for each experiment
    this_exp_data = loaded_dicts[0]
    
   ## Plot the traversed route
    fig_ori, ax_ori = plt.subplots()
    y_utmlat = pd.DataFrame(this_exp_data)['position_lat']
    y_utmlon = pd.DataFrame(this_exp_data)['position_lon']
    j = np.random.randint(0,6)    
    for ii in range(len(y_utmlat)):
        
        xx = y_utmlon.iloc[ii] 
        yy = y_utmlat.iloc[ii] 
        
        ax_ori.scatter(xx, yy, c=cforeacheachile[j], marker='o', s=2)
    
    ax_ori.scatter( y_utmlon.iloc[0], y_utmlat.iloc[0] , marker='*', s=15, label = 'first loc', c = 'k')
    ax_ori.legend()
    ax_ori.grid(True)
    plt.title(f"Tracjectory of Exp: {fn} \n Number of collected measurements: {len(y_utmlat)}") 
    plt.show()
    
    
    ## Plot PSD
    df = pd.DataFrame(this_exp_data)
    for r in range(len(df)):
        this_speed = df['gpsspeed'].iloc[r]
        
        fig, axx = plt.subplots(1, 5, figsize=(20, 4), num="narrowspectrum", sharey= False) 
        axx[0].set_ylabel('mag^2(db)')
  
        for p in range(0,5):
            this_bs = df.columns.values[p].split('-')[1]
            
            pxxc_ns_DB = np.nan_to_num(10.0 * np.log10(np.square(np.abs(df.iloc[r,p])) ))
            phase_ns   = np.angle(df.iloc[r,p])
            
            axx[p].plot(pxxc_ns_DB) # axx[p].plot(phase_ns)
            
            axx[p].set_title(f"{this_bs}")
            axx[p].grid(True)
                             
        plt.figure("narrowspectrum").suptitle( f"Current speed: {this_speed}", y= 1, color = 'g')
        plt.show()
        plt.pause(1)
        print("breaking after showing only the 1st sample for each experiment!\n\n")
        break
            
    ## Append now!
    full_data_df = pd.concat([full_data_df, pd.DataFrame(this_exp_data)])

print(" Full dataframe stacked all experiments ====> \n", len(full_data_df))    


# In[18]:


full_data_df


# In[ ]:




