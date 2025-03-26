from libs import * 


######################################## BaseStation coordinates ########################################
#########################################################################################################


# all_BS_coords = {
#   'hospital': (40.77105, -111.83712),  
#   'ustar': (40.76899, -111.84167),
#   'bes': (40.76134, -111.84629),
#   'honors': (40.76440, -111.83695),
#   'smt': (40.76740, -111.83118),
# }

def latlon_to_utm(lat_lon):
    lat, lon = lat_lon
    utm_coords = utm.from_latlon(lat, lon)
    return utm_coords[1], utm_coords[0]  # Northing, Easting


    
######################################### Coordinate plots with detour removal ##########################
#########################################################################################################


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
        routewas = 'green'
        filtered_df = green_detour(name, df, plotflag)  
    else: 
        print(". Bus route was orange.", end="")
        routewas = 'orange'
        filtered_df = orange_detour(name, df, plotflag)
    
    print(" The filtered df's shape:", filtered_df.shape)
    return routewas#, filtered_df      inplace is set to true so not needed to return df spcficially, which saves from catching it later      




#########################################################################################################
