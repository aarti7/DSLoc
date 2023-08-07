from libraries_FIELDS import *


#######################################################################################################################################################################################  PLOTS   ###################################################################
###############################################################################################################################################################################################################################################################


all_BS_coords = {
  'hospital': (40.77105, -111.83712),  
  'ustar': (40.76899, -111.84167),
  'bes': (40.76134, -111.84629),
  'honors': (40.76440, -111.83695),
  'smt': (40.76740, -111.83118),
}


##################################################################################################################################
##################################################################################################################################


def INFER_AND_plotall(tr_dict, trndmdl, trdf, vldf, trnsf, tsdf, dev, m, lr, dp, e, wd, lat_SS, lon_SS, afn, crt, opsize):
    
    print("In the inferring function")
    device = dev
    
#     fi , ax = plt.subplots(2,1)
#     train_loss_hist              =  tr_dict['Train']        
#     val_loss_hist                =  tr_dict['Val']
#     ax[0].plot(train_loss_hist, linewidth=2,  linestyle='--')
#     ax[0].set_title("Training loss")
#     ax[0].set_xticks([])

#     ax[1].plot(val_loss_hist, linewidth=2  , linestyle='--')
#     ax[1].set_title("Validation loss")
#     ax[1].set_xlabel('epochs')
#     fi.suptitle(f"{m} epochs:{e} LR:{lr} dropout:{dp} wdecay:{wd}")
    
    
    #############################################################
    test_ds_after_tr = DatasetcoustomClass_forTesting(trdf, trnsf, opsize)
    test_dl_after_tr = DataLoader(test_ds_after_tr, batch_size=1, shuffle=False)

    test_results_training_dsT = {}
    with torch.no_grad():
        for k, v in tqdm.tqdm(test_dl_after_tr):    
            res = trndmdl(v[0].float().to(device))
            test_results_training_dsT.update(dict(zip(k, torch.cat(  (res, v[1].to(device)), dim=1)   )))
    print("Inference on Train Data DONE")   
    
    
    
    #############################################################
    test_ds_after_tr = DatasetcoustomClass_forTesting(vldf, trnsf, opsize)
    test_dl_after_tr = DataLoader(test_ds_after_tr, batch_size=1, shuffle=False)

    test_results_training_dsV = {}
    with torch.no_grad():
        for k, v in tqdm.tqdm(test_dl_after_tr):    
            res = trndmdl(v[0].float().to(device))
            test_results_training_dsV.update(dict(zip(k, torch.cat(  (res, v[1].to(device)), dim=1)   )))
    print("Inference on Val Data DONE")   
    
    
    #############################################################
    test_ds_after_tr = DatasetcoustomClass_forTesting(tsdf, trnsf, opsize)
    test_dl_after_tr = DataLoader(test_ds_after_tr, batch_size=1, shuffle=False)
    
    test_results_training_dsTs = {}
    test_disterror = []
    test_veloerror = []

    test_speed_error = []
    test_track_error = []
    
    how_many_stationary_gt = 0
    
    with torch.no_grad():
        for k, v in tqdm.tqdm(test_dl_after_tr): 

            img         = v[0].float().to(device)
            
            res     = trndmdl(img) # inference!  # ARE UTM but scaled! #We predict UTMcoords & convert to GPS_lat-lon coords
            gt      = v[1]         # groundtruth!# ARE UTM but scaled! 

            if opsize == 2:
                res_lat     = res[0][0].cpu()
                res_lon     = res[0][1].cpu()
                
                gt_lat      = gt[0][0].cpu()
                gt_lon      = gt[0][1].cpu()
                
            if opsize == 4:            

                res_speed   = res[0][0].cpu()
                res_track   = res[0][1].cpu()

                gt_speed    = gt[0][0].cpu()
                gt_track    = gt[0][1].cpu()    
                
#                 if gt_speed == 0:
#                     how_many_stationary_gt +=1
                
#                 if np.any(np.array(gt_speed) != 0) and np.any(np.array(res_speed) != 0):
#                     vel_vector_error = cosine_similarity((res_speed, res_track), (gt_speed, gt_track))
#                 else:
#                     speed_error = np.abs(res_speed - gt_speed)
#                     track_error = math.cos(res_track - gt_track) 
#                     vel_vector_error = speed_error*track_error
                    
#                 test_veloerror.append(vel_vector_error)

                test_speed_error.append(np.abs(res_speed - gt_speed))
                test_track_error.append(np.abs(res_track - gt_track))


                res_lat     = res[0][2].cpu()
                res_lon     = res[0][3].cpu()
                                
                gt_lat      = gt[0][2].cpu()
                gt_lon      = gt[0][3].cpu()
                          
            
            res_lat = lat_SS.inverse_transform(np.array(res_lat).reshape(1, 1))
            res_lon = lon_SS.inverse_transform(np.array(res_lon).reshape(1, 1)) #print("predicted descaled UTM_coords:",res_lat, res_lon) 

            rescord     = utm.to_latlon(res_lat, res_lon, 12, 'T') #back-convert predictions to GPS lat-lon coordinates

            
            gt_lat  = lat_SS.inverse_transform(np.array(gt_lat).reshape(1, 1))
            gt_lon  = lon_SS.inverse_transform(np.array(gt_lon).reshape(1, 1)) #print("GT descaled UTM coords:", gt_lat, gt_lon)

            gtcord      = utm.to_latlon(gt_lat, gt_lon, 12, 'T') #back-convert GT to GPS lat-lon coordinates
            
            disterror   = calcDistLatLong(gtcord, rescord)
            test_disterror.append(disterror)
            

#             print("Prediction     : lat", res_lat , "lon", res_lon )        
#             print("rescaled pred  : lat", unscaled_res_lat,"rescaled pred lon",  unscaled_res_lon)
#             print("Inferred coords:",rescord)  
#             print("Ground Truth   : lat", gt_lat , "lon", gt_lon)        
#             print("rescaled gt    : lat", unscaled_gt_lat, "rescaled gt lon", unscaled_gt_lon)
#             print("Actual coords  :",  gtcord)            
#             print(disterror)

#             test_results_training_dsTs.update(dict(zip(k, torch.cat((res_loc, gt_loc.to(device) ), dim=1)   ))) # for plotting!
            test_results_training_dsTs.update(dict(zip(k, torch.cat((res, gt.to(device) ), dim=1)   ))) # Do UTM plotting not scaled and not lat and long!
            
    print("Inference on Test Data DONE")
    print("last tags: prediction", res, "\nGT", gt)
    
    #############################################################
    print("plotting")
    
    mean_DIST_error_test_ds = np.array(test_disterror).mean() #print("Mean loc error for test dataset", mean_loc_error_test_ds)
    
    figresults , axresults = plt.subplots(1,4, figsize=(14,3))


    for ax in axresults.flat:
        ax.set_xlim([-3.5, 3])
        ax.set_ylim([-3.5, 2])

    for k, est in test_results_training_dsT.items():
        est = est.cpu()
        if opsize == 2:        
            axresults[0].scatter(est[2], est[3], c='k', marker='x', s=2) 
            axresults[0].scatter(est[0], est[1], c='g', marker='x', s=2) 
        if opsize == 4:    
            axresults[0].scatter(est[6], est[7], c='k', marker='x', s=2) 
            axresults[0].scatter(est[2], est[3], c='g', marker='x', s=2) 
        
        axresults[0].set_title("inference on training points")

    for k, est in test_results_training_dsV.items():
        est = est.cpu()
        if opsize == 2:
            axresults[1].scatter(est[2], est[3], c='k', marker='x', s=2) 
            axresults[1].scatter(est[0], est[1], c='m', marker='x', s=2) 
        if opsize == 4:    
            axresults[1].scatter(est[6], est[7], c='k', marker='x', s=2) 
            axresults[1].scatter(est[2], est[3], c='m', marker='x', s=2) 
        
        axresults[1].set_title("inference on validation points")

    for k, est in test_results_training_dsTs.items():
        est = est.cpu()
        
        cc = 'C{}'.format(k)  # Use C0, C1, C2, ... for the color
        if opsize == 2:        
            axresults[2].scatter(est[2], est[3], c='k', marker='x', s=2) 
            axresults[2].scatter(est[0], est[1], c='r', marker='x', s=2) 
            axresults[2].plot([est[0], est[2]], [est[1], est[3]], '--', alpha=0.3, c=cc,  linewidth=1)
        if opsize == 4:
            axresults[2].scatter(est[6], est[7], c='k', marker='x', s=2) 
            axresults[2].scatter(est[2], est[3], c='r', marker='x', s=2) 
            axresults[2].plot([est[2], est[6]], [est[3], est[7]], '--', alpha=0.3, c=cc,  linewidth=1)

        axresults[2].set_title("inference on testing points")

        
    train_loss_hist              =  tr_dict['Train']        
    val_loss_hist                =  tr_dict['Val']
    
    
    
    
    
    print("val_loss_hist length", len(val_loss_hist),"\n", val_loss_hist)
        
    
    if len(val_loss_hist) == 0:
        val_loss_hist=[0]
    
    maxvalueTOPLOTonYaxis = (np.max(val_loss_hist)+2)
    print("maxvalueTOPLOTonYaxis", maxvalueTOPLOTonYaxis) 

    
    
    axresults[3].set_ylim(0, maxvalueTOPLOTonYaxis)
    axresults[3].set_xlim(0,e-1)
    axresults[3].set_title("loss")
    axresults[3].set_xlabel('epochs')
    
    axresults[3].plot(val_loss_hist,   linewidth=2,  linestyle='--', label = "valloss")
    axresults[3].plot(train_loss_hist, linewidth=2,  linestyle='--', label = "trnloss")

#     # MULTIPLOTS
#     ax3_bottom = axresults[3].inset_axes([0, 0,    e, 0.3]) #[x0, y0, width, height]
#     ax3_bottom.plot(val_loss_hist,   linewidth=2,  linestyle='--', label = "valloss")
    
#     ax3_top    = axresults[3].inset_axes([0, 0.51, e, 0.3])
#     ax3_top.plot(train_loss_hist, linewidth=2,  linestyle='--', label = "trnloss")
#     ax3_top.set_xticks([])
    
    
    plt.legend()
    plt.tight_layout() # Adjust spacing between subplots #plt.subplots_adjust(hspace=0.5)
    
    
    if opsize == 2:
        figresults.suptitle(f"{m} epochs:{e} LR:{lr} dropout:{dp} l2_reg:{wd} AF:{afn} LF:{crt} \nMean loc error(m)\
        {mean_DIST_error_test_ds} Max:{np.array(test_disterror).max()}, OP = {opsize}", y=-0.5)
    
    elif opsize == 4:
        figresults.suptitle(f"{m} epochs:{e} LR:{lr} dropout:{dp} l2_reg:{wd} AF:{afn} LF:{crt} \nMean loc error(m):\
        {mean_DIST_error_test_ds} Max:{np.array(test_disterror).max()}, OP = {opsize}, mean_vel_error:\
        {np.nanmean(np.array(test_speed_error))}, {np.nanmean(np.array(test_track_error))}", y=-0.5)
    
    plt.savefig(f"./storingfolder/eps_plots/all_results_{m}_epochs_{e}_LR_{lr}_dropout_{dp}_l2reg_{wd}_AF_{afn}_LF_{crt}.png",format='png')
    
    extent = axresults[2].get_window_extent().transformed(figresults.dpi_scale_trans.inverted()) # Save just the portion _inside_ of the test reults, that is the second axis's boundaries
    
    plt.savefig(f"./storingfolder/eps_plots/test_results_{m}_epochs_{e}_LR_{lr}_dropout_{dp}_l2reg_{wd}_AF_{afn}_LF_{crt}.eps", bbox_inches=extent, format='png')
    plt.savefig(f"./storingfolder/eps_plots/test_results_expanded{m}_epochs_{e}_LR_{lr}_dropout_{dp}_l2reg_{wd}_AF_{afn}_LF_{crt}.png", bbox_inches=extent.expanded(1.1, 1.2), format='png')
    plt.show()

    
    
    
    
#############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################



 

def initialize_model(model_name, num_output_is=1):
       
    if model_name == "resnet18":
        model        = models.resnet18(pretrained = False)
        num_features = model.fc.in_features                                              #the fc layer of resnet has this number of input features coming in
        model.fc     = nn.Linear(in_features=num_features, out_features=num_output_is)   # create a new 'linear layer that takes in the same number of input features but outputs as many as you want for your problem
    
    
 
    
    elif model_name == "myCONV2D_bigger":
        model        =  myCONV2D_bigger()
   
    elif model_name == "myCONV2D":
        model        =  myCONV2D()

    elif model_name == "myCONV2D_bigger":
        model        =  myCONV2D_bigger()
   
    elif model_name == "myCONV2D_bigger_diffactivations":
        model        =  myCONV2D_bigger_diffactivations()
    else:
        print("Invalid model name, exiting...")
        exit()
    return model


###############################################################################################################################################################################################################################################################

# create a nn class (just-for-fun choice :-) 
class mycustomRMSELoss(nn.Module):
    def __init__(self, whatreduction = None):
        super().__init__()
        self.mse = nn.MSELoss(reduction = whatreduction) # keeping 'mean' for R'Mean'SE
        
    def forward(self,yhat,y):
        rmse_loss_calc = torch.sqrt(self.mse(yhat,y))
#         print("rmse_loss_calc",rmse_loss_calc)
        return rmse_loss_calc



# create a nn class (just-for-fun choice :-) 
class mycustomMedianAbsoluteErrorLoss(nn.Module):
    def __init__(self, whatreduction = None):
        super().__init__()
        self.mse = nn.L1Loss(reduction = whatreduction)
        
    def forward(self,yhat,y):
        medianAbsoluteLoss = torch.median( self.mse(yhat,y))
#         pdb.set_trace()
#         print("medianAbsoluteLoss",medianAbsoluteLoss) #dim=0).values
        return medianAbsoluteLoss

  
    
    
#############################################################################################################################
class DatasetcoustomClass(Dataset): # simply converts a df to ds  #extends the 'Dataset' class # PyT needs a class defined for Dataset creation unlike TF

    def __init__(self, df, transform_fucntion_is=None, outputsize = 1):#, minval=0, maxval=1):

        self.thisdata  = df 
        
        self.outputs   = outputsize
        
        self.transform = transform_fucntion_is
        
#         self.minval = minval
#         self.maxval = maxval        
        

    def __len__(self):
        return len(self.thisdata)
    
#     def whatss(self):  
#         return  self.minval, self.maxval
    
    
    def __getitem__(self, idx):
              
        
        if  self.outputs == 2:
            label              = torch.tensor((
                                           self.thisdata.iloc[idx,-1:][0][1][0],
                                           self.thisdata.iloc[idx,-1:][0][1][1])) # only pos, speed=.iloc[idx,-1:][0][0]
        elif self.outputs == 3:
            label              = torch.tensor((
#                                            self.thisdata.iloc[idx,-1:][0][0],    #speed
                                           self.thisdata.iloc[idx,-1:][0][1],    #track
                                           self.thisdata.iloc[idx,-1:][0][2][0], #lat
                                           self.thisdata.iloc[idx,-1:][0][2][1]  #long
                                           ))       

        elif self.outputs == 4:
            label              = torch.tensor((
                                           self.thisdata.iloc[idx,-1:][0][0],
                                           self.thisdata.iloc[idx,-1:][0][1],
                                           self.thisdata.iloc[idx,-1:][0][2][0],
                                           self.thisdata.iloc[idx,-1:][0][2][1]
                                           ))                                # YES! speed and Track AND position! 
            
            
            
        else:
            print("number of output incompatible with this model")
            exit()
            
        twoDimage_psdallBS = torch.tensor(np.array(list(self.thisdata.iloc[idx, :-1])))      
         
#         twoDimage_psdallBS_subtractedmin = twoDimage_psdallBS-self.minval
#         twoDimage_psdallBS = twoDimage_psdallBS_subtractedmin/self.maxval
    
        if self.transform:
            twoDimage_psdallBS = twoDimage_psdallBS.repeat(3,1,1)
        else:
            twoDimage_psdallBS = torch.unsqueeze(twoDimage_psdallBS , dim=0) 
           
       
        return twoDimage_psdallBS, label  



###############################################################################################################################################################################################################################################################
###############################################################################################################################################################################################################################################################
###############################################################################################################################################################################################################################################################



class DatasetcoustomClass_forTesting(Dataset):
    # In test_stage, no labels will be needed inside the dataset object!
    
    def __init__(self, df, transform_fucntion_is=None, outputsize = 1):#, minval=0, maxval=1):
        self.thisdata = df    
        self.outputs   = outputsize
        
        self.transform = transform_fucntion_is
        
#         self.minval = minval
#         self.maxval = maxval
        

    def __len__(self):
        return len(self.thisdata)
    
#     def whatss(self):
        
#         return self.minval, self.maxval
    
    def __getitem__(self, idx): 

        if  self.outputs == 2:
            label              = torch.tensor((
                                           self.thisdata.iloc[idx,-1:][0][1][0],
                                           self.thisdata.iloc[idx,-1:][0][1][1])) # only pos, speed=.iloc[idx,-1:][0][0]
        elif self.outputs == 3:
            label              = torch.tensor((
#                                            self.thisdata.iloc[idx,-1:][0][0],  #speed
                                           self.thisdata.iloc[idx,-1:][0][1],    #track
                                           self.thisdata.iloc[idx,-1:][0][2][0], #lat
                                           self.thisdata.iloc[idx,-1:][0][2][1]  #long
                                           ))       
        elif self.outputs == 4:
            label              = torch.tensor((
                                           self.thisdata.iloc[idx,-1:][0][0],
                                           self.thisdata.iloc[idx,-1:][0][1],
                                           self.thisdata.iloc[idx,-1:][0][2][0],
                                           self.thisdata.iloc[idx,-1:][0][2][1]
                                           ))        # YES! speed and Track AND position!
        else:
            print("number of output incompatible with this model")
            exit()
        
        twoDimage_psdallBS = torch.tensor(np.array(list(self.thisdata.iloc[idx, :-1]))) 
        
#         twoDimage_psdallBS_subtractedmin = twoDimage_psdallBS-self.minval
#         twoDimage_psdallBS = twoDimage_psdallBS_subtractedmin/self.maxval
        
        if self.transform:
            twoDimage_psdallBS = twoDimage_psdallBS.repeat(3,1,1)# this adds third/channel dim AND repeats/copy data in them!
        else:
            twoDimage_psdallBS = torch.unsqueeze(twoDimage_psdallBS , dim=0)# this addsthird dimension, no copying will be done
        
        # later dl will add the batch dimension!    
       
        return str(idx), (twoDimage_psdallBS, label)


#################################################################################################################################
################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################




def trainVal_onego(dataloader_dict_tr, model, criterion, optimizer, num_epochs=11, batch_size_is = 2, devi='cpu'):    
    
    device=devi
    print("\nNOW Training!!!!! ====> ", model._get_name(),"\n")
    
    since = time.time()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = num_epochs, verbose=False)

    best_model_wts = deepcopy(model.state_dict())
    thisfileloss_history_dict =  dict((k, []) for k in dataloader_dict_tr.keys()) 
    max_loss  = 100000

    txcolr = { "Train": "\033[32m",
               "Val"  : "\033[35m",
               "norm" : "\033[30m"
         }
        
    for epoch in tqdm.tqdm(range(num_epochs)): 
        
#         print('Epoch {}/{}'.format(epoch, num_epochs - 1))
#         print('-' * 10)    

        loss_thisphase_accumulated_over_full_dataloader = 0

        for phase in ['Train', 'Val']:
            batchnum = 0
            for inputs, labels in  dataloader_dict_tr[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs.float()) 
#                 pdb.set_trace()
                
                # torch.sum(torch.abs   (torch.subtract(outputs, labels)))      ## nn.L1Loss(reduction='sum')
                # torch.sum(torch.square(torch.subtract(outputs, labels)))      ## nn.MSELoss(reduction='sum')

                loss = criterion(outputs, labels.to(device)) 
#                 print("loss is ", loss, loss.item())
                if phase == 'Train': 
                    loss.backward()
                    optimizer.step()     
                
                loss_thisphase_accumulated_over_full_dataloader += loss.item()
                batchnum = batchnum+1
            
            this_epoch_loss = loss_thisphase_accumulated_over_full_dataloader / (len(dataloader_dict_tr[phase])*batch_size_is )
            
#             print(txcolr[phase] + 'For Phase {} :::: The epoch loss is: {:.4f}'.format(phase, this_epoch_loss ) + txcolr["norm"])            
            thisfileloss_history_dict[phase].append(this_epoch_loss) 

            if phase == 'Val' and this_epoch_loss < max_loss:
                max_loss = this_epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict()) 

        scheduler.step() 
            
    train_val_time_consumed_eachfile_after_allepochs = time.time() - since
    print('Training complete for this file in {:.0f}m {:.0f}s'.format(train_val_time_consumed_eachfile_after_allepochs // 60, train_val_time_consumed_eachfile_after_allepochs % 60))

    model.load_state_dict(best_model_wts) 
    
    return model, thisfileloss_history_dict



       ##################################################################################################################################################################################### Additionals ############################################################
##############################################################################################################################################################################################################################################################
    '''
    lat_y  =  labels.iloc[ii][-1][0]
    long_x =  labels.iloc[ii][-1][1]
    ax_ori.scatter( long_x, lat_y, c='C{}'.format(i), marker='o', s=2)

    # Thus:
    #   longitude:  x axis
    #   latitude:   y axis


    # lati
    # | 
    # |
    # |
    # |
    # |_________________ longi



    # Convert to UTM easting and northing!!
    # (lat=y, lon=x) --->>>>UTM ----->>> easting=x, northing=y !! THIS IS WHERE IT FLIPS!
    
    # northing
    # | 
    # |
    # |
    # |
    # |_________________ easting

     
    # the UTM x-axis is referred to as easting, and the UTM y-axis is referred to as northing

    # Thus:
    #   easting:  x axis
    #   northing:   y axis

    easting_x  = labels.iloc[ii][-1][0], 
    northing_y = labels.iloc[ii][-1][1]
    ax_ori.scatter( easting_x, northing_y, c='C{}'.format(i), marker='o', s=2)

    '''


#############################################################################################################################################################################################################################################################################################################################################################################################
###############################################################################################################################


    
###############################################################################################################################################################################################################################################################
    
def split_tuple(row):
    row['speed'], row['pos'] = row['speed_postuple']
    return row

def split_pos(row):
    row['lat'], row['lon'] = row['pos']
    return row

def split_tuple_track(row):
    row['speed'], row['track'], row['pos'] = row['speed_postuple']
    return row
###############################################################################################################################################################################################################################################################

def calcDistLatLong(coord1, coord2):
    # pdb.set_trace()
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

###############################################################################################################################################################################################################################################################

def normalize_labels(labels, scaler):
    labels = labels.reshape(-1, 1)
    normalized_labels = scaler.fit_transform(labels)
    return torch.tensor(normalized_labels, dtype=torch.float32).squeeze()
###############################################################################################################################################################################################################################################################

def denormalize_labels(predicted_labels, scaler):
    predicted_labels = predicted_labels.detach().numpy()
    denormalized_labels = scaler.inverse_transform(predicted_labels)
    denormalized_labels = torch.tensor(denormalized_labels, dtype=torch.float32)
    return denormalized_labels


###############################################################################################################################################################################################################################################################


    
    
    
       
    
def quickplot_psd(df, this_exp_mtdata):
    ## Plot PSD
    
    nsamps    = this_exp_mtdata['nsamps'] #2**17
    samp_freq = this_exp_mtdata['rate']  #220000
    freq_res  = samp_freq/ nsamps
    
    for r in range(len(df)):
        this_speed = df['speed_postuple'].iloc[r][0]
        fig, axx = plt.subplots(1, 5, figsize=(20, 5), num="narrowspectrum", sharey= True)
        
        for ax in axx.flat: 
            ax.set_xlabel('frequency (Hz)')
                
        for p in range(0,5):
            
            this_bs = df.columns.values[p].split('-')[1]
            
             
            # x axis!
            freqs = np.fft.fftshift(np.fft.fftfreq(nsamps, 1/samp_freq))
            lenofpsd = len(df.iloc[r,p])
#             print("dont do the length and recalculation of the ds limits, instead read from dictionary like below")
#             ns_freq_res = freq_res* np.ceil(lenofpsd/2)
#             fxxc_ns = freqs[(freqs >= -ns_freq_res) & ( freqs <= ns_freq_res) ]
#             print(fxxc_ns[[0,-1]])
#             print("lens:", lenofpsd, len(fxxc_ns) )   
            
            ns_read = this_exp_mtdata['narrow_spectrum'] # same for all experiments: FDMAX + buffer
        
            ns_freq_res_left  = int(np.floor(-ns_read/ freq_res))*freq_res
            ns_freq_res_right = int(np.ceil(ns_read/freq_res))*freq_res
            fxxc_ns = freqs[(freqs >= ns_freq_res_left) & ( freqs <= ns_freq_res_right) ]
 

            #  y axis!

    
            ## if spectrum was stored
            pxxc_ns_DB = np.nan_to_num(10.0 * np.log10(np.square(np.abs(df.iloc[r,p])) ))
            phase_ns   = np.angle(df.iloc[r,p])
            axx[p].plot(fxxc_ns, pxxc_ns_DB)
            axx[0].set_ylabel('Power spectrum(db)')       

            
            
#             ## if linear psd from plt.psd was stored
#             pxxc_ns_linear = df.iloc[r,p]
#             pxxc_ns_DB = 10.0 * np.log10(pxxc_ns_linear)
#             axx[p].plot(fxxc_ns, 10.0 * np.log10(pxxc_ns_linear))
#             axx[0].set_ylabel('Power spectrum density(db/Hz)')     


            
            axx[p].set_title(f"{this_bs}")
            axx[p].grid(True)

        plt.figure("narrowspectrum").suptitle( f"Current speed: {this_speed}", y=1, color = 'g')
        plt.show()
        plt.pause(1)
#         print("breaking after showing only the 1st sample for each experiment!\n\n")
        break    

    
    
    
def quickplot_labels_non_utmed(i, fn, df):
   
    labels  = df["speed_postuple"]    
    bsnames = df.columns.values  
    
    fig_ori, ax_ori = plt.subplots(figsize=(10, 10))
    
    for ii in range(len(labels)):
        lat_y, long_x =  labels.iloc[ii][2][0], labels.iloc[ii][2][1]
        ax_ori.scatter( long_x, lat_y, c='C{}'.format(i), marker='o', s=2)
    ax_ori.scatter( labels.iloc[0][2][1], labels.iloc[0][2][0] , marker='*', s=400, label = 'first loc', c = 'y')

    [ax_ori.scatter(all_BS_coords[bs.split('-')[1]][1], all_BS_coords[bs.split('-')[1]][0], marker='1',
                    s=600, label= bs.split('-')[1] ) for b, bs in enumerate(bsnames) if b<5 ]    
    
    ax_ori.legend()
    ax_ori.grid(True)
    plt.title(f"Tracjectory of Exp: {fn} \n Number of collected measurements: {len(labels)}") 
    plt.show()  

    
    
    

def quickplot_labels_non_utmed_full_df(i, df):
    labels  = df["speed_postuple"]    
    
    fig_ori, ax_ori = plt.subplots(figsize=(10, 10))
    
    for ii in range(len(labels)):
        lat_y, long_x =  labels.iloc[ii][2][0], labels.iloc[ii][2][1]
        ax_ori.scatter( long_x, lat_y, c='C{}'.format(i), marker='o', s=2)
    ax_ori.scatter( labels.iloc[0][2][1], labels.iloc[0][2][0] , marker='*', s=400, label = 'first loc', c = 'y')


    ax_ori.legend()
    ax_ori.grid(True)
    plt.title(f"FULL DF not utmed: Number of collected measurements: {len(labels)}") 
    plt.show()


    
    
def quickplot_labels_tobeutmed_toshow(i, fn, df):
    
    labels  = df #["speed_postuple"]
        
    fig_ori, ax_ori = plt.subplots(figsize=(4, 4))
    for ii in range(len(labels)):
        easting_x, northing_y =  labels.iloc[ii][2][0], labels.iloc[ii][2][1]     # Convert to UTM easting and northing!!
    # (lat=y, lon=x) --->>>>UTM ----->>> easting=x, northing=y !! THIS IS WHERE IT FLIPS!
        ax_ori.scatter( easting_x, northing_y, c='C{}'.format(i), marker='o', s=2)
    
    ax_ori.scatter( labels.iloc[0][2][0], labels.iloc[0][2][1] , marker='*', s=400, label = 'first loc', c = 'y')
  
        
    ax_ori.legend()
    ax_ori.grid(True)
    plt.title(f"Tracjectory of Exp UTMED: {fn} \n Number of collected measurements: {len(labels)}") 
    plt.show()
        
    

    
    
    
def quickplot_labels_tobeutmed_toshow_full_df(i, df):
    
    labels  = df#["speed_postuple"]    
    
    fig_ori, ax_ori = plt.subplots(figsize=(4, 4))
    for ii in range(len(labels)):
        easting_x, northing_y =  labels.iloc[ii][2][0], labels.iloc[ii][2][1]     # Convert to UTM easting and northing!!
    # (lat=y, lon=x) --->>>>UTM ----->>> easting=x, northing=y !! THIS IS WHERE IT FLIPS!
        ax_ori.scatter( easting_x, northing_y, c='C{}'.format(i), marker='o', s=2)
    
    ax_ori.scatter( labels.iloc[0][2][0], labels.iloc[0][2][1] , marker='*', s=400, label = 'first loc', c = 'y')
    
    ax_ori.legend()
    ax_ori.grid(True)
    plt.title(f"FULL DF UTMED: Number of collected measurements: {len(labels)}") 
    plt.show()  
    
    

        