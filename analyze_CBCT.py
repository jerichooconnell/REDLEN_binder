
# coding: utf-8

# In[1]:

import numpy as np
import scipy.io as sio
import scipy
from ShowPlots import *
import matplotlib
import matplotlib.patches as mpatches
from matplotlib.offsetbox import (DrawingArea, OffsetImage,AnnotationBbox)
    
matplotlib.rc('xtick', labelsize=8) 
matplotlib.rc('ytick', labelsize=8) 

import matplotlib.pyplot as plt
from PIL import Image
from skimage import data, color, io, img_as_float

import matplotlib.colors as colors

# Defining some colormaps
cdict = {'red':  ((0.0, 0.0, 0.0),   # no red at 0   # all channels set to 1.0 at 0.5 to create white
                  (1.0, 1.0, 1.0)),  # set to 0.8 so its not too bright at 1

        'green': ((0.0, 0.0, 0.0),   # set to 0.8 so its not too bright at 0  # all channels set to 1.0 at 0.5 to create white
                  (1.0, 0.0, 0.0)),  # no green at 1

        'blue':  ((0.0, 0.0, 0.0),   # no blue at 0   # all channels set to 1.0 at 0.5 to create white
                  (1.0, 1.0, 1.0))   # no blue at 1
       }

purp = colors.LinearSegmentedColormap('GnRd', cdict)

cdict = {'red':  ((0.0, 0.0, 0.0),   # no red at 0   # all channels set to 1.0 at 0.5 to create white
                  (1.0, 0.0, 0.0)),  # set to 0.8 so its not too bright at 1

        'green': ((0.0, 0.0, 0.0),   # set to 0.8 so its not too bright at 0  # all channels set to 1.0 at 0.5 to create white
                  (1.0, 1.0, 1.0)),  # no green at 1

        'blue':  ((0.0, 0.0, 0.0),   # no blue at 0   # all channels set to 1.0 at 0.5 to create white
                  (1.0, 0.0, 0.0))   # no blue at 1
       }

# Create the colormap using the dictionary
green = colors.LinearSegmentedColormap('GnRd', cdict)

# This dictionary defines the colormap
cdict = {'red':  ((0.0, 0.0, 0.0),   # no red at 0   # all channels set to 1.0 at 0.5 to create white
                  (1.0, 1.0, 1.0)),  # set to 0.8 so its not too bright at 1

        'green': ((0.0, 0.0, 0.0),   # set to 0.8 so its not too bright at 0  # all channels set to 1.0 at 0.5 to create white
                  (1.0, 0.85, 0.85)),  # no green at 1

        'blue':  ((0.0, 0.0, 0.0),   # no blue at 0   # all channels set to 1.0 at 0.5 to create white
                  (1.0, 0.0, 0.0))   # no blue at 1
       }

# Create the colormap using the dictionary
gold = colors.LinearSegmentedColormap('GnRd', cdict)



def main(energy_bin,directory):
    '''
    Run this one first to plot original images
    :return:
    '''
    #load images
    #directory = "./data_no_smoothing/"
    dir_images = "./images/"
    slice_no = 15  # slice 9 in matlab
    #energy_bin = "0" #SEC1, EC, etc
    radius = 1.593

    data = sio.loadmat(directory + "binEC_uncorrected_round2_" + energy_bin +"_multiplex_corrected2.mat")
    img = data['Reconimg'][:,:,slice_no]

    size = np.shape(img)[0]

    #get regions of interest (ROI) of each vial, may have to tinker
    gold5_roi = getVialRegion("vial5Au", size)
    gold1_roi = getVialRegion("vial1Au", size)
    gadolinium5_roi = getVialRegion("vial5Gd", size)
    gadolinium1_roi = getVialRegion("vial1Gd", size)
    iodine5_roi = getVialRegion("vial5I", size)
    iodine1_roi = getVialRegion("vial1I", size)

    #vials_total = gold1_roi+gold5_roi+gadolinium1_roi+gadolinium5_roi+iodine1_roi+iodine5_roi

    #converting to Hounsfeld units
    noise_roi = getBackgroundRegion(size)
    number_of_noise_pixels = np.sum(noise_roi)
    noise_fbp_image = noise_roi * img
    background_noise = np.sum(noise_fbp_image) / number_of_noise_pixels
    ct_image = 1000 * ((img / background_noise) - 1)

    plot2DImage(ct_image,extent=(-radius,radius,-radius,radius),colourmap=plt.get_cmap("Greys_r"),label_x="x [cm]",
                label_y="y [cm]",  imkwargs={"vmin": -1000, "vmax": 2000},
                saveFile=dir_images+energy_bin+"_CTimage_slice"+str(slice_no)) #extent=(-radius,radius,-radius,radius)

    #save all slice numbers to find optimal K-edge image
    for i in range(1,24):
        image = data['Reconimg'][:,:,i]
        #np.save(energy_bin+"_CTimage_slice"+str(i)+".npy",image)

    #calculate cnr
    cnr_gold1, err_gold1, cnr_gold5, err_gold5 = calculateCNR(img, gold1_roi, gold5_roi, noise_roi)
    #print("Energy bin: ", energy_bin)
    #print("CNR of 1% Au vial: ", cnr_gold1, " +/- ", err_gold1)
    #print( "CNR of 5% Au vial: ", cnr_gold5, " +/- ", err_gold5)

    cnr_gadolinium1, err_gadolinium1, cnr_gadolinium5, err_gadolinium5 = calculateCNR(img, gadolinium1_roi,
                                                                             gadolinium5_roi, noise_roi)
    #print( "Energy bin: ", energy_bin)
    #print( "CNR of 1% Gd vial: ", cnr_gadolinium1, " +/- ", err_gadolinium1)
    #print( "CNR of 5% Gd vial: ", cnr_gadolinium5, " +/- ", err_gadolinium5)

    cnr_iodine1, err_iodine1, cnr_iodine5, err_iodine5 = calculateCNR(img, iodine1_roi,
                                                                      iodine5_roi, noise_roi)
    #print( "Energy bin: ", energy_bin)
    #print( "CNR of 1% I vial: ", cnr_iodine1, " +/- ", err_iodine1)
    #print( "CNR of 5% I vial: ", cnr_iodine5, " +/- ", err_iodine5)

    #saving CNR data to be loaded later
    gold_data = np.array([cnr_gold1, err_gold1, cnr_gold5, err_gold5])
    gadolinium_data = np.array([cnr_gadolinium1, err_gadolinium1, cnr_gadolinium5, err_gadolinium5])
    iodine_data = np.array([cnr_iodine1, err_iodine1, cnr_iodine5, err_iodine5])
    cnr_data = np.array([gold_data,gadolinium_data,iodine_data])

    np.save(directory+"CNRdata_"+energy_bin+"_multiplexslice_corrected2.npy",cnr_data)
    
def main_Kedge_subtracted_1(dir_data):
    '''
    After running main() method, do the K-edge subtraction
    :return:
    '''
    slice_no = 15
    m = 16

    radius = 1.593
    directory = "./images"
    #dir_data = "./data_no_smoothing/"
    
    #for i in range(15,16):
    i = 15
        

    img0 = sio.loadmat(dir_data + "binSEC0_multiplex_corrected2.mat")
    img1 = sio.loadmat(dir_data + "binSEC1_multiplex_corrected2.mat")
    img2 = sio.loadmat(dir_data + "binSEC2_multiplex_corrected2.mat")
    img3 = sio.loadmat(dir_data + "binSEC3_multiplex_corrected2.mat")
    img4 = sio.loadmat(dir_data + "binSEC4_multiplex_corrected2.mat")
    img5 = sio.loadmat(dir_data + "binSEC5_multiplex_corrected2.mat")

    #import ipdb; ipdb.set_trace() 

    img_I = np.average(img1['Reconimg'][:,:,slice_no:m],2) - np.average(img0['Reconimg'][:,:,slice_no:m],2)
    img_Gd = np.average(img3['Reconimg'][:,:,slice_no:m],2) - np.average(img2['Reconimg'][:,:,slice_no:m],2)
    img_Au = np.average(img5['Reconimg'][:,:,slice_no:m],2) - np.average(img4['Reconimg'][:,:,slice_no:m],2)
    
    img_Au -= np.amin(img_Au)
    img_Au /= np.amax(img_Au)
    
    scipy.misc.toimage(
        img_Au, cmin=0.0, cmax=1.0).save(
        '0.png')

    size = np.shape(img_I)[0] #same size as all 3 here
    noise_roi = getBackgroundRegion(size)
    number_of_noise_pixels = np.sum(noise_roi)
    noise_fbp_image = noise_roi * img_I
    background_noise = np.sum(noise_fbp_image) / number_of_noise_pixels
    ct_image_I = 1000 * ((img_I / background_noise) - 1)

    noise_fbp_image = noise_roi * img_Gd
    background_noise = np.sum(noise_fbp_image) / number_of_noise_pixels
    ct_image_Gd = 1000 * ((img_Gd / background_noise) - 1)

    noise_fbp_image = noise_roi * img_Au
    background_noise = np.sum(noise_fbp_image) / number_of_noise_pixels
    ct_image_Au = 1000 * ((img_Au / background_noise) - 1)
    
    plot_as_rgb(img_Au,img_Gd,img_I)

    plot2DImage(img_I, extent=(-radius,radius,-radius,radius), colourmap=plt.get_cmap("Purples"), label_x="x [cm]",
                label_y="y [cm]",  saveFile=directory + "/SEC-K_I/HU/Kedge_I_CTimage_slice"+str(i), plot=True,
                imkwargs={"vmin": None, "vmax": None})
    plot2DImage(img_Gd, extent=(-radius,radius,-radius,radius), colourmap=plt.get_cmap("Greens"), label_x="x [cm]",
                label_y="y [cm]",  saveFile=directory + "/SEC-K_Gd/HU/Kedge_Gd_CTimage_slice" + str(i), plot=True,
                imkwargs={"vmin": None, "vmax": None})
    plot2DImage(img_Au, extent=(-radius,radius,-radius,radius), colourmap=plt.get_cmap("YlOrBr"), label_x="x [cm]",
                label_y="y [cm]",  saveFile=directory + "/SEC-K_Au/HU/Kedge_Au_CTimage_slice" + str(i), plot=True,
                imkwargs={"vmin": None, "vmax": None})

    gold5_roi = getVialRegion("vial5Au", size)
    gold1_roi = getVialRegion("vial1Au", size)
    gadolinium5_roi = getVialRegion("vial5Gd", size)
    gadolinium1_roi = getVialRegion("vial1Gd", size)
    iodine5_roi = getVialRegion("vial5I", size)
    iodine1_roi = getVialRegion("vial1I", size)
    noise_roi = getBackgroundRegion(size)
    


    # calculate cnr
    cnr_gold1, err_gold1, cnr_gold5, err_gold5 = calculateCNR(img_Au, gold1_roi, gold5_roi, noise_roi)
    #print("Energy bin: K-edge SEC5 - SEC4")
    #print("CNR of 1% Au vial: ", cnr_gold1, " +/- ", err_gold1)
    #print("CNR of 5% Au vial: ", cnr_gold5, " +/- ", err_gold5)

    cnr_gadolinium1, err_gadolinium1, cnr_gadolinium5, err_gadolinium5 = calculateCNR(img_Gd, gadolinium1_roi,
                                                                                      gadolinium5_roi, noise_roi)
    #print( "Energy bin: K-edge SEC3 - SEC2")
    #print( "CNR of 1% Gd vial: ", cnr_gadolinium1, " +/- ", err_gadolinium1)
    #print( "CNR of 5% Gd vial: ", cnr_gadolinium5, " +/- ", err_gadolinium5)

    cnr_iodine1, err_iodine1, cnr_iodine5, err_iodine5 = calculateCNR(img_I, iodine1_roi,
                                                                      iodine5_roi, noise_roi)
    #print( "Energy bin: K-edge SEC1 - SEC0")
    #print( "CNR of 1% I vial: ", cnr_iodine1, " +/- ", err_iodine1)
    #print( "CNR of 5% I vial: ", cnr_iodine5, " +/- ", err_iodine5)

    gold_data = np.array([cnr_gold1, err_gold1, cnr_gold5, err_gold5])
    gadolinium_data = np.array([cnr_gadolinium1, err_gadolinium1, cnr_gadolinium5, err_gadolinium5])
    iodine_data = np.array([cnr_iodine1, err_iodine1, cnr_iodine5, err_iodine5])
    cnr_data = np.array([gold_data, gadolinium_data, iodine_data])

    np.save(dir_data + "CNRdata_Kedge_SECmultiplex_corrected2.npy", cnr_data)
    

    
def main_Kedge_subtracted_2(dir_data):
    '''
    After running main() method, do the K-edge subtraction
    :return:
    '''
    slice_no = 15
    m = 16

    radius = 1.593
    directory = "./images"
    #dir_data = "./data_no_smoothing/"
    
    #for i in range(15,16):
    i = 15
        

    img0 = sio.loadmat(dir_data + "binSEC0_multiplex_corrected2.mat")['Reconimg'][:,:,slice_no]
    img1 = sio.loadmat(dir_data + "binSEC1_multiplex_corrected2.mat")['Reconimg'][:,:,slice_no]
    img2 = sio.loadmat(dir_data + "binSEC2_multiplex_corrected2.mat")['Reconimg'][:,:,slice_no]
    img3 = sio.loadmat(dir_data + "binSEC3_multiplex_corrected2.mat")['Reconimg'][:,:,slice_no]
    img4 = sio.loadmat(dir_data + "binSEC4_multiplex_corrected2.mat")['Reconimg'][:,:,slice_no]
    img5 = sio.loadmat(dir_data + "binSEC5_multiplex_corrected2.mat")['Reconimg'][:,:,slice_no]

    #import ipdb; ipdb.set_trace() 

    size = np.shape(img0)[0] #same size as all 3 here
    
    noise_roi = getBackgroundRegion(size)
    
    mask_background0 = img0*noise_roi
    mask_background1 = img1*noise_roi
    mask_background2 = img2*noise_roi
    mask_background3 = img3*noise_roi
    mask_background4 = img4*noise_roi
    mask_background5 = img5*noise_roi
    

    signal_background0= np.average(mask_background0[mask_background0 != 0])
    signal_background1= np.average(mask_background1[mask_background1 != 0])
    signal_background2= np.average(mask_background2[mask_background2 != 0])
    signal_background3= np.average(mask_background3[mask_background3 != 0])
    signal_background4= np.average(mask_background4[mask_background4 != 0])
    signal_background5= np.average(mask_background5[mask_background5 != 0])
    
    img0 = img0 / signal_background0
    img1 = img1 / signal_background1
    img2 = img2 / signal_background2
    img3 = img3 / signal_background3
    img4 = img4 / signal_background4
    img5 = img5 / signal_background5
   
   # img0[img0 < signal_background0 - 0.05] = 0
   # img1[img1 < signal_background1 - 0.05] = 0
   # img2[img2 < signal_background2 - 0.05] = 0
   # img3[img3 < signal_background3 - 0.05] = 0
   # img4[img4 < signal_background4 - 0.05] = 0
   # img5[img5 < signal_background5 - 0.05] = 0
    
    plt.figure(77)
    plt.imshow(img0)
    plt.figure(78)
    plt.imshow(img1)

    img_I = (img1 - img0)/    (img1 + img0) 
    img_Gd = (img3  - img2)/  (img3 + img2)
    img_Au = (img5  - img4)/  (img5 + img4)
    
    img_I[img_I < -0.003] = 0
    img_Gd[img_Gd < -0.001] = 0
    img_Au[img_Au < 0.0019] = 0
    img_I[img_I > 0.1] = 0
    img_Gd[img_Gd > 0.12] = 0
    img_Au[img_Au > 0.07] = 0
    #size = np.shape(img_I)[0] #same size as all 3 here
    #noise_roi = getBackgroundRegion(size)
    #number_of_noise_pixels = np.sum(noise_roi)
    #noise_fbp_image = noise_roi * img_I
    #background_noise = np.sum(noise_fbp_image) / number_of_noise_pixels
    #ct_image_I = 1000 * ((img_I - background_noise) / (background_noise - air_noise))

    #noise_fbp_image = noise_roi * img_Gd
    #background_noise = np.sum(noise_fbp_image) / number_of_noise_pixels
    #ct_image_Gd = 1000 * ((img_Gd / background_noise) - 1)

    #noise_fbp_image = noise_roi * img_Au
    #background_noise = np.sum(noise_fbp_image) / number_of_noise_pixels
    #ct_image_Au = 1000 * ((img_Au / background_noise) - 1)
    

    plot2DImage(img_I, extent=(-radius,radius,-radius,radius), colourmap=plt.get_cmap("Purples"), label_x="x [cm]",
                label_y="y [cm]",  saveFile=directory + "/SEC-K_I/HU/Kedge_I_CTimage_slice"+str(i), plot=True,
                imkwargs={"vmin": None, "vmax": None})
    plot2DImage(img_Gd, extent=(-radius,radius,-radius,radius), colourmap=plt.get_cmap("Greens"), label_x="x [cm]",
                label_y="y [cm]",  saveFile=directory + "/SEC-K_Gd/HU/Kedge_Gd_CTimage_slice" + str(i), plot=True,
                imkwargs={"vmin": None, "vmax": None})
    plot2DImage(img_Au, extent=(-radius,radius,-radius,radius), colourmap=plt.get_cmap("YlOrBr"), label_x="x [cm]",
                label_y="y [cm]",  saveFile=directory + "/SEC-K_Au/HU/Kedge_Au_CTimage_slice" + str(i), plot=True,
                imkwargs={"vmin": None, "vmax": None})

    plt.figure()
    plot_as_rgb(img_Au,img_Gd,img_I)
    
    plt.figure()
    gold5_roi = getVialRegion("vial5Au", size)
    gold1_roi = getVialRegion("vial1Au", size)
    gadolinium5_roi = getVialRegion("vial5Gd", size)
    gadolinium1_roi = getVialRegion("vial1Gd", size)
    iodine5_roi = getVialRegion("vial5I", size)
    iodine1_roi = getVialRegion("vial1I", size)
    noise_roi = getBackgroundRegion(size)
    


    # calculate cnr
    cnr_gold1, err_gold1, cnr_gold5, err_gold5 = calculateCNR(img_Au, gold1_roi, gold5_roi, noise_roi)
    #print("Energy bin: K-edge SEC5 - SEC4")
    #print("CNR of 1% Au vial: ", cnr_gold1, " +/- ", err_gold1)
    #print("CNR of 5% Au vial: ", cnr_gold5, " +/- ", err_gold5)

    cnr_gadolinium1, err_gadolinium1, cnr_gadolinium5, err_gadolinium5 = calculateCNR(img_Gd, gadolinium1_roi,
                                                                                      gadolinium5_roi, noise_roi)
    #print( "Energy bin: K-edge SEC3 - SEC2")
    #print( "CNR of 1% Gd vial: ", cnr_gadolinium1, " +/- ", err_gadolinium1)
    #print( "CNR of 5% Gd vial: ", cnr_gadolinium5, " +/- ", err_gadolinium5)

    cnr_iodine1, err_iodine1, cnr_iodine5, err_iodine5 = calculateCNR(img_I, iodine1_roi,
                                                                      iodine5_roi, noise_roi)
    #print( "Energy bin: K-edge SEC1 - SEC0")
    #print( "CNR of 1% I vial: ", cnr_iodine1, " +/- ", err_iodine1)
    #print( "CNR of 5% I vial: ", cnr_iodine5, " +/- ", err_iodine5)

    gold_data = np.array([cnr_gold1, err_gold1, cnr_gold5, err_gold5])
    gadolinium_data = np.array([cnr_gadolinium1, err_gadolinium1, cnr_gadolinium5, err_gadolinium5])
    iodine_data = np.array([cnr_iodine1, err_iodine1, cnr_iodine5, err_iodine5])
    cnr_data = np.array([gold_data, gadolinium_data, iodine_data])

    np.save(dir_data + "CNRdata_Kedge_SECmultiplex_corrected2.npy", cnr_data)

def main_Kedge_subtracted_HU(dir_data):
    '''
    After running main() method, do the K-edge subtraction
    :return:
    '''
    slice_no = 15
    m = 16

    radius = 1.593
    directory = "./images"
    #dir_data = "./data_no_smoothing/"
    
    #for i in range(15,16):
    i = 15
        

    img0 = sio.loadmat(dir_data + "binSEC0_multiplex_corrected2.mat")['Reconimg'][:,:,slice_no]
    img1 = sio.loadmat(dir_data + "binSEC1_multiplex_corrected2.mat")['Reconimg'][:,:,slice_no]
    img2 = sio.loadmat(dir_data + "binSEC2_multiplex_corrected2.mat")['Reconimg'][:,:,slice_no]
    img3 = sio.loadmat(dir_data + "binSEC3_multiplex_corrected2.mat")['Reconimg'][:,:,slice_no]
    img4 = sio.loadmat(dir_data + "binSEC4_multiplex_corrected2.mat")['Reconimg'][:,:,slice_no]
    img5 = sio.loadmat(dir_data + "binSEC5_multiplex_corrected2.mat")['Reconimg'][:,:,slice_no]
    img6 = sio.loadmat(dir_data + "binSEC6_multiplex_corrected2.mat")['Reconimg'][:,:,slice_no]
    #import ipdb; ipdb.set_trace() 

    size = np.shape(img0)[0] #same size as all 3 here
    
    noise_roi = getBackgroundRegion(size)
    
    #mask_background0 = img0*noise_roi
   # mask_background1 = img1*noise_roi
   # mask_background2 = img2*noise_roi
   # mask_background3 = img3*noise_roi
   # mask_background4 = img4*noise_roi
   # mask_background5 = img5*noise_roi
    

   # signal_background0= np.average(mask_background0[mask_background0 != 0])
   # signal_background1= np.average(mask_background1[mask_background1 != 0])
   # signal_background2= np.average(mask_background2[mask_background2 != 0])
   # signal_background3= np.average(mask_background3[mask_background3 != 0])
   # signal_background4= np.average(mask_background4[mask_background4 != 0])
   # signal_background5= np.average(mask_background5[mask_background5 != 0])
    
   # img0 = img0 / signal_background0
   # img1 = img1 / signal_background1
   # img2 = img2 / signal_background2
   # img3 = img3 / signal_background3
   # img4 = img4 / signal_background4
   # img5 = img5 / signal_background5
   
   # img0[img0 < signal_background0 - 0.05] = 0
   # img1[img1 < signal_background1 - 0.05] = 0
   # img2[img2 < signal_background2 - 0.05] = 0
   # img3[img3 < signal_background3 - 0.05] = 0
   # img4[img4 < signal_background4 - 0.05] = 0
   # img5[img5 < signal_background5 - 0.05] = 0
    '''
    plt.figure(77)
    plt.imshow(img0)
    plt.figure(78)
    plt.imshow(img1)
    '''
    img_I = (img1 - img0)
    img_Gd = (img3  - img2)
    img_Au = (img5  - img4)
    
   # img_I[img_I < -0.003] = 0
   # img_Gd[img_Gd < -0.001] = 0
   # img_Au[img_Au < 0.0019] = 0
   # img_I[img_I > 0.1] = 0
   # img_Gd[img_Gd > 0.12] = 0
   # img_Au[img_Au > 0.07] = 0
   # #size = np.shape(img_I)[0] #same size as all 3 here
    
    noise_roi = getBackgroundRegion(size)
    air_roi = getAirRegion(size)
    
    number_of_noise_pixels = np.sum(noise_roi)
    number_of_air_pixels = np.sum(air_roi)
    
    noise_fbp_image = noise_roi * img0
    noise_air_image = air_roi * img0
    background_noise0 = np.sum(noise_fbp_image) / number_of_noise_pixels
    air_noise0 = np.sum(noise_air_image) / number_of_air_pixels
    ct_image0 = 1000 * ((img0 - background_noise0)/(background_noise0 - air_noise0))

    noise_fbp_image = noise_roi * img1
    noise_air_image = air_roi * img1
    background_noise1 = np.sum(noise_fbp_image) / number_of_noise_pixels
    air_noise1 = np.sum(noise_air_image) / number_of_air_pixels
    ct_image1 = 1000 * ((img1  - background_noise1)/(background_noise1 - air_noise1))

    noise_fbp_image = noise_roi * img2
    noise_air_image = air_roi * img2  
    background_noise2 = np.sum(noise_fbp_image) / number_of_noise_pixels
    air_noise2 = np.sum(noise_air_image) / number_of_air_pixels
    ct_image2 = 1000 * ((img2 - background_noise2)/(background_noise2 - air_noise2)) 
    
    noise_fbp_image = noise_roi * img3
    noise_air_image = air_roi * img3
    background_noise3 = np.sum(noise_fbp_image) / number_of_noise_pixels
    air_noise3 = np.sum(noise_air_image) / number_of_air_pixels
    ct_image3 = 1000 * ((img3 - background_noise3)/(background_noise3 - air_noise3))
   
    noise_fbp_image = noise_roi * img4
    noise_air_image = air_roi * img4  
    background_noise4 = np.sum(noise_fbp_image) / number_of_noise_pixels
    air_noise4 = np.sum(noise_air_image) / number_of_air_pixels
    ct_image4 = 1000 * ((img4 - background_noise4)/(background_noise4 - air_noise4))

    noise_fbp_image = noise_roi * img5
    noise_air_image = air_roi * img5  
    background_noise5 = np.sum(noise_fbp_image) / number_of_noise_pixels
    air_noise5 = np.sum(noise_air_image) / number_of_air_pixels
    ct_image5 = 1000 * ((img5 - background_noise5)/(background_noise5 - air_noise5))
    
    noise_fbp_image = noise_roi * img6
    noise_air_image = air_roi * img6  
    background_noise6 = np.sum(noise_fbp_image) / number_of_noise_pixels
    air_noise6 = np.sum(noise_air_image) / number_of_air_pixels
    ct_image6 = 1000 * ((img6 - background_noise6)/(background_noise6 - air_noise6))
    
    ct_image_I = (ct_image1 - ct_image0)
    ct_image_Gd = (ct_image3  - ct_image2)
    ct_image_Au = (ct_image5  - ct_image4)
    
    gold5_roi = getVialRegion("vial5Au", size)
    gold1_roi = getVialRegion("vial1Au", size)
    gadolinium5_roi = getVialRegion("vial5Gd", size)
    gadolinium1_roi = getVialRegion("vial1Gd", size)
    iodine5_roi = getVialRegion("vial5I", size)
    iodine1_roi = getVialRegion("vial1I", size)
    noise_roi = getBackgroundRegion(size)
    
    np_Gd = np.sum(gadolinium5_roi)
    np_Au = np.sum(gold5_roi)
    np_I = np.sum(iodine5_roi)
    
    av_Gd = np.sum(gadolinium5_roi*ct_image_Gd) /np_Gd 
    av_Au = np.sum(gold5_roi*ct_image_Au)       /np_Au
    av_I  = np.sum(iodine5_roi*ct_image_I)      /np_I
    
    ct_image_I /= av_I
    ct_image_Au /= av_Au
    ct_image_Gd /= av_Gd
    
    ct_image_I *= 5
    ct_image_Au *= 5
    ct_image_Gd *= 5
    
    TEXT_SIZE = 8
    
    fig, ax = plt.subplots(nrows=2,ncols=2)
    
    im = ax[1,1].imshow(ct_image6, extent=(-radius,radius,-radius,radius), cmap=plt.get_cmap("gray"), vmin = -1000)
    
    cbar = fig.colorbar(im, ax=ax[1,1])
    cbar.set_label("HU", rotation=270,labelpad=TEXT_SIZE,size=TEXT_SIZE)
    #ax[1,1].set_xticks(size= TEXT_SIZE)
    #ax[1,1].set_yticks(size = TEXT_SIZE)
    ax[1,1].set_xlabel("x [cm]",size=TEXT_SIZE,labelpad=2)
    ax[1,1].set_ylabel("y [cm]",size=TEXT_SIZE,labelpad=1)
    ax[1,1].set_xticks(ticks = [-1.5, 0, 1.5])
    ax[1,1].set_yticks(ticks = [-1.5, 0, 1.5])
    ax[1,1].set_title('d) CT image',size=TEXT_SIZE)
    
    im = ax[0,0].imshow(ct_image_I, extent=(-radius,radius,-radius,radius), cmap=purp, vmin = 0, vmax = 5)
    
    cbar = fig.colorbar(im, ax=ax[0,0])
    cbar.set_label("% I", rotation=270,labelpad=TEXT_SIZE,size=TEXT_SIZE)
    #ax[0,0].set_xticks(size= TEXT_SIZE)
    #ax[0,0].set_yticks(size = TEXT_SIZE)
    ax[0,0].set_xlabel("x [cm]",size=TEXT_SIZE,labelpad=2)
    ax[0,0].set_ylabel("y [cm]",size=TEXT_SIZE,labelpad=1)
    ax[0,0].set_xticks(ticks = [-1.5, 0, 1.5])
    ax[0,0].set_yticks(ticks = [-1.5, 0, 1.5])
    ax[0,0].set_title('a) Iodine',size=TEXT_SIZE)
    
    im = ax[0,1].imshow(ct_image_Gd, extent=(-radius,radius,-radius,radius), cmap=green, vmin = 0, vmax = 5)
    
    cbar = fig.colorbar(im, ax=ax[0,1])
    cbar.set_label("% Gd", rotation=270,labelpad=TEXT_SIZE,size=TEXT_SIZE)
    #ax[0,0].set_xticks(size= TEXT_SIZE)
    #ax[0,0].set_yticks(size = TEXT_SIZE)
    ax[0,1].set_xlabel("x [cm]",size=TEXT_SIZE,labelpad=2)
    ax[0,1].set_ylabel("y [cm]",size=TEXT_SIZE,labelpad=1)
    ax[0,1].set_xticks(ticks = [-1.5, 0, 1.5])
    ax[0,1].set_yticks(ticks = [-1.5, 0, 1.5])
    ax[0,1].set_title('b) Gadolinium',size=TEXT_SIZE)
    
    im = ax[1,0].imshow(ct_image_Au, extent=(-radius,radius,-radius,radius), cmap=gold, vmin = 0, vmax = 5)
    
    cbar = fig.colorbar(im, ax=ax[1,0])
    cbar.set_label("% Au", rotation=270,labelpad=TEXT_SIZE,size=TEXT_SIZE)
    #ax[0,0].set_xticks(size= TEXT_SIZE)
    #ax[0,0].set_yticks(size = TEXT_SIZE)
    ax[1,0].set_xlabel("x [cm]",size=TEXT_SIZE,labelpad=2)
    ax[1,0].set_ylabel("y [cm]",size=TEXT_SIZE,labelpad=1)
    ax[1,0].set_xticks(ticks = [-1.5, 0, 1.5])
    ax[1,0].set_yticks(ticks = [-1.5, 0, 1.5])
    ax[1,0].set_title('c) Gold',size=TEXT_SIZE)
    
    fig.tight_layout()
    
    plt.savefig('4_layout2.png',dpi = 800)
    
    '''
    subplot2DImage(ct_image_I, extent=(-radius,radius,-radius,radius), colourmap=plt.get_cmap("Purples"), label_x="x [cm]",
                label_y="y [cm]",label_cb = "% Au",  saveFile=directory + "/SEC-K_I/HU/Kedge_I_CTimage_slice"+str(i), plot=True,
                imkwargs={"vmin": 0, "vmax": 5})

    subplot2DImage(ct_image_Gd, extent=(-radius,radius,-radius,radius), colourmap=plt.get_cmap("Greens"), label_x="x [cm]",
                label_y="y [cm]", label_cb = "% Gd", saveFile=directory + "/SEC-K_Gd/HU/Kedge_Gd_CTimage_slice" + str(i), plot=True,
                imkwargs={"vmin": 0, "vmax": 5})

    subplot2DImage(ct_image_Au, extent=(-radius,radius,-radius,radius), colourmap=plt.get_cmap("YlOrBr"), label_x="x [cm]",
                label_y="y [cm]",label_cb = "% I",  saveFile=directory + "/SEC-K_Au/HU/Kedge_Au_CTimage_slice" + str(i), plot=True,
                imkwargs={"vmin": 0, "vmax": 5})
    '''
    
    ct_image_Au[ct_image_Au < 0] = 0
    ct_image_I[ct_image_I < 0] = 0
    ct_image_Gd[ct_image_Gd < 0] = 0

    ct_image_Au[ct_image_Au > 300] = 300
    ct_image_I[ct_image_I > 300] = 300
    ct_image_Gd[ct_image_Gd > 300] = 300
 
    #plt.figure()
    #plot_as_rgb2(ct_image_Au,ct_image_Gd,ct_image_I,img6)
    
    #plt.figure()
    


    # calculate cnr
    cnr_gold1, err_gold1, cnr_gold5, err_gold5 = calculateCNR(img_Au, gold1_roi, gold5_roi, noise_roi)
    #print("Energy bin: K-edge SEC5 - SEC4")
    #print("CNR of 1% Au vial: ", cnr_gold1, " +/- ", err_gold1)
    #print("CNR of 5% Au vial: ", cnr_gold5, " +/- ", err_gold5)

    cnr_gadolinium1, err_gadolinium1, cnr_gadolinium5, err_gadolinium5 = calculateCNR(img_Gd, gadolinium1_roi,
                                                                                      gadolinium5_roi, noise_roi)
    #print( "Energy bin: K-edge SEC3 - SEC2")
    #print( "CNR of 1% Gd vial: ", cnr_gadolinium1, " +/- ", err_gadolinium1)
    #print( "CNR of 5% Gd vial: ", cnr_gadolinium5, " +/- ", err_gadolinium5)

    cnr_iodine1, err_iodine1, cnr_iodine5, err_iodine5 = calculateCNR(img_I, iodine1_roi,
                                                                      iodine5_roi, noise_roi)
    #print( "Energy bin: K-edge SEC1 - SEC0")
    #print( "CNR of 1% I vial: ", cnr_iodine1, " +/- ", err_iodine1)
    #print( "CNR of 5% I vial: ", cnr_iodine5, " +/- ", err_iodine5)

    gold_data = np.array([cnr_gold1, err_gold1, cnr_gold5, err_gold5])
    gadolinium_data = np.array([cnr_gadolinium1, err_gadolinium1, cnr_gadolinium5, err_gadolinium5])
    iodine_data = np.array([cnr_iodine1, err_iodine1, cnr_iodine5, err_iodine5])
    cnr_data = np.array([gold_data, gadolinium_data, iodine_data])

    #np.save(dir_data + "CNRdata_Kedge_SECmultiplex_corrected2.npy", cnr_data)

def main_Kedge_subtracted_HU_norm(dir_data):
    '''
    After running main() method, do the K-edge subtraction
    :return:
    '''
    slice_no = 15
    m = 16
    
    i5 = []
    gd5 = []
    au5 = []
    
    i1 = []
    gd1 = []
    au1 = []
    
    i0 = []
    gd0 = []
    au0 = []

    radius = 1.593
    directory = "./images"
    #dir_data = "./data_no_smoothing/"
    
    #for i in range(15,16):
    i = 15
    
    #plt.style.use('dark_background')
    fig, ax = plt.subplots(nrows=3,ncols=1,figsize=(10,10), dpi=300)
    
    ax[0] = plt.subplot(2, 2, 1)
    ax[1] = plt.subplot(2, 2, 2)
    ax[2] = plt.subplot(2, 2, 3)

    
    SIZE = 8
    width = 0.1
    indices = [0] 
    
    for slice_no in range(4,20):

        img0 = sio.loadmat(dir_data + "binSEC0_multiplex_corrected2.mat")['Reconimg'][:,:,slice_no]
        img1 = sio.loadmat(dir_data + "binSEC1_multiplex_corrected2.mat")['Reconimg'][:,:,slice_no]
        img2 = sio.loadmat(dir_data + "binSEC2_multiplex_corrected2.mat")['Reconimg'][:,:,slice_no]
        img3 = sio.loadmat(dir_data + "binSEC3_multiplex_corrected2.mat")['Reconimg'][:,:,slice_no]
        img4 = sio.loadmat(dir_data + "binSEC4_multiplex_corrected2.mat")['Reconimg'][:,:,slice_no]
        img5 = sio.loadmat(dir_data + "binSEC5_multiplex_corrected2.mat")['Reconimg'][:,:,slice_no]
        img6 = sio.loadmat(dir_data + "binSEC6_multiplex_corrected2.mat")['Reconimg'][:,:,slice_no]
        #import ipdb; ipdb.set_trace() 

        size = np.shape(img0)[0] #same size as all 3 here

        noise_roi = getBackgroundRegion(size)

        '''
        plt.figure(77)
        plt.imshow(img0)
        plt.figure(78)
        plt.imshow(img1)
        '''
        img_I = (img1 - img0)
        img_Gd = (img3  - img2)
        img_Au = (img5  - img4)

       # img_I[img_I < -0.003] = 0
       # img_Gd[img_Gd < -0.001] = 0
       # img_Au[img_Au < 0.0019] = 0
       # img_I[img_I > 0.1] = 0
       # img_Gd[img_Gd > 0.12] = 0
       # img_Au[img_Au > 0.07] = 0
       # #size = np.shape(img_I)[0] #same size as all 3 here

        noise_roi = getBackgroundRegion(size)
        air_roi = getAirRegion(size)

        number_of_noise_pixels = np.sum(noise_roi)
        number_of_air_pixels = np.sum(air_roi)

        noise_fbp_image = noise_roi * img0
        noise_air_image = air_roi * img0
        background_noise0 = np.sum(noise_fbp_image) / number_of_noise_pixels
        air_noise0 = np.sum(noise_air_image) / number_of_air_pixels
        ct_image0 = 1000 * ((img0 - background_noise0)/(background_noise0 - air_noise0))

        noise_fbp_image = noise_roi * img1
        noise_air_image = air_roi * img1
        background_noise1 = np.sum(noise_fbp_image) / number_of_noise_pixels
        air_noise1 = np.sum(noise_air_image) / number_of_air_pixels
        ct_image1 = 1000 * ((img1  - background_noise1)/(background_noise1 - air_noise1))

        noise_fbp_image = noise_roi * img2
        noise_air_image = air_roi * img2  
        background_noise2 = np.sum(noise_fbp_image) / number_of_noise_pixels
        air_noise2 = np.sum(noise_air_image) / number_of_air_pixels
        ct_image2 = 1000 * ((img2 - background_noise2)/(background_noise2 - air_noise2)) 

        noise_fbp_image = noise_roi * img3
        noise_air_image = air_roi * img3
        background_noise3 = np.sum(noise_fbp_image) / number_of_noise_pixels
        air_noise3 = np.sum(noise_air_image) / number_of_air_pixels
        ct_image3 = 1000 * ((img3 - background_noise3)/(background_noise3 - air_noise3))

        noise_fbp_image = noise_roi * img4
        noise_air_image = air_roi * img4  
        background_noise4 = np.sum(noise_fbp_image) / number_of_noise_pixels
        air_noise4 = np.sum(noise_air_image) / number_of_air_pixels
        ct_image4 = 1000 * ((img4 - background_noise4)/(background_noise4 - air_noise4))

        noise_fbp_image = noise_roi * img5
        noise_air_image = air_roi * img5  
        background_noise5 = np.sum(noise_fbp_image) / number_of_noise_pixels
        air_noise5 = np.sum(noise_air_image) / number_of_air_pixels
        ct_image5 = 1000 * ((img5 - background_noise5)/(background_noise5 - air_noise5))

        noise_fbp_image = noise_roi * img6
        noise_air_image = air_roi * img6  
        background_noise6 = np.sum(noise_fbp_image) / number_of_noise_pixels
        air_noise6 = np.sum(noise_air_image) / number_of_air_pixels
        ct_image6 = 1000 * ((img6 - background_noise6)/(background_noise6 - air_noise6))

        ct_image_I = (ct_image1 - ct_image0) #/(ct_image1 + ct_image0)
        ct_image_Gd = (ct_image3  - ct_image2) #/(ct_image3  + ct_image2)
        ct_image_Au = (ct_image5  - ct_image4) #/(ct_image5  + ct_image4)

        #all_image = (ct_image_Gd+ct_image_Au)/2
        #import ipdb; ipdb.set_trace()

        #ct_image_I = ct_image_I / all_image
        #ct_image_Au = ct_image_Au / all_image
        #ct_image_Gd = ct_image_Gd / all_image

        #ct_image = np.dstack((ct_image_Gd,ct_image_Au,ct_image_I,np.ones(ct_image_I.shape)*50))
        #that3 = np.argmax(ct_image,axis=2)

        ct_image = np.dstack((ct_image_Gd,ct_image_Au,ct_image_I))
        that3 = np.argmax(ct_image,axis=2)
        that4 = np.argmin(ct_image,axis=2)
        difference = np.std(ct_image,axis=2) # - np.min(ct_image,axis=2)
        '''
        plot2DImage(that3, extent=(-radius,radius,-radius,radius), colourmap=plt.get_cmap("Greys_r"), label_x="x [cm]",
                    label_y="y [cm]",label_cb = "HU",  saveFile=directory + "/SEC-K_I/HU/Kedge_I_CTimage_slice"+str(i), plot=True,
                    imkwargs={"vmin": None, "vmax": None}) 
        plot2DImage(difference, extent=(-radius,radius,-radius,radius), colourmap=plt.get_cmap("Purples"), label_x="x [cm]",
                    label_y="y [cm]",label_cb = "HU",  saveFile=directory + "/SEC-K_I/HU/Kedge_I_CTimage_slice"+str(i), plot=True,
                    imkwargs={"vmin": None, "vmax": None})
        plot2DImage(ct_image_Gd, extent=(-radius,radius,-radius,radius), colourmap=plt.get_cmap("Greens"), label_x="x [cm]",
                    label_y="y [cm]", label_cb = "HU", saveFile=directory + "/SEC-K_Gd/HU/Kedge_Gd_CTimage_slice" + str(i), plot=True,
                    imkwargs={"vmin": None, "vmax": None})
        plot2DImage(ct_image_Au, extent=(-radius,radius,-radius,radius), colourmap=plt.get_cmap("YlOrBr"), label_x="x [cm]",
                    label_y="y [cm]",label_cb = "HU",  saveFile=directory + "/SEC-K_Au/HU/Kedge_Au_CTimage_slice" + str(i), plot=True,
                    imkwargs={"vmin": None, "vmax": None})
        '''
        gold5_roi = getVialRegion("vial5Au", size)
        gold1_roi = getVialRegion("vial1Au", size)
        gadolinium5_roi = getVialRegion("vial5Gd", size)
        gadolinium1_roi = getVialRegion("vial1Gd", size)
        iodine5_roi = getVialRegion("vial5I", size)
        iodine1_roi = getVialRegion("vial1I", size)
        noise_roi = getBackgroundRegion2(size)


        np_Gd = np.sum(gadolinium5_roi)
        np_Au = np.sum(gold5_roi)
        np_I = np.sum(iodine5_roi)

        np_Gd1 = np.sum(gadolinium1_roi)
        np_Au1 = np.sum(gold1_roi)
        np_I1 = np.sum(iodine1_roi)
        np_n = np.sum(noise_roi)

        av_Gd = np.sum(gadolinium5_roi*ct_image_Gd) /np_Gd 
        av_Au = np.sum(gold5_roi*ct_image_Au)       /np_Au
        av_I  = np.sum(iodine5_roi*ct_image_I)      /np_I

        ct_image_I /= av_I
        ct_image_Au /= av_Au
        ct_image_Gd /= av_Gd

        ct_image_I *= 5
        ct_image_Au *= 5
        ct_image_Gd *= 5

        av_Gd = np.sum(gadolinium5_roi*ct_image_Gd) /np_Gd 
        av_Au = np.sum(gold5_roi*ct_image_Au)       /np_Au
        av_I  = np.sum(iodine5_roi*ct_image_I)      /np_I    

        std_Gd = np.std(ct_image_Gd[gadolinium5_roi != 0]) 
        std_Au = np.std(ct_image_Au[gold5_roi != 0])
        std_I  = np.std(ct_image_I[iodine5_roi != 0])   

        av_Gd1 = np.sum(gadolinium1_roi*ct_image_Gd) /np_Gd1 
        av_Au1 = np.sum(gold1_roi*ct_image_Au)       /np_Au1
        av_I1  = np.sum(iodine1_roi*ct_image_I)      /np_I1    

        std_Gd1 = np.std(ct_image_Gd[gadolinium1_roi != 0]) 
        std_Au1 = np.std(ct_image_Au[gold1_roi != 0])
        std_I1  = np.std(ct_image_I[iodine1_roi != 0]) 

        av_Gd0 = np.sum(noise_roi*ct_image_Gd) /np_n 
        av_Au0 = np.sum(noise_roi*ct_image_Au) /np_n
        av_I0  = np.sum(noise_roi*ct_image_I)  /np_n

        std_Gd0 = np.std(ct_image_Gd[noise_roi != 0])
        std_I0 = np.std(ct_image_I[noise_roi != 0])
        std_Au0 = np.std(ct_image_Au[noise_roi != 0]) 
        #import ipdb; ipdb.set_trace()
        '''
        fig, ax = plt.subplots(nrows=2,ncols=2,figsize=(10,5), dpi=300)

        ax[0,0] = plt.subplot(2, 2, 1)
        ax[0,1] = plt.subplot(2, 2, 2)
        ax[1,0] = plt.subplot(2, 2, 3)
        ax[1,1] = plt.subplot(2, 2, 4)

        SIZE = 8
        width = 0.8
        indices = [0]
        ax[0,0].bar(indices, av_I, width=width,
                color='#4616DC', label='5% I', yerr=std_I, ecolor="k")
        ax[0,0].bar([i + 0.25 * 0.4 * width for i in indices], av_I1,
                width=0.8 * width, color='#A666EE', label='1% I', yerr=std_I1, ecolor="k")
        ax[0,0].bar([i + 0.25 * 0.4 * width for i in indices], av_I0,
                width=0.8 * width, color='#A666EE', label='1% I', yerr=std_I1, ecolor="k")
        ax[0,0].set_ylabel("%I", size=SIZE)
        ax[0,0].set_title("Iodine Concentrations")
        #plt.savefig(dir_images+"CNR_multiplex_iodine_SEC.png")
        #ax[0,0].show()
        #indices = [1]
        ax[0,1].bar(indices, av_Gd, width=width,
                color='#147A00', label='5% Gd', yerr=std_Gd, ecolor="k")
        ax[0,1].bar([i + 0.25 * 0.4 * width for i in indices], av_Gd1,
                width=0.8 * width, color='#37D710', label='1% Gd', yerr=std_Gd1, ecolor="k")
        ax[0,1].set_ylabel("%Gd", size=SIZE)
        ax[0,1].set_title("Gadolinium Concentrations")

        ax[1,0].bar(indices, av_Au, width=width,
                color='#CB870F', label='5% Au', yerr=std_Au, ecolor="k")
        ax[1,0].bar([i + 0.25 * 0.4 * width for i in indices], av_Au1,
                width=0.8 * width, color='#F0E102', label='1% Au', yerr=std_Au1, ecolor="k")
        ax[1,0].set_xticks([])
        ax[0,0].set_xticks([])
        ax[0,1].set_xticks([])

        ax[1,0].set_ylabel("%Au", size=SIZE)
        ax[1,0].legend(loc=0, fontsize=SIZE)
        ax[0,0].legend(loc=0, fontsize=SIZE)
        ax[0,1].legend(loc=0, fontsize=SIZE)
        ax[1,0].set_title("Gold Concentrations")
        #ax[1,0].show()
        fig.tight_layout()


        fig, ax = plt.subplots(nrows=2,ncols=2,figsize=(10,5), dpi=300)

        ax[0,0] = plt.subplot(1, 1, 1)


        SIZE = 8
        width = 0.8
        indices = [0]
        ax[0,0].bar(indices, av_I, width=width,
                color='#4616DC', label='5% I', ecolor="k")
        indices[0] += 1
        ax[0,0].bar([i + 0.25 * 0.4 * width for i in indices], av_I1,
                width=0.8 * width, color='#A666EE', label='1% I', ecolor="k")
        indices[0] += 1
        ax[0,0].bar([i + 0.25 * 0.3* width for i in indices], av_I0,
                width=0.6 * width, color='#EE82EE', label='0% I', ecolor="k")
        indices[0] += 1
        indices[0] += 1


        ax[0,0].set_ylabel("% Contrast Agent", size=SIZE)
        ax[0,0].set_title("Concentrations")
        #plt.savefig(dir_images+"CNR_multiplex_iodine_SEC.png")
        #ax[0,0].show(
        ax[0,0].bar(indices, av_Gd, width=width,
                color='#147A00', label='5% Gd', ecolor="k")
        indices[0] += 1
        ax[0,0].bar(indices, av_Gd1,
                width=0.8 * width, color='#37D710', label='1% Gd', ecolor="k")
        indices[0] += 1
        ax[0,0].bar(indices, av_Gd0,
                width=0.6 * width, color='#7CFC00', label='0% Gd', ecolor="k")
        indices[0] += 1
        indices[0] += 1

        ax[0,0].bar(indices, av_Au, width=width,
                color='#CB870F', label='5% Au', ecolor="k")
        indices[0] += 1
        ax[0,0].bar(indices, av_Au1,
                width=0.8 * width, color='#F0E102', label='1% Au', ecolor="k")
        indices[0] += 1
        ax[0,0].bar(indices, av_Au0,
                width=0.6 * width, color='#BDB76B', label='0% Au', ecolor="k")

        ax[0,0].grid(True)
        ax[0,0].set_ylabel('ROI')
        ax[0,0].set_xticks([0,1,2,4,5,6,8,9,10])
        ax[0,0].set_xticklabels(['5% I','1% I','0% I','5% Gd','1% Gd','0% Gd','5% Au','1% Au','0% Au'])
        #ax[0,0].legend(loc=0, fontsize=SIZE)


        #ax[1,0].show()
        fig.tight_layout()
        plt.savefig('concentrations.png')
        '''
        indices2 = 0 + (slice_no - 11.5)*0.03
        msize = 6
        ax[0].scatter(5 + indices2, av_I,marker='_',
                color='darkgray',s=msize)
        indices[0] += 1
        ax[1].scatter(5+ indices2, av_Gd,
                color='darkgray',marker='_',s=msize)
        indices[0] += 1
        ax[2].scatter(5+ indices2, av_Au,
                color='darkgray',marker='_',s=msize)
        indices[0] += 1
        indices[0] += 1

        ax[0].scatter(1+ indices2, av_I1, color='darkgray',marker='_',s=msize)
        indices[0] += 1
        ax[1].scatter(2+ indices2, av_Gd1, color='darkgray',marker='_',s=msize)
        indices[0] += 1
        ax[2].scatter(1+ indices2, av_Au1, color='darkgray',marker='_',s=msize)
        indices[0] += 1
        indices[0] += 1


        ax[0].scatter(0+ indices2, av_I0, color='darkgray',marker='_',s=msize)
        indices[0] += 1
        ax[1].scatter(0+ indices2, av_Gd0, color='darkgray',marker='_',s=msize)
        indices[0] += 1
        ax[2].scatter(0+ indices2, av_Au0, color='darkgray',marker='_',s=msize)
        
        i5.append(av_I)
        gd5.append(av_Gd)
        au5.append(av_Au)

        i1.append(av_I1)
        gd1.append(av_Gd1)
        au1.append(av_Au1)

        i0.append(av_I0)
        gd0.append(av_Gd0)
        au0.append(av_Au0)
        
        '''
        ax[0,0].grid(True)
        ax[0,0].set_ylabel('ROI')
        ax[0,0].set_xticks([0,1,2,4,5,6,8,9,10])
        ax[0,0].set_xticklabels(['5% I','5% Gd','5% Au','1% I','2% Gd','1% Au','0% I','0% Gd','0% Au'])
        '''
    
    arr_img = plt.imread('5%au.png', format='png')

    imagebox = OffsetImage(arr_img, zoom=0.1)
    imagebox.image.axes = ax[2]
    offset = 1.7

    ab = AnnotationBbox(imagebox, (5,0),
                        xybox=(0, -7),
                        xycoords=("data", "axes fraction"),
                        boxcoords="offset points",
                        box_alignment=(.5, offset),
                        bboxprops={"edgecolor" : "k",
                        "facecolor":'k'})

    ax[2].add_artist(ab)
    
    arr_img = plt.imread('1%au.png', format='png')

    imagebox = OffsetImage(arr_img, zoom=0.1)
    imagebox.image.axes = ax[2]

    ab = AnnotationBbox(imagebox, (1,0),
                        xybox=(0, -7),
                        xycoords=("data", "axes fraction"),
                        boxcoords="offset points",
                        box_alignment=(.5, offset),
                        bboxprops={"edgecolor" : "k",
                        "facecolor":'k'})

    ax[2].add_artist(ab)
    
    arr_img = plt.imread('0%au.png', format='png')

    imagebox = OffsetImage(arr_img, zoom=0.1)
    imagebox.image.axes = ax[2]

    ab = AnnotationBbox(imagebox, (0,0),
                        xybox=(0, -7),
                        xycoords=("data", "axes fraction"),
                        boxcoords="offset points",
                        box_alignment=(.5, offset),
                        bboxprops={"edgecolor" : "k",
                        "facecolor":'k'})

    ax[2].add_artist(ab)
    
    arr_img = plt.imread('5%i.png', format='png')

    imagebox = OffsetImage(arr_img, zoom=0.1)
    imagebox.image.axes = ax[0]

    ab = AnnotationBbox(imagebox, (5,0),
                        xybox=(0, -7),
                        xycoords=("data", "axes fraction"),
                        boxcoords="offset points",
                        box_alignment=(.5, offset),
                        bboxprops={"edgecolor" : "k",
                        "facecolor":'k'})

    ax[0].add_artist(ab)
    
    arr_img = plt.imread('1%i.png', format='png')

    imagebox = OffsetImage(arr_img, zoom=0.1)
    imagebox.image.axes = ax[0]

    ab = AnnotationBbox(imagebox, (1,0),
                        xybox=(0, -7),
                        xycoords=("data", "axes fraction"),
                        boxcoords="offset points",
                        box_alignment=(.5, offset),
                        bboxprops={"edgecolor" : "k",
                        "facecolor":'k'})

    ax[0].add_artist(ab)
    
    arr_img = plt.imread('0%i.png', format='png')

    imagebox = OffsetImage(arr_img, zoom=0.1)
    imagebox.image.axes = ax[0]

    ab = AnnotationBbox(imagebox, (0,0),
                        xybox=(0, -7),
                        xycoords=("data", "axes fraction"),
                        boxcoords="offset points",
                        box_alignment=(.5, offset),
                        bboxprops={"edgecolor" : "k",
                        "facecolor":'k'})

    ax[0].add_artist(ab)
    arr_img = plt.imread('5%gd.png', format='png')

    imagebox = OffsetImage(arr_img, zoom=0.1)
    imagebox.image.axes = ax[1]

    ab = AnnotationBbox(imagebox, (5,0),
                        xybox=(0, -7),
                        xycoords=("data", "axes fraction"),
                        boxcoords="offset points",
                        box_alignment=(.5, offset),
                        bboxprops={"edgecolor" : "k",
                        "facecolor":'k'})

    ax[1].add_artist(ab)
    
    arr_img = plt.imread('2%gd.png', format='png')

    imagebox = OffsetImage(arr_img, zoom=0.1)
    imagebox.image.axes = ax[1]

    ab = AnnotationBbox(imagebox, (2,0),
                        xybox=(0, -7),
                        xycoords=("data", "axes fraction"),
                        boxcoords="offset points",
                        box_alignment=(.5, offset),
                        bboxprops={"edgecolor" : "k",
                        "facecolor":'k'})

    ax[1].add_artist(ab)
    
    arr_img = plt.imread('0%gd.png', format='png')

    imagebox = OffsetImage(arr_img, zoom=0.1)
    imagebox.image.axes = ax[1]

    ab = AnnotationBbox(imagebox, (0,0),
                        xybox=(0, -7),
                        xycoords=("data", "axes fraction"),
                        boxcoords="offset points",
                        box_alignment=(.5, offset),
                        bboxprops={"edgecolor" : "k",
                        "facecolor":'k'})

    ax[1].add_artist(ab)
    
    ax[0].errorbar(5, np.mean(i5), yerr = np.std(i5), marker='x',
                  color=(1.0,0,1.0))
    indices[0] += 1
    ax[1].errorbar(5, np.mean(gd5), yerr = np.std(gd5),
                  color=(0,1.0,0),marker='x',)
    indices[0] += 1
    ax[2].errorbar(5, np.mean(au5), yerr = np.std(au5),
                  color=(1.0,0.85,0),marker='x',)
    indices[0] += 1
    indices[0] += 1

    ax[0].errorbar(1, np.mean(i1), yerr = np.std(i1), color=(1.0,0,1.0),marker='x')
    indices[0] += 1
    ax[1].errorbar(2, np.mean(gd1), yerr = np.std(gd1), color=(0,1.0,0),marker='x')
    indices[0] += 1
    ax[2].errorbar(1, np.mean(au1), yerr = np.std(au1), color=(1.0,0.85,0),marker='x')
    indices[0] += 1
    indices[0] += 1


    ax[0].errorbar(0, np.mean(i0), yerr = np.std(i0), color=(1.0,0,1.0),marker='x')
    indices[0] += 1
    ax[1].errorbar(0, np.mean(gd0), yerr = np.std(gd0), color=(0,1.0,0),marker='x')
    indices[0] += 1
    ax[2].errorbar(0, np.mean(au0), yerr = np.std(au0), color=(1.0,0.85,0),marker='x')  
    
    x = [0,1,5]
    y = [np.mean(i0),np.mean(i1),np.mean(i5)]
    ax[0].plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)),'--', color=(1.0,0,1.0),linewidth=0.5,label='Linear fit')
    
    x = [0,2,5]
    y = [np.mean(gd0),np.mean(gd1),np.mean(gd5)]
    ax[1].plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)),'--g',linewidth=0.5,label='Linear fit')
    
    x = [0,1,5]
    y = [np.mean(au0),np.mean(au1),np.mean(au5)]
    ax[2].plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)),'--', color=(1.0,0.85,0),linewidth=0.5,label='Linear fit')

    x = [0,1,5]
    ax[0].plot(x, x,'--', color='k',linewidth=0.5,label='Theoretical')
    
    x = [0,2,5]
    ax[1].plot(x, x,'--', color='k',linewidth=0.5,label='Theoretical')
    
    x = [0,1,5]
    ax[2].plot(x, x,'--', color='k',linewidth=0.5,label='Theoretical')
    #fig.patch.set_facecolor('k')

    ax[2].set_title('Gold')
    ax[0].set_title('Iodine')
    ax[1].set_title('Gadolinium')
    
    ax[2].set_ylabel('%Au')
    ax[0].set_ylabel('%I')
    ax[1].set_ylabel('%Gd')
    
    SIZE =8
    ax[2].set_xlabel('ROI',fontsize=SIZE)
    ax[0].set_xlabel('ROI',fontsize=SIZE)
    ax[1].set_xlabel('ROI',fontsize=SIZE)
    ax[0].grid(True)
    ax[1].grid(True)
    ax[2].grid(True)
    #ax[0,0].legend(loc=0, fontsize=SIZE)
    
    ax[0].set_xticks([0,1,5])
    ax[0].set_xticklabels(['0%','1%','5%'])
    ax[1].set_xticks([0,2,5])
    ax[1].set_xticklabels(['0%','2%','5%'])    
    ax[2].set_xticks([0,1,5])
    ax[2].set_xticklabels(['0%','1%','5%'])
    
    ax[0].set_yticks([0,1,5])
    ax[1].set_yticks([0,2,5])   
    ax[2].set_yticks([0,1,5])
    
    ax[0].legend(loc=4)
    ax[1].legend(loc=4)              
    ax[2].legend(loc=4)             

    
    #fig.tight_layout()
    plt.savefig('concentrations.png')
    
    
    ct_image_Au[ct_image_Au < 0] = 0
    ct_image_I[ct_image_I < 0] = 0
    ct_image_Gd[ct_image_Gd < 0] = 0

    ct_image_Au[ct_image_Au > 5] = 5
    ct_image_I[ct_image_I > 5] = 5
    ct_image_Gd[ct_image_Gd > 5] = 5
    
    im_1 = np.zeros(that3.shape)
    im_2 = np.zeros(that3.shape)
    im_3 = np.zeros(that3.shape)
     
    im_1[that3 == 0] = 1
    im_2[that3 == 1] = 1
    im_3[that3 == 2] = 1
    
    

    '''
    #plt.figure()
    plot_as_rgb2(im_2*difference,im_1*difference,im_3*difference,im_2,im_1,im_3,img6)
    
    #plt.figure()
    

    
    # calculate cnr
    cnr_gold1, err_gold1, cnr_gold5, err_gold5 = calculateCNR(img_Au, gold1_roi, gold5_roi, noise_roi)
    print("Energy bin: K-edge SEC5 - SEC4")
    print("CNR of 1% Au vial: ", cnr_gold1, " +/- ", err_gold1)
    print("CNR of 5% Au vial: ", cnr_gold5, " +/- ", err_gold5)

    cnr_gadolinium1, err_gadolinium1, cnr_gadolinium5, err_gadolinium5 = calculateCNR(img_Gd, gadolinium1_roi,
                                                                                      gadolinium5_roi, noise_roi)
    print( "Energy bin: K-edge SEC3 - SEC2")
    print( "CNR of 1% Gd vial: ", cnr_gadolinium1, " +/- ", err_gadolinium1)
    print( "CNR of 5% Gd vial: ", cnr_gadolinium5, " +/- ", err_gadolinium5)

    cnr_iodine1, err_iodine1, cnr_iodine5, err_iodine5 = calculateCNR(img_I, iodine1_roi,
                                                                      iodine5_roi, noise_roi)
    print( "Energy bin: K-edge SEC1 - SEC0")
    print( "CNR of 1% I vial: ", cnr_iodine1, " +/- ", err_iodine1)
    print( "CNR of 5% I vial: ", cnr_iodine5, " +/- ", err_iodine5)

    gold_data = np.array([cnr_gold1, err_gold1, cnr_gold5, err_gold5])
    gadolinium_data = np.array([cnr_gadolinium1, err_gadolinium1, cnr_gadolinium5, err_gadolinium5])
    iodine_data = np.array([cnr_iodine1, err_iodine1, cnr_iodine5, err_iodine5])
    cnr_data = np.array([gold_data, gadolinium_data, iodine_data])

    np.save(dir_data + "CNRdata_Kedge_SECmultiplex_corrected2.npy", cnr_data)
    '''
    
def main_Kedge_subtracted_HU_plot(dir_data):
    '''
    After running main() method, do the K-edge subtraction
    :return:
    '''
    slice_no = 15
    m = 16

    radius = 1.593
    directory = "./images"
    #dir_data = "./data_no_smoothing/"
    
    #for i in range(15,16):
    i = 15
    
    fig, ax = plt.subplots(nrows=2,ncols=2,figsize=(10,5), dpi=300)
    
    ax[0,0] = plt.subplot(1, 1, 1)

    
    SIZE = 8
    width = 0.1
    indices = [0] 
    '''

    img0 = np.mean(sio.loadmat(dir_data + "binSEC0_multiplex_corrected2.mat")['Reconimg'][:,:,slice_no:slice_no + 1],axis = 2)
    img1 = np.mean(sio.loadmat(dir_data + "binSEC1_multiplex_corrected2.mat")['Reconimg'][:,:,slice_no:slice_no + 1],axis = 2)
    img2 = np.mean(sio.loadmat(dir_data + "binSEC2_multiplex_corrected2.mat")['Reconimg'][:,:,slice_no:slice_no + 1],axis = 2)
    img3 = np.mean(sio.loadmat(dir_data + "binSEC3_multiplex_corrected2.mat")['Reconimg'][:,:,slice_no:slice_no + 1],axis = 2)
    img4 = np.mean(sio.loadmat(dir_data + "binSEC4_multiplex_corrected2.mat")['Reconimg'][:,:,slice_no:slice_no + 1],axis = 2)
    img5 = np.mean(sio.loadmat(dir_data + "binSEC5_multiplex_corrected2.mat")['Reconimg'][:,:,slice_no:slice_no + 1],axis = 2)
    img6 = np.mean(sio.loadmat(dir_data + "binSEC6_multiplex_corrected2.mat")['Reconimg'][:,:,slice_no:slice_no + 1],axis = 2)
    '''
    img0 = sio.loadmat(dir_data + "binSEC0_multiplex_corrected2.mat")['Reconimg'][:,:,slice_no]
    img1 = sio.loadmat(dir_data + "binSEC1_multiplex_corrected2.mat")['Reconimg'][:,:,slice_no]
    img2 = sio.loadmat(dir_data + "binSEC2_multiplex_corrected2.mat")['Reconimg'][:,:,slice_no]
    img3 = sio.loadmat(dir_data + "binSEC3_multiplex_corrected2.mat")['Reconimg'][:,:,slice_no]
    img4 = sio.loadmat(dir_data + "binSEC4_multiplex_corrected2.mat")['Reconimg'][:,:,slice_no]
    img5 = sio.loadmat(dir_data + "binSEC5_multiplex_corrected2.mat")['Reconimg'][:,:,slice_no]
    img6 = sio.loadmat(dir_data + "binSEC6_multiplex_corrected2.mat")['Reconimg'][:,:,slice_no]
    #import ipdb; ipdb.set_trace() 
    
    size = np.shape(img0)[0] #same size as all 3 here

    noise_roi = getBackgroundRegion(size)

    '''
        plt.figure(77)
        plt.imshow(img0)
        plt.figure(78)
        plt.imshow(img1)
        '''
    img_I = (img1 - img0)
    img_Gd = (img3  - img2)
    img_Au = (img5  - img4)

    # img_I[img_I < -0.003] = 0
    # img_Gd[img_Gd < -0.001] = 0
    # img_Au[img_Au < 0.0019] = 0
    # img_I[img_I > 0.1] = 0
    # img_Gd[img_Gd > 0.12] = 0
    # img_Au[img_Au > 0.07] = 0
    # #size = np.shape(img_I)[0] #same size as all 3 here

    noise_roi = getBackgroundRegion(size)
    air_roi = getAirRegion(size)

    number_of_noise_pixels = np.sum(noise_roi)
    number_of_air_pixels = np.sum(air_roi)

    noise_fbp_image = noise_roi * img0
    noise_air_image = air_roi * img0
    background_noise0 = np.sum(noise_fbp_image) / number_of_noise_pixels
    air_noise0 = np.sum(noise_air_image) / number_of_air_pixels
    ct_image0 = 1000 * ((img0 - background_noise0)/(background_noise0 - air_noise0))

    noise_fbp_image = noise_roi * img1
    noise_air_image = air_roi * img1
    background_noise1 = np.sum(noise_fbp_image) / number_of_noise_pixels
    air_noise1 = np.sum(noise_air_image) / number_of_air_pixels
    ct_image1 = 1000 * ((img1  - background_noise1)/(background_noise1 - air_noise1))

    noise_fbp_image = noise_roi * img2
    noise_air_image = air_roi * img2  
    background_noise2 = np.sum(noise_fbp_image) / number_of_noise_pixels
    air_noise2 = np.sum(noise_air_image) / number_of_air_pixels
    ct_image2 = 1000 * ((img2 - background_noise2)/(background_noise2 - air_noise2)) 

    noise_fbp_image = noise_roi * img3
    noise_air_image = air_roi * img3
    background_noise3 = np.sum(noise_fbp_image) / number_of_noise_pixels
    air_noise3 = np.sum(noise_air_image) / number_of_air_pixels
    ct_image3 = 1000 * ((img3 - background_noise3)/(background_noise3 - air_noise3))

    noise_fbp_image = noise_roi * img4
    noise_air_image = air_roi * img4  
    background_noise4 = np.sum(noise_fbp_image) / number_of_noise_pixels
    air_noise4 = np.sum(noise_air_image) / number_of_air_pixels
    ct_image4 = 1000 * ((img4 - background_noise4)/(background_noise4 - air_noise4))

    noise_fbp_image = noise_roi * img5
    noise_air_image = air_roi * img5  
    background_noise5 = np.sum(noise_fbp_image) / number_of_noise_pixels
    air_noise5 = np.sum(noise_air_image) / number_of_air_pixels
    ct_image5 = 1000 * ((img5 - background_noise5)/(background_noise5 - air_noise5))

    noise_fbp_image = noise_roi * img6
    noise_air_image = air_roi * img6  
    background_noise6 = np.sum(noise_fbp_image) / number_of_noise_pixels
    air_noise6 = np.sum(noise_air_image) / number_of_air_pixels
    ct_image6 = 1000 * ((img6 - background_noise6)/(background_noise6 - air_noise6))

    ct_image_I = (ct_image1 - ct_image0) #/(ct_image1 + ct_image0)
    ct_image_Gd = (ct_image3  - ct_image2) #/(ct_image3  + ct_image2)
    ct_image_Au = (ct_image5  - ct_image4) #/(ct_image5  + ct_image4)

    #all_image = (ct_image_Gd+ct_image_Au)/2
    #import ipdb; ipdb.set_trace()

    #ct_image_I = ct_image_I / all_image
    #ct_image_Au = ct_image_Au / all_image
    #ct_image_Gd = ct_image_Gd / all_image

    #ct_image = np.dstack((ct_image_Gd,ct_image_Au,ct_image_I,np.ones(ct_image_I.shape)*50))
    #that3 = np.argmax(ct_image,axis=2)

    ct_image = np.dstack((ct_image_Gd,ct_image_Au,ct_image_I))
    that3 = np.argmax(ct_image,axis=2)
    that4 = np.argmin(ct_image,axis=2)
    difference = np.std(ct_image,axis=2) # - np.min(ct_image,axis=2)
    '''
        plot2DImage(that3, extent=(-radius,radius,-radius,radius), colourmap=plt.get_cmap("Greys_r"), label_x="x [cm]",
                    label_y="y [cm]",label_cb = "HU",  saveFile=directory + "/SEC-K_I/HU/Kedge_I_CTimage_slice"+str(i), plot=True,
                    imkwargs={"vmin": None, "vmax": None}) 
        plot2DImage(difference, extent=(-radius,radius,-radius,radius), colourmap=plt.get_cmap("Purples"), label_x="x [cm]",
                    label_y="y [cm]",label_cb = "HU",  saveFile=directory + "/SEC-K_I/HU/Kedge_I_CTimage_slice"+str(i), plot=True,
                    imkwargs={"vmin": None, "vmax": None})
        plot2DImage(ct_image_Gd, extent=(-radius,radius,-radius,radius), colourmap=plt.get_cmap("Greens"), label_x="x [cm]",
                    label_y="y [cm]", label_cb = "HU", saveFile=directory + "/SEC-K_Gd/HU/Kedge_Gd_CTimage_slice" + str(i), plot=True,
                    imkwargs={"vmin": None, "vmax": None})
        plot2DImage(ct_image_Au, extent=(-radius,radius,-radius,radius), colourmap=plt.get_cmap("YlOrBr"), label_x="x [cm]",
                    label_y="y [cm]",label_cb = "HU",  saveFile=directory + "/SEC-K_Au/HU/Kedge_Au_CTimage_slice" + str(i), plot=True,
                    imkwargs={"vmin": None, "vmax": None})
        '''
    gold5_roi = getVialRegion("vial5Au", size)
    gold1_roi = getVialRegion("vial1Au", size)
    gadolinium5_roi = getVialRegion("vial5Gd", size)
    gadolinium1_roi = getVialRegion("vial1Gd", size)
    iodine5_roi = getVialRegion("vial5I", size)
    iodine1_roi = getVialRegion("vial1I", size)
    noise_roi = getBackgroundRegion2(size)

    mask_all = (gold5_roi+gold1_roi+gadolinium5_roi+gadolinium1_roi+iodine5_roi+ iodine1_roi)
    mask_all[mask_all == 1] = -1
    mask_all[mask_all == 0] = 10
    mask_all[mask_all == -1] = 0
    plt.figure(1)
    plt.imshow(ct_image6*mask_all)
    plt.show()
    np_Gd = np.sum(gadolinium5_roi)
    np_Au = np.sum(gold5_roi)
    np_I = np.sum(iodine5_roi)

    np_Gd1 = np.sum(gadolinium1_roi)
    np_Au1 = np.sum(gold1_roi)
    np_I1 = np.sum(iodine1_roi)
    np_n = np.sum(noise_roi)

    av_Gd = np.sum(gadolinium5_roi*ct_image_Gd) /np_Gd 
    av_Au = np.sum(gold5_roi*ct_image_Au)       /np_Au
    av_I  = np.sum(iodine5_roi*ct_image_I)      /np_I

    ct_image_I /= av_I
    ct_image_Au /= av_Au
    ct_image_Gd /= av_Gd

    ct_image_I *= 5
    ct_image_Au *= 5
    ct_image_Gd *= 5

    av_Gd = np.sum(gadolinium5_roi*ct_image_Gd) /np_Gd 
    av_Au = np.sum(gold5_roi*ct_image_Au)       /np_Au
    av_I  = np.sum(iodine5_roi*ct_image_I)      /np_I    

    std_Gd = np.std(ct_image_Gd[gadolinium5_roi != 0]) 
    std_Au = np.std(ct_image_Au[gold5_roi != 0])
    std_I  = np.std(ct_image_I[iodine5_roi != 0])   

    av_Gd1 = np.sum(gadolinium1_roi*ct_image_Gd) /np_Gd1 
    av_Au1 = np.sum(gold1_roi*ct_image_Au)       /np_Au1
    av_I1  = np.sum(iodine1_roi*ct_image_I)      /np_I1    

    std_Gd1 = np.std(ct_image_Gd[gadolinium1_roi != 0]) 
    std_Au1 = np.std(ct_image_Au[gold1_roi != 0])
    std_I1  = np.std(ct_image_I[iodine1_roi != 0]) 

    av_Gd0 = np.sum(noise_roi*ct_image_Gd) /np_n 
    av_Au0 = np.sum(noise_roi*ct_image_Au) /np_n
    av_I0  = np.sum(noise_roi*ct_image_I)  /np_n

    std_Gd0 = np.std(ct_image_Gd[noise_roi != 0])
    std_I0 = np.std(ct_image_I[noise_roi != 0])
    std_Au0 = np.std(ct_image_Au[noise_roi != 0]) 
    #import ipdb; ipdb.set_trace()
    '''
        fig, ax = plt.subplots(nrows=2,ncols=2,figsize=(10,5), dpi=300)

        ax[0,0] = plt.subplot(2, 2, 1)
        ax[0,1] = plt.subplot(2, 2, 2)
        ax[1,0] = plt.subplot(2, 2, 3)
        ax[1,1] = plt.subplot(2, 2, 4)

        SIZE = 8
        width = 0.8
        indices = [0]
        ax[0,0].bar(indices, av_I, width=width,
                color='#4616DC', label='5% I', yerr=std_I, ecolor="k")
        ax[0,0].bar([i + 0.25 * 0.4 * width for i in indices], av_I1,
                width=0.8 * width, color='#A666EE', label='1% I', yerr=std_I1, ecolor="k")
        ax[0,0].bar([i + 0.25 * 0.4 * width for i in indices], av_I0,
                width=0.8 * width, color='#A666EE', label='1% I', yerr=std_I1, ecolor="k")
        ax[0,0].set_ylabel("%I", size=SIZE)
        ax[0,0].set_title("Iodine Concentrations")
        #plt.savefig(dir_images+"CNR_multiplex_iodine_SEC.png")
        #ax[0,0].show()
        #indices = [1]
        ax[0,1].bar(indices, av_Gd, width=width,
                color='#147A00', label='5% Gd', yerr=std_Gd, ecolor="k")
        ax[0,1].bar([i + 0.25 * 0.4 * width for i in indices], av_Gd1,
                width=0.8 * width, color='#37D710', label='1% Gd', yerr=std_Gd1, ecolor="k")
        ax[0,1].set_ylabel("%Gd", size=SIZE)
        ax[0,1].set_title("Gadolinium Concentrations")

        ax[1,0].bar(indices, av_Au, width=width,
                color='#CB870F', label='5% Au', yerr=std_Au, ecolor="k")
        ax[1,0].bar([i + 0.25 * 0.4 * width for i in indices], av_Au1,
                width=0.8 * width, color='#F0E102', label='1% Au', yerr=std_Au1, ecolor="k")
        ax[1,0].set_xticks([])
        ax[0,0].set_xticks([])
        ax[0,1].set_xticks([])

        ax[1,0].set_ylabel("%Au", size=SIZE)
        ax[1,0].legend(loc=0, fontsize=SIZE)
        ax[0,0].legend(loc=0, fontsize=SIZE)
        ax[0,1].legend(loc=0, fontsize=SIZE)
        ax[1,0].set_title("Gold Concentrations")
        #ax[1,0].show()
        fig.tight_layout()

    '''
    fig, ax = plt.subplots(nrows=2,ncols=2,figsize=(10,5), dpi=300)

    ax[0,0] = plt.subplot(1, 1, 1)

    '''
    SIZE = 8
    width = 0.8
    indices = [0]
    ax[0,0].bar(indices, av_I, width=width,
                color='#4616DC', label='5% I',yerr=std_I, ecolor="k")
    indices[0] += 1
    ax[0,0].bar([i + 0.25 * 0.4 * width for i in indices], av_I1,
                width=0.8 * width, color='#A666EE',yerr=std_I1, label='1% I', ecolor="k")
    indices[0] += 1
    ax[0,0].bar([i + 0.25 * 0.3* width for i in indices], av_I0,
                width=0.6 * width, color='#EE82EE',yerr=std_I0, label='0% I', ecolor="k")
    indices[0] += 1
    indices[0] += 1

    
    ax[0,0].set_ylabel("% Contrast Agent", size=SIZE)
    ax[0,0].set_title("Concentrations")
    #plt.savefig(dir_images+"CNR_multiplex_iodine_SEC.png")
    #ax[0,0].show(
    ax[0,0].bar(indices, av_Gd, width=width,
                color='#147A00', label='5% Gd',yerr=std_Gd, ecolor="k")
    indices[0] += 1
    ax[0,0].bar(indices, av_Gd1,
                width=0.8 * width, color='#37D710',yerr=std_Gd1, label='1% Gd', ecolor="k")
    indices[0] += 1
    ax[0,0].bar(indices, av_Gd0,
                width=0.6 * width, color='#7CFC00',yerr=std_Gd0, label='0% Gd', ecolor="k")
    indices[0] += 1
    indices[0] += 1

    ax[0,0].bar(indices, av_Au, width=width,
                color='#CB870F', label='5% Au',yerr=std_Au, ecolor="k")
    indices[0] += 1
    ax[0,0].bar(indices, av_Au1,
                width=0.8 * width, color='#F0E102',yerr=std_Au1, label='1% Au', ecolor="k")
    indices[0] += 1
    ax[0,0].bar(indices, av_Au0,
                width=0.6 * width, color='#BDB76B',yerr=std_Au0, label='0% Au', ecolor="k")

    ax[0,0].grid(True)
    ax[0,0].set_ylabel('ROI')
    ax[0,0].set_xticks([0,1,2,4,5,6,8,9,10])
    ax[0,0].set_xticklabels(['5% I','1% I','0% I','5% Gd','1% Gd','0% Gd','5% Au','1% Au','0% Au'])
    
    #ax[0,0].legend(loc=0, fontsize=SIZE)

    '''
    
    SIZE = 8
    width = 1
    #ax[1,0].show()
    fig.tight_layout()
    plt.savefig('concentrations.png')


    ax[0,0].bar(indices, av_I, width=0.8 * width,
                color='#4616DC', label='5% I',yerr=std_I, ecolor="k")
    indices[0] += 1
    ax[0,0].bar(indices, av_Gd, width=0.8 * width,
                color='#147A00', label='5% Gd',yerr=std_Gd, ecolor="k")
    indices[0] += 1
    ax[0,0].bar(indices, av_Au, width=0.8 * width,
                color='#CB870F', label='5% Au',yerr=std_Au, ecolor="k")
    indices[0] += 1
    indices[0] += 1

    ax[0,0].bar([i + 0.25 * 0.4 * width for i in indices], av_I1,
                width=0.8 * width, color='#4616DC',yerr=std_I1, label='1% I', ecolor="k")
    indices[0] += 1
    ax[0,0].bar(indices, av_Gd1,
                width=0.8 * width, color='#147A00',yerr=std_Gd1, label='1% Gd', ecolor="k")
    indices[0] += 1
    ax[0,0].bar(indices, av_Au1,
                width=0.8 * width, color='#CB870F',yerr=std_Au1, label='1% Au', ecolor="k")
    indices[0] += 1
    indices[0] += 1


    ax[0,0].bar(indices, av_I0,
                width=0.8 * width, color='#4616DC',yerr=std_I0, label='0% I', ecolor="k")
    indices[0] += 1
    ax[0,0].bar(indices, av_Gd0,
                width=0.8 * width, color='#147A00',yerr=std_Gd0, label='0% Gd', ecolor="k")
    indices[0] += 1
    ax[0,0].bar(indices, av_Au0,
                width=0.8 * width, color='#CB870F',yerr=std_Au0, label='0% Au', ecolor="k")   

    ax[0,0].grid(True)
    ax[0,0].set_ylabel('ROI')
    ax[0,0].set_xticks([0,1,2,4,5,6,8,9,10])
    ax[0,0].set_xticklabels(['5% I','5% Gd','5% Au','1% I','1% Gd','1% Au','0% I','0% Gd','0% Au'])
    indices[0] = 0 + (slice_no - 7)*0.1
    
    plt.title('Concentrations slice {}'.format(slice_no))
    #ax[0,0].legend(loc=0, fontsize=SIZE)

    fig.tight_layout()
    plt.savefig('concentrations.png')
    '''
    TEXT_SIZE = 8
    
    fig, ax = plt.subplots(nrows=2,ncols=2)
    
    im = ax[1,1].imshow(ct_image6, extent=(-radius,radius,-radius,radius), cmap=plt.get_cmap("gray"), vmin = -1000)
    
    cbar = fig.colorbar(im, ax=ax[1,1])
    cbar.set_label("HU", rotation=270,labelpad=TEXT_SIZE,size=TEXT_SIZE)
    #ax[1,1].set_xticks(size= TEXT_SIZE)
    #ax[1,1].set_yticks(size = TEXT_SIZE)
    ax[1,1].set_xlabel("x [cm]",size=TEXT_SIZE,labelpad=2)
    ax[1,1].set_ylabel("y [cm]",size=TEXT_SIZE,labelpad=1)
    ax[1,1].set_xticks(ticks = [-1.5, 0, 1.5])
    ax[1,1].set_yticks(ticks = [-1.5, 0, 1.5])
    ax[1,1].set_title('d) CT image',size=TEXT_SIZE)
    
    im = ax[0,0].imshow(ct_image_I, extent=(-radius,radius,-radius,radius), cmap=purp, vmin = 0, vmax = 5)
    
    cbar = fig.colorbar(im, ax=ax[0,0])
    cbar.set_label("% I", rotation=270,labelpad=TEXT_SIZE,size=TEXT_SIZE)
    #ax[0,0].set_xticks(size= TEXT_SIZE)
    #ax[0,0].set_yticks(size = TEXT_SIZE)
    ax[0,0].set_xlabel("x [cm]",size=TEXT_SIZE,labelpad=2)
    ax[0,0].set_ylabel("y [cm]",size=TEXT_SIZE,labelpad=1)
    ax[0,0].set_xticks(ticks = [-1.5, 0, 1.5])
    ax[0,0].set_yticks(ticks = [-1.5, 0, 1.5])
    ax[0,0].set_title('a) Iodine',size=TEXT_SIZE)
    
    im = ax[0,1].imshow(ct_image_Gd, extent=(-radius,radius,-radius,radius), cmap=green, vmin = 0, vmax = 5)
    
    cbar = fig.colorbar(im, ax=ax[0,1])
    cbar.set_label("% Gd", rotation=270,labelpad=TEXT_SIZE,size=TEXT_SIZE)
    #ax[0,0].set_xticks(size= TEXT_SIZE)
    #ax[0,0].set_yticks(size = TEXT_SIZE)
    ax[0,1].set_xlabel("x [cm]",size=TEXT_SIZE,labelpad=2)
    ax[0,1].set_ylabel("y [cm]",size=TEXT_SIZE,labelpad=1)
    ax[0,1].set_xticks(ticks = [-1.5, 0, 1.5])
    ax[0,1].set_yticks(ticks = [-1.5, 0, 1.5])
    ax[0,1].set_title('b) Gadolinium',size=TEXT_SIZE)
    
    im = ax[1,0].imshow(ct_image_Au, extent=(-radius,radius,-radius,radius), cmap=gold, vmin = 0, vmax = 5)
    
    cbar = fig.colorbar(im, ax=ax[1,0])
    cbar.set_label("% Au", rotation=270,labelpad=TEXT_SIZE,size=TEXT_SIZE)
    #ax[0,0].set_xticks(size= TEXT_SIZE)
    #ax[0,0].set_yticks(size = TEXT_SIZE)
    ax[1,0].set_xlabel("x [cm]",size=TEXT_SIZE,labelpad=2)
    ax[1,0].set_ylabel("y [cm]",size=TEXT_SIZE,labelpad=1)
    ax[1,0].set_xticks(ticks = [-1.5, 0, 1.5])
    ax[1,0].set_yticks(ticks = [-1.5, 0, 1.5])
    ax[1,0].set_title('c) Gold',size=TEXT_SIZE)
    
    fig.tight_layout()
    
    plt.savefig('4_layout2.png',dpi = 800)
    
    '''
    ct_image_Au[ct_image_Au < 0] = 0
    ct_image_I[ct_image_I < 0] = 0
    ct_image_Gd[ct_image_Gd < 0] = 0

    ct_image_Au[ct_image_Au > 5] = 5
    ct_image_I[ct_image_I > 5] = 5
    ct_image_Gd[ct_image_Gd > 5] = 5
    
    im_1 = np.zeros(that3.shape)
    im_2 = np.zeros(that3.shape)
    im_3 = np.zeros(that3.shape)
     
    im_1[that3 == 0] = 1
    im_2[that3 == 1] = 1
    im_3[that3 == 2] = 1
    
    


    #plt.figure()
    plot_as_rgb2(im_2*difference,im_1*difference,im_3*difference,im_2,im_1,im_3,img6)
    
    #plt.figure()



    # calculate cnr
    cnr_gold1, err_gold1, cnr_gold5, err_gold5 = calculateCNR(ct_image_Au, gold1_roi, gold5_roi, noise_roi)
    #print("Energy bin: K-edge SEC5 - SEC4")
    #print("CNR of 1% Au vial: ", cnr_gold1, " +/- ", err_gold1)
    #print("CNR of 5% Au vial: ", cnr_gold5, " +/- ", err_gold5)

    cnr_gadolinium1, err_gadolinium1, cnr_gadolinium5, err_gadolinium5 = calculateCNR(ct_image_Gd, gadolinium1_roi,
                                                                                      gadolinium5_roi, noise_roi)
    #print( "Energy bin: K-edge SEC3 - SEC2")
    #print( "CNR of 1% Gd vial: ", cnr_gadolinium1, " +/- ", err_gadolinium1)
    #print( "CNR of 5% Gd vial: ", cnr_gadolinium5, " +/- ", err_gadolinium5)

    cnr_iodine1, err_iodine1, cnr_iodine5, err_iodine5 = calculateCNR(ct_image_I, iodine1_roi,
                                                                      iodine5_roi, noise_roi)
    #print( "Energy bin: K-edge SEC1 - SEC0")
    #print( "CNR of 1% I vial: ", cnr_iodine1, " +/- ", err_iodine1)
    #print( "CNR of 5% I vial: ", cnr_iodine5, " +/- ", err_iodine5)

    gold_data = np.array([cnr_gold1, err_gold1, cnr_gold5, err_gold5])
    gadolinium_data = np.array([cnr_gadolinium1, err_gadolinium1, cnr_gadolinium5, err_gadolinium5])
    iodine_data = np.array([cnr_iodine1, err_iodine1, cnr_iodine5, err_iodine5])
    cnr_data = np.array([gold_data, gadolinium_data, iodine_data])

    np.save(dir_data + "CNRdata_Kedge_SECmultiplex_corrected2.npy", cnr_data)
    
def main_Kedge_subtracted_plot(mat1,mat2,mat3,dir_data):
    '''
    After running main() method, do the K-edge subtraction
    :return:
    '''
    slice_no = 15
    m = 16

    radius = 1.593
    directory = "./images"
    #dir_data = "./data_no_smoothing/"
    
    #for i in range(15,16):
    i = 15
        

    img_I = sio.loadmat(mat1)['Reconimg'][:,:,slice_no]
    img_Gd = sio.loadmat(mat2)['Reconimg'][:,:,slice_no]
    img_Au = sio.loadmat(mat3)['Reconimg'][:,:,slice_no]

    size = np.shape(img_I)[0] #same size as all 3 here
    noise_roi = getBackgroundRegion(size)
    number_of_noise_pixels = np.sum(noise_roi)
    noise_fbp_image = noise_roi * img_I
    background_noise = np.sum(noise_fbp_image) / number_of_noise_pixels
    ct_image_I = 1000 * ((img_I / background_noise) - 1)

    noise_fbp_image = noise_roi * img_Gd
    background_noise = np.sum(noise_fbp_image) / number_of_noise_pixels
    ct_image_Gd = 1000 * ((img_Gd / background_noise) - 1)

    noise_fbp_image = noise_roi * img_Au
    background_noise = np.sum(noise_fbp_image) / number_of_noise_pixels
    ct_image_Au = 1000 * ((img_Au / background_noise) - 1)
    
    plot_as_rgb(img_Au,img_Gd,img_I)
    
    
    plot2DImage(img_I, extent=(-radius,radius,-radius,radius), colourmap=plt.get_cmap("Purples"), label_x="x [cm]",
                label_y="y [cm]", saveFile=directory + "/SEC-K_I/HU/Kedge_I_CTimage_slice"+str(i), plot=True,
                imkwargs={"vmin": None, "vmax":None})
    plot2DImage(img_Gd, extent=(-radius,radius,-radius,radius), colourmap=plt.get_cmap("Greens"), label_x="x [cm]",
                label_y="y [cm]", saveFile=directory + "/SEC-K_Gd/HU/Kedge_Gd_CTimage_slice" + str(i), plot=True,
                imkwargs={"vmin": None, "vmax": None})
    plot2DImage(img_Au, extent=(-radius,radius,-radius,radius), colourmap=plt.get_cmap("YlOrBr"), label_x="x [cm]",
                label_y="y [cm]", saveFile=directory + "/SEC-K_Au/HU/Kedge_Au_CTimage_slice" + str(i), plot=True,
                imkwargs={"vmin": None, "vmax": None})

    gold5_roi = getVialRegion("vial5Au", size)
    gold1_roi = getVialRegion("vial1Au", size)
    gadolinium5_roi = getVialRegion("vial5Gd", size)
    gadolinium1_roi = getVialRegion("vial1Gd", size)
    iodine5_roi = getVialRegion("vial5I", size)
    iodine1_roi = getVialRegion("vial1I", size)
    noise_roi = getBackgroundRegion(size)
    


    # calculate cnr
    cnr_gold1, err_gold1, cnr_gold5, err_gold5 = calculateCNR(img_Au, gold1_roi, gold5_roi, noise_roi)
    #print("Energy bin: K-edge SEC5 - SEC4")
    #print("CNR of 1% Au vial: ", cnr_gold1, " +/- ", err_gold1)
    #print("CNR of 5% Au vial: ", cnr_gold5, " +/- ", err_gold5)

    cnr_gadolinium1, err_gadolinium1, cnr_gadolinium5, err_gadolinium5 = calculateCNR(img_Gd, gadolinium1_roi,
                                                                                      gadolinium5_roi, noise_roi)
    #print( "Energy bin: K-edge SEC3 - SEC2")
    #print( "CNR of 1% Gd vial: ", cnr_gadolinium1, " +/- ", err_gadolinium1)
    #print( "CNR of 5% Gd vial: ", cnr_gadolinium5, " +/- ", err_gadolinium5)

    cnr_iodine1, err_iodine1, cnr_iodine5, err_iodine5 = calculateCNR(img_I, iodine1_roi,
                                                                      iodine5_roi, noise_roi)
    #print( "Energy bin: K-edge SEC1 - SEC0")
    #print( "CNR of 1% I vial: ", cnr_iodine1, " +/- ", err_iodine1)
    #print( "CNR of 5% I vial: ", cnr_iodine5, " +/- ", err_iodine5)

    gold_data = np.array([cnr_gold1, err_gold1, cnr_gold5, err_gold5])
    gadolinium_data = np.array([cnr_gadolinium1, err_gadolinium1, cnr_gadolinium5, err_gadolinium5])
    iodine_data = np.array([cnr_iodine1, err_iodine1, cnr_iodine5, err_iodine5])
    cnr_data = np.array([gold_data, gadolinium_data, iodine_data])

    np.save(dir_data + "CNRdata_Kedge_SECmultiplex_corrected2.npy", cnr_data)
    
def main_Kedge_subtracted_plot_max(mat1,mat2,mat3,dir_data):
    '''
    After running main() method, do the K-edge subtraction
    :return:
    '''
    slice_no = 15
    m = 16

    radius = 1.593
    directory = "./images"
    #dir_data = "./data_no_smoothing/"
    
    #for i in range(15,16):
    i = 15
        

    img_I = sio.loadmat(mat1)['Reconimg'][:,:,slice_no]
    img_Gd = sio.loadmat(mat2)['Reconimg'][:,:,slice_no]
    img_Au = sio.loadmat(mat3)['Reconimg'][:,:,slice_no]

    size = np.shape(img_I)[0] #same size as all 3 here
    noise_roi = getBackgroundRegion(size)
    number_of_noise_pixels = np.sum(noise_roi)
    noise_fbp_image = noise_roi * img_I
    background_noise = np.sum(noise_fbp_image) / number_of_noise_pixels
    ct_image_I = 1000 * ((img_I / background_noise) - 1)

    noise_fbp_image = noise_roi * img_Gd
    background_noise = np.sum(noise_fbp_image) / number_of_noise_pixels
    ct_image_Gd = 1000 * ((img_Gd / background_noise) - 1)

    noise_fbp_image = noise_roi * img_Au
    background_noise = np.sum(noise_fbp_image) / number_of_noise_pixels
    ct_image_Au = 1000 * ((img_Au / background_noise) - 1)
    
    noise_roi = getBackgroundRegion(size)
    air_roi = getAirRegion(size)
    
    number_of_noise_pixels = np.sum(noise_roi)
    number_of_air_pixels = np.sum(air_roi)
    
    noise_fbp_image = noise_roi * img_I
    noise_air_image = air_roi * img_I
    background_noise0 = np.sum(noise_fbp_image) / number_of_noise_pixels
    air_noise0 = np.sum(noise_air_image) / number_of_air_pixels
    ct_image0 = 1000 * ((img_I - background_noise0)/(background_noise0 - air_noise0))

    noise_fbp_image = noise_roi * img1
    noise_air_image = air_roi * img1
    background_noise1 = np.sum(noise_fbp_image) / number_of_noise_pixels
    air_noise1 = np.sum(noise_air_image) / number_of_air_pixels
    ct_image1 = 1000 * ((img1  - background_noise1)/(background_noise1 - air_noise1))

    noise_fbp_image = noise_roi * img2
    noise_air_image = air_roi * img2  
    background_noise2 = np.sum(noise_fbp_image) / number_of_noise_pixels
    air_noise2 = np.sum(noise_air_image) / number_of_air_pixels
    ct_image2 = 1000 * ((img2 - background_noise2)/(background_noise2 - air_noise2)) 
    
    plot_as_rgb(img_Au,img_Gd,img_I)
    
    
    plot2DImage(img_I, extent=(-radius,radius,-radius,radius), colourmap=plt.get_cmap("Purples"), label_x="x [cm]",
                label_y="y [cm]", saveFile=directory + "/SEC-K_I/HU/Kedge_I_CTimage_slice"+str(i), plot=True,
                imkwargs={"vmin": None, "vmax":None})
    plot2DImage(img_Gd, extent=(-radius,radius,-radius,radius), colourmap=plt.get_cmap("Greens"), label_x="x [cm]",
                label_y="y [cm]", saveFile=directory + "/SEC-K_Gd/HU/Kedge_Gd_CTimage_slice" + str(i), plot=True,
                imkwargs={"vmin": None, "vmax": None})
    plot2DImage(img_Au, extent=(-radius,radius,-radius,radius), colourmap=plt.get_cmap("YlOrBr"), label_x="x [cm]",
                label_y="y [cm]", saveFile=directory + "/SEC-K_Au/HU/Kedge_Au_CTimage_slice" + str(i), plot=True,
                imkwargs={"vmin": None, "vmax": None})

    gold5_roi = getVialRegion("vial5Au", size)
    gold1_roi = getVialRegion("vial1Au", size)
    gadolinium5_roi = getVialRegion("vial5Gd", size)
    gadolinium1_roi = getVialRegion("vial1Gd", size)
    iodine5_roi = getVialRegion("vial5I", size)
    iodine1_roi = getVialRegion("vial1I", size)
    noise_roi = getBackgroundRegion(size)
    


    # calculate cnr
    cnr_gold1, err_gold1, cnr_gold5, err_gold5 = calculateCNR(img_Au, gold1_roi, gold5_roi, noise_roi)
    #print("Energy bin: K-edge SEC5 - SEC4")
    #print("CNR of 1% Au vial: ", cnr_gold1, " +/- ", err_gold1)
    #print("CNR of 5% Au vial: ", cnr_gold5, " +/- ", err_gold5)

    cnr_gadolinium1, err_gadolinium1, cnr_gadolinium5, err_gadolinium5 = calculateCNR(img_Gd, gadolinium1_roi,
                                                                                      gadolinium5_roi, noise_roi)
    #print( "Energy bin: K-edge SEC3 - SEC2")
    #print( "CNR of 1% Gd vial: ", cnr_gadolinium1, " +/- ", err_gadolinium1)
    #print( "CNR of 5% Gd vial: ", cnr_gadolinium5, " +/- ", err_gadolinium5)

    cnr_iodine1, err_iodine1, cnr_iodine5, err_iodine5 = calculateCNR(img_I, iodine1_roi,
                                                                      iodine5_roi, noise_roi)
    #print( "Energy bin: K-edge SEC1 - SEC0")
    #print( "CNR of 1% I vial: ", cnr_iodine1, " +/- ", err_iodine1)
    #print( "CNR of 5% I vial: ", cnr_iodine5, " +/- ", err_iodine5)

    gold_data = np.array([cnr_gold1, err_gold1, cnr_gold5, err_gold5])
    gadolinium_data = np.array([cnr_gadolinium1, err_gadolinium1, cnr_gadolinium5, err_gadolinium5])
    iodine_data = np.array([cnr_iodine1, err_iodine1, cnr_iodine5, err_iodine5])
    cnr_data = np.array([gold_data, gadolinium_data, iodine_data])

    np.save(dir_data + "CNRdata_Kedge_SECmultiplex_corrected2.npy", cnr_data)    
    
def main_Kedge_subtracted_ML(mat1,mat2,mat3,dir_data):
    '''
    After running main() method, do the K-edge subtraction
    :return:
    '''
    slice_no = 15
    m = 16

    radius = 1.593
    directory = "./images"
    #dir_data = "./data_no_smoothing/"
    
    #for i in range(15,16):
    i = 15
        

    img_I = sio.loadmat(mat1)['im_hard']
    img_Gd = sio.loadmat(mat2)['im_hard']
    img_Au = sio.loadmat(mat3)['im_hard']

    size = np.shape(img_I)[0] #same size as all 3 here
    noise_roi = getBackgroundRegion(size)
    number_of_noise_pixels = np.sum(noise_roi)
    noise_fbp_image = noise_roi * img_I
    background_noise = np.sum(noise_fbp_image) / number_of_noise_pixels
    ct_image_I = 1000 * ((img_I / background_noise) - 1)

    noise_fbp_image = noise_roi * img_Gd
    background_noise = np.sum(noise_fbp_image) / number_of_noise_pixels
    ct_image_Gd = 1000 * ((img_Gd / background_noise) - 1)

    noise_fbp_image = noise_roi * img_Au
    background_noise = np.sum(noise_fbp_image) / number_of_noise_pixels
    ct_image_Au = 1000 * ((img_Au / background_noise) - 1)

    plot2DImage(img_I, extent=(-radius,radius,-radius,radius), colourmap=plt.get_cmap("Purples"), label_x="x [cm]",
                label_y="y [cm]", saveFile=directory + "/SEC-K_I/HU/Kedge_I_CTimage_slice"+str(i), plot=True,
                imkwargs={"vmin":0 , "vmax":5})
    plot2DImage(img_Gd, extent=(-radius,radius,-radius,radius), colourmap=plt.get_cmap("Greens"), label_x="x [cm]",
                label_y="y [cm]", saveFile=directory + "/SEC-K_Gd/HU/Kedge_Gd_CTimage_slice" + str(i), plot=True,
                imkwargs={"vmin": 0, "vmax": 5})
    plot2DImage(img_Au, extent=(-radius,radius,-radius,radius), colourmap=plt.get_cmap("YlOrBr"), label_x="x [cm]",
                label_y="y [cm]", saveFile=directory + "/SEC-K_Au/HU/Kedge_Au_CTimage_slice" + str(i), plot=True,
                imkwargs={"vmin": 0, "vmax": 5})
    
    img_I[img_I > 7] = 0
    img_I[img_I < -1] = 0
    
    img_Au[img_Au > 7] = 0
    img_Au[img_Au < -1] = 0

    img_Gd[img_Gd > 7] = 0
    img_Gd[img_Gd < -1] = 0
    
    
    plot_as_rgb(img_Au,img_Gd,img_I)

    gold5_roi = getVialRegion("vial5Au", size)
    gold1_roi = getVialRegion("vial1Au", size)
    gadolinium5_roi = getVialRegion("vial5Gd", size)
    gadolinium1_roi = getVialRegion("vial1Gd", size)
    iodine5_roi = getVialRegion("vial5I", size)
    iodine1_roi = getVialRegion("vial1I", size)
    noise_roi = getBackgroundRegion(size)
    


    # calculate cnr
    cnr_gold1, err_gold1, cnr_gold5, err_gold5 = calculateCNR(img_Au, gold1_roi, gold5_roi, noise_roi)
    #print("Energy bin: K-edge SEC5 - SEC4")
    #print("CNR of 1% Au vial: ", cnr_gold1, " +/- ", err_gold1)
    #print("CNR of 5% Au vial: ", cnr_gold5, " +/- ", err_gold5)

    cnr_gadolinium1, err_gadolinium1, cnr_gadolinium5, err_gadolinium5 = calculateCNR(img_Gd, gadolinium1_roi,
                                                                                      gadolinium5_roi, noise_roi)
    #print( "Energy bin: K-edge SEC3 - SEC2")
    #print( "CNR of 1% Gd vial: ", cnr_gadolinium1, " +/- ", err_gadolinium1)
    #print( "CNR of 5% Gd vial: ", cnr_gadolinium5, " +/- ", err_gadolinium5)

    cnr_iodine1, err_iodine1, cnr_iodine5, err_iodine5 = calculateCNR(img_I, iodine1_roi,
                                                                      iodine5_roi, noise_roi)
    #print( "Energy bin: K-edge SEC1 - SEC0")
    #print( "CNR of 1% I vial: ", cnr_iodine1, " +/- ", err_iodine1)
    #print( "CNR of 5% I vial: ", cnr_iodine5, " +/- ", err_iodine5)

    gold_data = np.array([cnr_gold1, err_gold1, cnr_gold5, err_gold5])
    gadolinium_data = np.array([cnr_gadolinium1, err_gadolinium1, cnr_gadolinium5, err_gadolinium5])
    iodine_data = np.array([cnr_iodine1, err_iodine1, cnr_iodine5, err_iodine5])
    cnr_data = np.array([gold_data, gadolinium_data, iodine_data])

    np.save(dir_data + "CNRdata_Kedge_SECmultiplex_corrected2.npy", cnr_data)
    
def plot_as_rgb(r,g,b):
    
    r -= np.amin(r)
    r /= np.amax(r)
    
    g -= np.amin(g)
    g /= np.amax(g)
    
    b -= np.amin(b)
    b /= np.amax(b)
    
    
    rgbArray = np.zeros((120,120,3), 'uint8')
    rgbArray[..., 0] = r*256
    rgbArray[..., 1] = g*256
    rgbArray[..., 2] = b*256
    img = Image.fromarray(rgbArray)
    plt.imshow(img)
    plt.show()

def plot_as_rgb2(r,g,b,m_1,m_2,m_3,gray):
 
    
    r -= np.amin(r)
    r /= np.amax(r)
    
    g -= np.amin(g)
    g /= np.amax(g)
    
    b -= np.amin(b)
    b /= np.amax(b)
    
    gray -= np.amin(gray)
    gray /= np.amax(gray)
    
    cm_purp = matplotlib.cm.get_cmap('Purples')
    cm_green = matplotlib.cm.get_cmap('Greens')
    cm_gold = matplotlib.cm.get_cmap('YlOrBr')
    
    rr = gold(r)
    gg = green(g)
    bb = purp(b)
    
    rr = np.uint8(rr * 255)
    gg = np.uint8(gg * 255)
    bb = np.uint8(bb * 255)
    
    #import ipdb; ipdb.set_trace()
    
    rr[m_1 == 0,:] = 0
    gg[m_2 == 0,:] = 0
    bb[m_3 == 0,:] = 0
    
    rgbArray = rr + gg + bb
    #rgbArray = np.zeros((120,120,3), 'uint8')
    #rgbArray[..., 0] = r*249 + b*207+ g*78
    #rgbArray[..., 1] = r*247 + b*65 + g*238
    #rgbArray[..., 2] = r*102 + b*217+ g*188
    
    alpha = 0.2

    # Construct RGB version of grey-level image
    img_color = np.dstack((gray, gray, gray))
    # Convert the input image and color mask to Hue Saturation Value (HSV)
    # colorspace
    img_hsv = color.rgb2hsv(img_color)
    #color_mask_hsv = color.rgb2hsv(rgbArray)

    # Replace the hue and saturation f the original image
    # with that of the color mask
    #img_hsv[..., 0] = color_mask_hsv[..., 0]
    #img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

    #img_masked = color.hsv2rgb(img_hsv)
    
    import scipy
    # Display the output
    f, (ax0, ax1, ax2) = plt.subplots(1, 3,
                                      subplot_kw={'xticks': [], 'yticks': []})
    ax0.imshow(gray, cmap=plt.cm.gray)
    scipy.misc.imsave('gray.png', gray)
    ax1.imshow(rgbArray)
    scipy.misc.imsave('rgb.jpg', rgbArray)
    #ax2.imshow(img_masked)
    plt.show()
    
    


def plot_as_rgb3(r,g,b,gray):
    
    r -= np.amin(r)
    r /= np.amax(r)
    
    g -= np.amin(g)
    g /= np.amax(g)
    
    b -= np.amin(b)
    b /= np.amax(b)
    
    gray -= np.amin(gray)
    gray /= np.amax(gray)
    
    rgbArray = np.zeros((120,120,3), 'uint8')
    rgbArray[..., 0] = r*256
    rgbArray[..., 1] = g*256
    rgbArray[..., 2] = b*256
    
    alpha = 0.2

    # Construct RGB version of grey-level image
    img_color = np.dstack((gray, gray, gray))
    img_color2 = np.dstack((r, g, b))
    img = Image.fromarray(img_color, 'RGBA')
    img2 = Image.fromarray(img_color2, 'RGBA')
    
    # alpha-blend the images with varying values of alpha
    a1 = Image.blend(img, img2, alpha=.2)
    a2 = Image.blend(img, img2, alpha=.4)

    # Display the output
    f, (ax0, ax1, ax2) = plt.subplots(1, 3,
                                      subplot_kw={'xticks': [], 'yticks': []})
    ax0.imshow(gray, cmap=plt.cm.gray)
    ax1.imshow(img)
    ax2.imshow(a2)
    plt.show()
    
def main_Kedge_subtract_normalize(dir_data):
    '''
    After running main() method, do the K-edge subtraction
    :return:
    '''
    slice_no = 15
    m = 16

    radius = 1.593
    directory = "./images"
    #dir_data = "./data_no_smoothing/"
    
    #for i in range(15,16):
    i = 15

    img0 = sio.loadmat(dir_data + "binSEC0_multiplex_corrected2.mat")
    img1 = sio.loadmat(dir_data + "binSEC1_multiplex_corrected2.mat")
    img2 = sio.loadmat(dir_data + "binSEC2_multiplex_corrected2.mat")
    img3 = sio.loadmat(dir_data + "binSEC3_multiplex_corrected2.mat")
    img4 = sio.loadmat(dir_data + "binSEC4_multiplex_corrected2.mat")
    img5 = sio.loadmat(dir_data + "binSEC5_multiplex_corrected2.mat")
    img6 = sio.loadmat(dir_data + "binSEC6_multiplex_corrected2.mat")

    #import ipdb; ipdb.set_trace() 
    '''
    img6['Reconimg'][:,:,slice_no] = img6['Reconimg'][:,:,slice_no]/np.linalg.norm(img6['Reconimg'][:,:,slice_no])

    img_I =  img6['Reconimg'][:,:,slice_no] - img0['Reconimg'][:,:,slice_no] / np.linalg.norm(img0['Reconimg'][:,:,slice_no])
    img_Gd = img6['Reconimg'][:,:,slice_no] - (img1['Reconimg'][:,:,slice_no] + img0['Reconimg'][:,:,slice_no]) /np.linalg.norm(img1['Reconimg'][:,:,slice_no] + img0['Reconimg'][:,:,slice_no])
    img_Au = img6['Reconimg'][:,:,slice_no] - (img1['Reconimg'][:,:,slice_no] + img0['Reconimg'][:,:,slice_no] + img3['Reconimg'][:,:,slice_no] + img4['Reconimg'][:,:,slice_no])/ np.linalg.norm(img1['Reconimg'][:,:,slice_no] + img0['Reconimg'][:,:,slice_no] + img3['Reconimg'][:,:,slice_no] + img4['Reconimg'][:,:,slice_no])
    '''
    img_I =  img6['Reconimg'][:,:,slice_no] - img0['Reconimg'][:,:,slice_no] 
    img_Gd = img6['Reconimg'][:,:,slice_no] - (img1['Reconimg'][:,:,slice_no] + img0['Reconimg'][:,:,slice_no]) 
    img_Au = img6['Reconimg'][:,:,slice_no] - (img1['Reconimg'][:,:,slice_no] + img0['Reconimg'][:,:,slice_no] + img3['Reconimg'][:,:,slice_no] + img4['Reconimg'][:,:,slice_no])

    
#     img_I =  img0['Reconimg'][:,:,slice_no]
#     img_Gd = img2['Reconimg'][:,:,slice_no] + img1['Reconimg'][:,:,slice_no] + img0['Reconimg'][:,:,slice_no]
#     img_Au = img2['Reconimg'][:,:,slice_no] + img1['Reconimg'][:,:,slice_no] + img0['Reconimg'][:,:,slice_no] + img3['Reconimg'][:,:,slice_no] + img4['Reconimg'][:,:,slice_no]
    
    

    size = np.shape(img_I)[0] #same size as all 3 here
    noise_roi = getBackgroundRegion(size)
    number_of_noise_pixels = np.sum(noise_roi)
    noise_fbp_image = noise_roi * img_I
    background_noise = np.sum(noise_fbp_image) / number_of_noise_pixels
    ct_image_I = 1000 * ((img_I / background_noise) - 1)

    noise_fbp_image = noise_roi * img_Gd
    background_noise = np.sum(noise_fbp_image) / number_of_noise_pixels
    ct_image_Gd = 1000 * ((img_Gd / background_noise) - 1)

    noise_fbp_image = noise_roi * img_Au
    background_noise = np.sum(noise_fbp_image) / number_of_noise_pixels
    ct_image_Au = 1000 * ((img_Au / background_noise) - 1)
    
    gold5_roi = getVialRegion("vial5Au", size)
    gold1_roi = getVialRegion("vial1Au", size)
    gadolinium5_roi = getVialRegion("vial5Gd", size)
    gadolinium1_roi = getVialRegion("vial1Gd", size)
    iodine5_roi = getVialRegion("vial5I", size)
    iodine1_roi = getVialRegion("vial1I", size)
    noise_roi = getBackgroundRegion(size)

    plot2DImage(img_I, extent=(-radius,radius,-radius,radius), colourmap=plt.get_cmap("Purples"), label_x="x [cm]",
                label_y="y [cm]", saveFile=directory + "/SEC-K_I/HU/Kedge_I_CTimage_slice"+str(i), plot=True,
                imkwargs={"vmin": None, "vmax":None})
    plot2DImage(img_Gd, extent=(-radius,radius,-radius,radius), colourmap=plt.get_cmap("Greens"), label_x="x [cm]",
                label_y="y [cm]", saveFile=directory + "/SEC-K_Gd/HU/Kedge_Gd_CTimage_slice" + str(i), plot=True,
                imkwargs={"vmin": None, "vmax": None})
    plot2DImage(img_Au, extent=(-radius,radius,-radius,radius), colourmap=plt.get_cmap("YlOrBr"), label_x="x [cm]",
                label_y="y [cm]", saveFile=directory + "/SEC-K_Au/HU/Kedge_Au_CTimage_slice" + str(i), plot=True,
                imkwargs={"vmin": None, "vmax": None})

    


    # calculate cnr
    cnr_gold1, err_gold1, cnr_gold5, err_gold5 = calculateCNR(img_Au, gold1_roi, gold5_roi, noise_roi)
    #print("Energy bin: K-edge SEC5 - SEC4")
    #print("CNR of 1% Au vial: ", cnr_gold1, " +/- ", err_gold1)
    #print("CNR of 5% Au vial: ", cnr_gold5, " +/- ", err_gold5)

    cnr_gadolinium1, err_gadolinium1, cnr_gadolinium5, err_gadolinium5 = calculateCNR(img_Gd, gadolinium1_roi,
                                                                                      gadolinium5_roi, noise_roi)
    #print( "Energy bin: K-edge SEC3 - SEC2")
    #print( "CNR of 1% Gd vial: ", cnr_gadolinium1, " +/- ", err_gadolinium1)
    #print( "CNR of 5% Gd vial: ", cnr_gadolinium5, " +/- ", err_gadolinium5)

    cnr_iodine1, err_iodine1, cnr_iodine5, err_iodine5 = calculateCNR(img_I, iodine1_roi,
                                                                      iodine5_roi, noise_roi)
    #print( "Energy bin: K-edge SEC1 - SEC0")
    #print( "CNR of 1% I vial: ", cnr_iodine1, " +/- ", err_iodine1)
    #print( "CNR of 5% I vial: ", cnr_iodine5, " +/- ", err_iodine5)

    gold_data = np.array([cnr_gold1, err_gold1, cnr_gold5, err_gold5])
    gadolinium_data = np.array([cnr_gadolinium1, err_gadolinium1, cnr_gadolinium5, err_gadolinium5])
    iodine_data = np.array([cnr_iodine1, err_iodine1, cnr_iodine5, err_iodine5])
    cnr_data = np.array([gold_data, gadolinium_data, iodine_data])

    np.save(dir_data + "CNRdata_Kedge_SECmultiplex_corrected2.npy", cnr_data)

def main_Kedge_subtract_scale(dir_data):
    '''
    After running main() method, do the K-edge subtraction
    :return:
    '''
    slice_no = 15
    m = 16

    radius = 1.593
    directory = "./images"
    #dir_data = "./data_no_smoothing/"
    
    #for i in range(15,16):
    i = 15

    img0 = sio.loadmat(dir_data + "binSEC0_multiplex_corrected2.mat")
    img1 = sio.loadmat(dir_data + "binSEC1_multiplex_corrected2.mat")
    img2 = sio.loadmat(dir_data + "binSEC2_multiplex_corrected2.mat")
    img3 = sio.loadmat(dir_data + "binSEC3_multiplex_corrected2.mat")
    img4 = sio.loadmat(dir_data + "binSEC4_multiplex_corrected2.mat")
    img5 = sio.loadmat(dir_data + "binSEC5_multiplex_corrected2.mat")

    #import ipdb; ipdb.set_trace() 

    img_I = (img1['Reconimg'][:,:,slice_no] - img0['Reconimg'][:,:,slice_no]) /(img1['Reconimg'][:,:,slice_no] + img0['Reconimg'][:,:,slice_no])
    img_Gd = (img3['Reconimg'][:,:,slice_no] - img2['Reconimg'][:,:,slice_no])/(img3['Reconimg'][:,:,slice_no] + img2['Reconimg'][:,:,slice_no])
    img_Au = (img5['Reconimg'][:,:,slice_no] - img4['Reconimg'][:,:,slice_no])/(img5['Reconimg'][:,:,slice_no] + img4['Reconimg'][:,:,slice_no])
    
#     img_I =  img0['Reconimg'][:,:,slice_no]
#     img_Gd = img2['Reconimg'][:,:,slice_no] + img1['Reconimg'][:,:,slice_no] + img0['Reconimg'][:,:,slice_no]
#     img_Au = img2['Reconimg'][:,:,slice_no] + img1['Reconimg'][:,:,slice_no] + img0['Reconimg'][:,:,slice_no] + img3['Reconimg'][:,:,slice_no] + img4['Reconimg'][:,:,slice_no]
    
    

    size = np.shape(img_I)[0] #same size as all 3 here
    noise_roi = getBackgroundRegion(size)
    number_of_noise_pixels = np.sum(noise_roi)
    noise_fbp_image = noise_roi * img_I
    background_noise = np.sum(noise_fbp_image) / number_of_noise_pixels
    ct_image_I = 1000 * ((img_I / background_noise) - 1)

    noise_fbp_image = noise_roi * img_Gd
    background_noise = np.sum(noise_fbp_image) / number_of_noise_pixels
    ct_image_Gd = 1000 * ((img_Gd / background_noise) - 1)

    noise_fbp_image = noise_roi * img_Au
    background_noise = np.sum(noise_fbp_image) / number_of_noise_pixels
    ct_image_Au = 1000 * ((img_Au / background_noise) - 1)
    
    gold5_roi = getVialRegion("vial5Au", size)
    gold1_roi = getVialRegion("vial1Au", size)
    gadolinium5_roi = getVialRegion("vial5Gd", size)
    gadolinium1_roi = getVialRegion("vial1Gd", size)
    iodine5_roi = getVialRegion("vial5I", size)
    iodine1_roi = getVialRegion("vial1I", size)
    noise_roi = getBackgroundRegion(size)

    plot2DImage(img_I, extent=(-radius,radius,-radius,radius), colourmap=plt.get_cmap("Purples"), label_x="x [cm]",
                label_y="y [cm]", saveFile=directory + "/SEC-K_I/HU/Kedge_I_CTimage_slice"+str(i), plot=True,
                imkwargs={"vmin": 0, "vmax":-0.05})
    plot2DImage(img_Gd, extent=(-radius,radius,-radius,radius), colourmap=plt.get_cmap("Greens"), label_x="x [cm]",
                label_y="y [cm]", saveFile=directory + "/SEC-K_Gd/HU/Kedge_Gd_CTimage_slice" + str(i), plot=True,
                imkwargs={"vmin": 0, "vmax": -0.05})
    plot2DImage(img_Au, extent=(-radius,radius,-radius,radius), colourmap=plt.get_cmap("YlOrBr"), label_x="x [cm]",
                label_y="y [cm]", saveFile=directory + "/SEC-K_Au/HU/Kedge_Au_CTimage_slice" + str(i), plot=True,
                imkwargs={"vmin": 0, "vmax": 0.1})

    


    # calculate cnr
    cnr_gold1, err_gold1, cnr_gold5, err_gold5 = calculateCNR(img_Au, gold1_roi, gold5_roi, noise_roi)
    #print("Energy bin: K-edge SEC5 - SEC4")
    #print("CNR of 1% Au vial: ", cnr_gold1, " +/- ", err_gold1)
    #print("CNR of 5% Au vial: ", cnr_gold5, " +/- ", err_gold5)

    cnr_gadolinium1, err_gadolinium1, cnr_gadolinium5, err_gadolinium5 = calculateCNR(img_Gd, gadolinium1_roi,
                                                                                      gadolinium5_roi, noise_roi)
    #print( "Energy bin: K-edge SEC3 - SEC2")
    #print( "CNR of 1% Gd vial: ", cnr_gadolinium1, " +/- ", err_gadolinium1)
    #print( "CNR of 5% Gd vial: ", cnr_gadolinium5, " +/- ", err_gadolinium5)

    cnr_iodine1, err_iodine1, cnr_iodine5, err_iodine5 = calculateCNR(img_I, iodine1_roi,
                                                                      iodine5_roi, noise_roi)
    #print( "Energy bin: K-edge SEC1 - SEC0")
    #print( "CNR of 1% I vial: ", cnr_iodine1, " +/- ", err_iodine1)
    #print( "CNR of 5% I vial: ", cnr_iodine5, " +/- ", err_iodine5)

    gold_data = np.array([cnr_gold1, err_gold1, cnr_gold5, err_gold5])
    gadolinium_data = np.array([cnr_gadolinium1, err_gadolinium1, cnr_gadolinium5, err_gadolinium5])
    iodine_data = np.array([cnr_iodine1, err_iodine1, cnr_iodine5, err_iodine5])
    cnr_data = np.array([gold_data, gadolinium_data, iodine_data])

    np.save(dir_data + "CNRdata_Kedge_SECmultiplex_corrected2.npy", cnr_data)



def calculateCNR(img, gold1_roi, gold5_roi, noise_roi):
    '''
    I mean, this works for 1% and 5% vials of any elements really
    Helper function to calculate CNR in regions of interest
    :param img:
    :param gold1_roi:
    :param gold5_roi:
    :param noise_roi:
    :return:
    '''
    mask_gold1 = img*gold1_roi
    mask_gold5 = img*gold5_roi
    mask_background = img*noise_roi

    signal_gold1 = np.average(mask_gold1[mask_gold1 != 0])
    signal_gold5 = np.average(mask_gold5[mask_gold5 != 0])
    signal_background = np.average(mask_background[mask_background != 0])
    noise_background = np.std(mask_background[mask_background != 0])
    noise_gold1 = np.std(mask_gold1[mask_gold1 != 0])
    noise_gold5 = np.std(mask_gold5[mask_gold5 != 0])

    cnr_gold1 = np.abs(signal_gold1 - signal_background) / noise_background
    cnr_gold5 = np.abs(signal_gold5 - signal_background) / noise_background
    err_gold1 = np.sqrt(noise_gold1**2 + noise_background**2) / noise_background
    err_gold5 = np.sqrt(noise_gold5 ** 2 + noise_background ** 2) / noise_background
    return cnr_gold1, err_gold1, cnr_gold5, err_gold5

def getVialRegion(vial_type, size):
    '''
    Hard coded for the SARRP phantom
    :param vial_type: one of "vial5Au" or "vial1Au"
    :param size: one dimension of image pixel size
    :return: mask with vial ROI
    '''
    mask = np.zeros([size, size])
    radius = 7
    if vial_type == "vial5Au":
        x_c, y_c = 92, 44
    elif vial_type == "vial1Au":
        x_c, y_c = 33, 77
    elif vial_type == "vial5Gd":
        x_c, y_c = 33, 44
    elif vial_type == "vial1Gd":
        x_c, y_c = 92,77
    elif vial_type == "vial1I":
        x_c, y_c = 62, 27
    elif vial_type == "vial5I":
        x_c, y_c = 62, 94
    else:
        print( "wrong vial type")

    for i in range(0, size):
        for j in range(0, size):
            if (i - x_c) ** 2 + (j - y_c) ** 2 < (radius ** 2):
                mask[j][i] = 1
    return mask

def getBackgroundRegion(size):
    #grabs the water region in middle of phantom
    mask = np.zeros([size, size])
    radius = 10
    for i in range(0, size):
        for j in range(0, size):
            # first ROI
            if (i-83) ** 2 + (j-99) ** 2 < (radius ** 2): #36, 32 #near gold 5%
                mask[j][i] = 1
                

    return mask
def getBackgroundRegion2(size):
    #grabs the water region in middle of phantom
    mask = np.zeros([size, size])
    radius = 10
    for i in range(0, size):
        for j in range(0, size):
            # first ROI
            if (i-(43)) ** 2 + (j-(22)) ** 2 < (radius ** 2): #36, 32 #near gold 5%
                mask[j][i] = 1
                

    return mask
def getAirRegion(size):
    #grabs the water region in middle of phantom
    mask = np.zeros([size, size])
    radius = 7
    for i in range(0, size):
        for j in range(0, size):
            # first ROI
            if (i-10) ** 2 + (j-10) ** 2 < (radius ** 2): #36, 32 #near gold 5%
                mask[j][i] = 1
                

    return mask

def end_plotCNR(dir_data):
    # For plotting CC
    SIZE = 14
    #dir_data = "./data_no_smoothing/"
    dir_images = "./images/"

    # *******************************************************
    # For plotting SEC
    # load all CNR data
    gold_dataK, gadolinium_dataK, iodine_dataK = np.load(dir_data + "CNRdata_Kedge_SECmultiplex_corrected2.npy")
    gold_data1, gadolinium_data1, iodine_data1 = np.load(dir_data + "CNRdata_1_multiplexslice_corrected2.npy")
    gold_data2, gadolinium_data2, iodine_data2 = np.load(dir_data + "CNRdata_2_multiplexslice_corrected2.npy")
    gold_data3, gadolinium_data3, iodine_data3 = np.load(dir_data + "CNRdata_3_multiplexslice_corrected2.npy")
    gold_data0, gadolinium_data0, iodine_data0 = np.load(dir_data + "CNRdata_0_multiplexslice_corrected2.npy")
    gold_data4, gadolinium_data4, iodine_data4 = np.load(dir_data + "CNRdata_4_multiplexslice_corrected2.npy")
    gold_data5, gadolinium_data5, iodine_data5 = np.load(dir_data + "CNRdata_5_multiplexslice_corrected2.npy")
    gold_data6, gadolinium_data6, iodine_data6 = np.load(dir_data + "CNRdata_6_multiplexslice_corrected2.npy")

    #cnr_gold1, err_gold1, cnr_gold5, err_gold5 = gold_data
    
    fig, ax = plt.subplots(nrows=2,ncols=2,figsize=(10,5), dpi=300)
    
    ax[0,0] = plt.subplot(1, 2, 1)
    ax[0,1] = plt.subplot(2, 2, 2)
    ax[1,0] = plt.subplot(2, 2, 4)
    
    # and finally, plot iodine contrast
    i_cnr1 = np.array([iodine_dataK[0], iodine_data0[0], iodine_data1[0], iodine_data6[0]])
    i_err1 = np.array([iodine_dataK[1], iodine_data0[1], iodine_data1[1], iodine_data6[1]])
    i_cnr5 = np.array([iodine_dataK[2], iodine_data0[2], iodine_data1[2], iodine_data6[2]])
    i_err5 = np.array([iodine_dataK[3], iodine_data0[3], iodine_data1[3], iodine_data6[3]])
    indices = np.arange(len(i_cnr1))
    width = 0.8
    ax[0,0].bar(indices, i_cnr5, width=width,
            color='#4616DC', label='5% I', yerr=i_err5, ecolor="k")
    ax[0,0].bar([i + 0.25 * 0.4 * width for i in indices], i_cnr1,
            width=0.8 * width, color='#A666EE', label='1% I', yerr=i_err1, ecolor="k")
    ax[0,0].axhline(4, color="r", linewidth=3, linestyle="--")
    ax[0,0].grid(True)
    ax[0,0].xaxis.set_tick_params(length=0)    
    ax[0,0].set_xticks(indices + width / 2.)
    ax[0,0].set_xticklabels(["K-edge               ", "16-33 keV               ", "33-41 keV               ", "Sum               "])
    #ax[0,0].set_yticks(size=SIZE)
    ax[0,0].set_ylabel("CNR", size=SIZE)
    ax[0,0].legend(loc=0, fontsize=SIZE)
    ax[0,0].set_ylim([0, 60])
    ax[0,0].set_title("Iodine Contrast")
    #plt.savefig(dir_images+"CNR_multiplex_iodine_SEC.png")
    #ax[0,0].show()

    #now plot gadolinium contrast
    gd_cnr1 = np.array([gadolinium_dataK[0], gadolinium_data2[0], gadolinium_data3[0], gadolinium_data6[0]])
    gd_err1 = np.array([gadolinium_dataK[1], gadolinium_data2[1], gadolinium_data3[1], gadolinium_data6[1]])
    gd_cnr5 = np.array([gadolinium_dataK[2], gadolinium_data2[2], gadolinium_data3[2], gadolinium_data6[2]])
    gd_err5 = np.array([gadolinium_dataK[3], gadolinium_data2[3], gadolinium_data3[3], gadolinium_data6[3]])
    indices = np.arange(len(gd_cnr1))
    width = 0.8
    ax[0,1].bar(indices, gd_cnr5, width=width,
            color='#147A00', label='5% Gd', yerr=gd_err5, ecolor="k")
    ax[0,1].bar([i + 0.25 * 0.4 * width for i in indices], gd_cnr1,
            width=0.8 * width, color='#37D710', label='1% Gd', yerr=gd_err1, ecolor="k")
    ax[0,1].axhline(4, color="r", linewidth=3, linestyle="--")
    ax[0,1].grid(True)
    #ax = plt.gca() 
    ax[0,1].yaxis.set_ticks([10,20])    
    ax[0,1].xaxis.set_tick_params(length=0)
    ax[0,1].set_xticks(indices + width / 2.)
    ax[0,1].set_xticklabels(["K-edge               ", "41-50 keV               ", "50-64 keV               ", "Sum               "])
    #ax[0,1].set_yticks(size=SIZE)
    ax[0,1].set_ylabel("CNR", size=SIZE)
    ax[0,1].legend(loc=0, fontsize=SIZE)
    ax[0,1].set_ylim([0, 25])
    ax[0,1].set_title("Gadolinium Contrast")
    #plt.savefig(dir_images+"CNR_multiplex_gadolinium_SEC.png")
    #ax[0,1].show()
    
    #Plot the gold CNR first
    au_cnr1 = np.array([gold_dataK[0], gold_data4[0], gold_data5[0], gold_data6[0]])
    au_err1 = np.array([gold_dataK[1], gold_data4[1], gold_data5[1], gold_data6[1]])
    au_cnr5 = np.array([gold_dataK[2], gold_data4[2], gold_data5[2], gold_data6[2]])
    au_err5 = np.array([gold_dataK[3], gold_data4[3], gold_data5[3], gold_data6[3]])
    indices = np.arange(len(au_cnr1))
    width = 0.8
    ax[1,0].bar(indices, au_cnr5, width=width,
            color='#CB870F', label='5% Au', yerr=au_err5, ecolor="k")
    ax[1,0].bar([i + 0.25 * 0.4 * width for i in indices], au_cnr1,
            width=0.8 * width, color='#F0E102', label='1% Au', yerr=au_err1, ecolor="k")
    ax[1,0].axhline(4, color="r", linewidth=3, linestyle="--") #this is the Rose criterion
    ax[1,0].grid(True)
    #import ipdb; ipdb.set_trace()
    #ax = plt.gca() 
    ax[1,0].yaxis.set_ticks([10,20])
    ax[1,0].xaxis.set_tick_params(length=0)
    ax[1,0].set_xticks(indices + width/2.)
    ax[1,0].set_xticklabels(["K-edge               ", " 64-81 keV                 ", "81-120 keV              ", "Sum               "])
    ax[1,0].set_ylabel("CNR", size=SIZE)
    ax[1,0].legend(loc=0, fontsize=SIZE)
    ax[1,0].set_ylim([0, 25])
    ax[1,0].set_title("Gold Contrast")
    #ax[1,0].show()
    fig.tight_layout()
    #plt.savefig(dir_images+"CNR_multiplex_gold_SEC.png")    

def end_plotCNR_3(dir_data):
    # For plotting CC
    SIZE = 18
    #dir_data = "./data_no_smoothing/"
    dir_images = "./images/"

    # *******************************************************
    # For plotting SEC
    # load all CNR data
    gold_dataK, gadolinium_dataK, iodine_dataK = np.load(dir_data + "CNRdata_Kedge_SECmultiplex_corrected2.npy")
    gold_data1, gadolinium_data1, iodine_data1 = gold_dataK, gadolinium_dataK, iodine_dataK
    gold_data2, gadolinium_data2, iodine_data2 = gold_dataK, gadolinium_dataK, iodine_dataK
    gold_data3, gadolinium_data3, iodine_data3 = gold_dataK, gadolinium_dataK, iodine_dataK
    gold_data0, gadolinium_data0, iodine_data0 = gold_dataK, gadolinium_dataK, iodine_dataK 
    gold_data4, gadolinium_data4, iodine_data4 =gold_dataK, gadolinium_dataK, iodine_dataK 
    gold_data5, gadolinium_data5, iodine_data5 =gold_dataK, gadolinium_dataK, iodine_dataK 
    gold_data6, gadolinium_data6, iodine_data6 =gold_dataK, gadolinium_dataK, iodine_dataK 
    #cnr_gold1, err_gold1, cnr_gold5, err_gold5 = gold_data

    
    # and finally, plot iodine contrast
    i_cnr1 = np.array([iodine_dataK[0], iodine_data0[0], iodine_data1[0], iodine_data6[0]])
    i_err1 = np.array([iodine_dataK[1], iodine_data0[1], iodine_data1[1], iodine_data6[1]])
    i_cnr5 = np.array([iodine_dataK[2], iodine_data0[2], iodine_data1[2], iodine_data6[2]])
    i_err5 = np.array([iodine_dataK[3], iodine_data0[3], iodine_data1[3], iodine_data6[3]])
    indices = np.arange(len(i_cnr1))
    width = 0.8
    plt.bar(indices, i_cnr5, width=width,
            color='#4616DC', label='5% I', yerr=i_err5, ecolor="k")
    plt.bar([i + 0.25 * 0.4 * width for i in indices], i_cnr1,
            width=0.8 * width, color='#A666EE', label='1% I', yerr=i_err1, ecolor="k")
    plt.axhline(4, color="r", linewidth=3, linestyle="--")
    plt.grid(True)
    plt.xticks(indices + width / 2., ["K-edge", "SEC0", "SEC1", "Sum"], size=SIZE)
    plt.yticks(size=SIZE)
    plt.ylabel("CNR", size=SIZE)
    plt.legend(loc=0, fontsize=SIZE)
    plt.ylim([0, 60])
    plt.xlim([-0.5, 0.5])
    plt.title("CNR of Multiplexed Spectral CT \n with Iodine Contrast - SEC")
    #plt.savefig(dir_images+"CNR_multiplex_iodine_SEC.png")
    plt.show()

    #now plot gadolinium contrast
    gd_cnr1 = np.array([gadolinium_dataK[0], gadolinium_data2[0], gadolinium_data3[0], gadolinium_data6[0]])
    gd_err1 = np.array([gadolinium_dataK[1], gadolinium_data2[1], gadolinium_data3[1], gadolinium_data6[1]])
    gd_cnr5 = np.array([gadolinium_dataK[2], gadolinium_data2[2], gadolinium_data3[2], gadolinium_data6[2]])
    gd_err5 = np.array([gadolinium_dataK[3], gadolinium_data2[3], gadolinium_data3[3], gadolinium_data6[3]])
    indices = np.arange(len(gd_cnr1))
    width = 0.8
    plt.bar(indices, gd_cnr5, width=width,
            color='#147A00', label='5% Gd', yerr=gd_err5, ecolor="k")
    plt.bar([i + 0.25 * 0.4 * width for i in indices], gd_cnr1,
            width=0.8 * width, color='#37D710', label='1% Gd', yerr=gd_err1, ecolor="k")
    plt.axhline(4, color="r", linewidth=3, linestyle="--")
    plt.grid(True)
    plt.xticks(indices + width / 2., ["K-edge", "SEC2", "SEC3", "Sum"], size=SIZE)
    plt.yticks(size=SIZE)
    plt.ylabel("CNR", size=SIZE)
    plt.legend(loc=0, fontsize=SIZE)
    plt.ylim([0, 60])
    plt.xlim([-0.5, 0.5])
    plt.title("CNR of Multiplexed Spectral CT \n with Gadolinium Contrast - SEC")
    plt.savefig(dir_images+"CNR_multiplex_gadolinium_SEC.png")
    plt.show()
    
    #Plot the gold CNR first
    au_cnr1 = np.array([gold_dataK[0], gold_data4[0], gold_data5[0], gold_data6[0]])
    au_err1 = np.array([gold_dataK[1], gold_data4[1], gold_data5[1], gold_data6[1]])
    au_cnr5 = np.array([gold_dataK[2], gold_data4[2], gold_data5[2], gold_data6[2]])
    au_err5 = np.array([gold_dataK[3], gold_data4[3], gold_data5[3], gold_data6[3]])
    indices = np.arange(len(au_cnr1))
    width = 0.8
    plt.bar(indices, au_cnr5, width=width,
            color='#CB870F', label='5% Au', yerr=au_err5, ecolor="k")
    plt.bar([i + 0.25 * 0.4 * width for i in indices], au_cnr1,
            width=0.8 * width, color='#F0E102', label='1% Au', yerr=au_err1, ecolor="k")
    plt.axhline(4, color="r", linewidth=3, linestyle="--") #this is the Rose criterion
    plt.grid(True)
    plt.xticks(indices + width / 2., ["K-edge", "SEC4", "SEC5", "Sum"], size=SIZE)
    plt.yticks(size=SIZE)
    plt.ylabel("CNR", size=SIZE)
    plt.legend(loc=0, fontsize=SIZE)
    plt.ylim([0, 60])
    plt.xlim([-0.5, 0.5])
    plt.title("CNR of Multiplexed Spectral CT \n with Gold Contrast - SEC")
    plt.savefig(dir_images+"CNR_multiplex_gold_SEC.png")
    plt.show()


def main_Kedge_subtract(dir_data):
    '''
    After running main() method, do the K-edge subtraction
    :return:
    '''
    slice_no = 15
    m = 16

    radius = 1.593
    directory = "./images"
    #dir_data = "./data_no_smoothing/"
    
    #for i in range(15,16):
    i = 15

    img0 = sio.loadmat(dir_data + "binSEC0_multiplex_corrected2.mat")
    img1 = sio.loadmat(dir_data + "binSEC1_multiplex_corrected2.mat")
    img2 = sio.loadmat(dir_data + "binSEC2_multiplex_corrected2.mat")
    img3 = sio.loadmat(dir_data + "binSEC3_multiplex_corrected2.mat")
    img4 = sio.loadmat(dir_data + "binSEC4_multiplex_corrected2.mat")
    img5 = sio.loadmat(dir_data + "binSEC5_multiplex_corrected2.mat")

    #import ipdb; ipdb.set_trace() 

    img_I = np.average(img1['Reconimg'][:,:,slice_no:m],2) - np.average(img0['Reconimg'][:,:,slice_no:m],2)
    img_Gd = np.average(img3['Reconimg'][:,:,slice_no:m],2) - np.average(img2['Reconimg'][:,:,slice_no:m],2)
    img_Au = np.average(img5['Reconimg'][:,:,slice_no:m],2) - np.average(img4['Reconimg'][:,:,slice_no:m],2)

    size = np.shape(img_I)[0] #same size as all 3 here
    noise_roi = getBackgroundRegion(size)
    number_of_noise_pixels = np.sum(noise_roi)
    noise_fbp_image = noise_roi * img_I
    background_noise = np.sum(noise_fbp_image) / number_of_noise_pixels
    ct_image_I = 1000 * ((img_I / background_noise) - 1)

    noise_fbp_image = noise_roi * img_Gd
    background_noise = np.sum(noise_fbp_image) / number_of_noise_pixels
    ct_image_Gd = 1000 * ((img_Gd / background_noise) - 1)

    noise_fbp_image = noise_roi * img_Au
    background_noise = np.sum(noise_fbp_image) / number_of_noise_pixels
    ct_image_Au = 1000 * ((img_Au / background_noise) - 1)

    plot2DImage(img_I, extent=(-radius,radius,-radius,radius), colourmap=plt.get_cmap("Greys_r"), label_x="x [cm]",
                label_y="y [cm]",  saveFile=directory + "/SEC-K_I/HU/Kedge_I_CTimage_slice"+str(i), plot=True,
                imkwargs={"vmin": None, "vmax": None})
    plot2DImage(img_Gd, extent=(-radius,radius,-radius,radius), colourmap=plt.get_cmap("Greys_r"), label_x="x [cm]",
                label_y="y [cm]",  saveFile=directory + "/SEC-K_Gd/HU/Kedge_Gd_CTimage_slice" + str(i), plot=True,
                imkwargs={"vmin": None, "vmax": None})
    plot2DImage(img_Au, extent=(-radius,radius,-radius,radius), colourmap=plt.get_cmap("Greys_r"), label_x="x [cm]",
                label_y="y [cm]",  saveFile=directory + "/SEC-K_Au/HU/Kedge_Au_CTimage_slice" + str(i), plot=True,
                imkwargs={"vmin": None, "vmax": None})

    gold5_roi = getVialRegion("vial5Au", size)
    gold1_roi = getVialRegion("vial1Au", size)
    gadolinium5_roi = getVialRegion("vial5Gd", size)
    gadolinium1_roi = getVialRegion("vial1Gd", size)
    iodine5_roi = getVialRegion("vial5I", size)
    iodine1_roi = getVialRegion("vial1I", size)
    noise_roi = getBackgroundRegion(size)

    # calculate cnr
    cnr_gold1, err_gold1, cnr_gold5, err_gold5 = calculateCNR(img_Au, gold1_roi, gold5_roi, noise_roi)
    #print("Energy bin: K-edge SEC5 - SEC4")
    #print("CNR of 1% Au vial: ", cnr_gold1, " +/- ", err_gold1)
    #print("CNR of 5% Au vial: ", cnr_gold5, " +/- ", err_gold5)

    cnr_gadolinium1, err_gadolinium1, cnr_gadolinium5, err_gadolinium5 = calculateCNR(img_Gd, gadolinium1_roi,
                                                                                      gadolinium5_roi, noise_roi)
    #print( "Energy bin: K-edge SEC3 - SEC2")
    #print( "CNR of 1% Gd vial: ", cnr_gadolinium1, " +/- ", err_gadolinium1)
    #print( "CNR of 5% Gd vial: ", cnr_gadolinium5, " +/- ", err_gadolinium5)

    cnr_iodine1, err_iodine1, cnr_iodine5, err_iodine5 = calculateCNR(img_I, iodine1_roi,
                                                                      iodine5_roi, noise_roi)
    #print( "Energy bin: K-edge SEC1 - SEC0")
    #print( "CNR of 1% I vial: ", cnr_iodine1, " +/- ", err_iodine1)
    #print( "CNR of 5% I vial: ", cnr_iodine5, " +/- ", err_iodine5)

    gold_data = np.array([cnr_gold1, err_gold1, cnr_gold5, err_gold5])
    gadolinium_data = np.array([cnr_gadolinium1, err_gadolinium1, cnr_gadolinium5, err_gadolinium5])
    iodine_data = np.array([cnr_iodine1, err_iodine1, cnr_iodine5, err_iodine5])
    cnr_data = np.array([gold_data, gadolinium_data, iodine_data])

    np.save(dir_data + "CNRdata_Kedge_SECmultiplex_corrected2.npy", cnr_data)


def plot_composite():
    
        # Function to change the image size
    def changeImageSize(maxWidth, 
                        maxHeight, 
                        image):

        widthRatio  = maxWidth/image.size[0]
        heightRatio = maxHeight/image.size[1]

        newWidth    = int(widthRatio*image.size[0])
        newHeight   = int(heightRatio*image.size[1])

        newImage    = image.resize((newWidth, newHeight))
        return newImage

    # Take two images for blending them together   
    image1 = Image.open("./rgb.jpg")
    image2 = Image.open("./gray.png")

    # Make the images of uniform size
    image3 = changeImageSize(800, 800, image1)
    image4 = changeImageSize(800, 800, image2)

    # Make sure images got an alpha channel
    image5 = image3.convert("RGBA")
    image6 = image4.convert("RGBA")

    # Display the images
    #image5.show()
    #image6.show()

    # alpha-blend the images with varying values of alpha
    alphaBlended1 = Image.blend(image5, image6, alpha=.1)
    alphaBlended2 = Image.blend(image5, image6, alpha=.5)

    # Display the alpha-blended images
    #plt.figure()
    #plt.imshow(alphaBlended1)
    #plt.figure()
    #plt.imshow(alphaBlended2)

    f, (ax0, ax1, ax2) = plt.subplots(1, 3,
                                      subplot_kw={'xticks': [], 'yticks': []})
    ax0.imshow(image2, cmap=plt.cm.gray)
    ax0.set_title('a) CT image',size = 10)
    ax1.imshow(image1)
    ax1.set_title('b) Composite K-edge image',size = 10)
    ax2.imshow(alphaBlended2)
    ax2.set_title('c) Overlay',size = 10)
    f.savefig('rgb3.png',dpi=600)
    plt.show()
