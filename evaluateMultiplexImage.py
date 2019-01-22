import numpy as np
import scipy.io as sio
from CDtools.PlottingTools.ShowPlots import plot2DImage
import matplotlib.pyplot as plt

def main():
    '''
    Run this one first to plot original images
    :return:
    '''
    #load images
    directory = "./data_smoothing/"
    dir_images = "./images/"
    slice_no = 9  # slice 9 in matlab
    energy_bin = "SEC0" #SEC1, EC, etc
    radius = 1.593

    data = sio.loadmat(directory + "bin" + energy_bin + "_multiplex_corrected2.mat")
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
                label_y="y [cm]", label_cb="HU", imkwargs={"vmin": -1000, "vmax": 2000},
                saveFile=dir_images+energy_bin+"_CTimage_slice"+str(slice_no)) #extent=(-radius,radius,-radius,radius)

    #save all slice numbers to find optimal K-edge image
    for i in range(1,24):
        image = data['Reconimg'][:,:,i]
        np.save(energy_bin+"_CTimage_slice"+str(i)+".npy",image)

    #calculate cnr
    cnr_gold1, err_gold1, cnr_gold5, err_gold5 = calculateCNR(img, gold1_roi, gold5_roi, noise_roi)
    print "Energy bin: ", energy_bin
    print "CNR of 1% Au vial: ", cnr_gold1, " +/- ", err_gold1
    print "CNR of 5% Au vial: ", cnr_gold5, " +/- ", err_gold5

    cnr_gadolinium1, err_gadolinium1, cnr_gadolinium5, err_gadolinium5 = calculateCNR(img, gadolinium1_roi,
                                                                                      gadolinium5_roi, noise_roi)
    print "Energy bin: ", energy_bin
    print "CNR of 1% Gd vial: ", cnr_gadolinium1, " +/- ", err_gadolinium1
    print "CNR of 5% Gd vial: ", cnr_gadolinium5, " +/- ", err_gadolinium5

    cnr_iodine1, err_iodine1, cnr_iodine5, err_iodine5 = calculateCNR(img, iodine1_roi,
                                                                      iodine5_roi, noise_roi)
    print "Energy bin: ", energy_bin
    print "CNR of 1% I vial: ", cnr_iodine1, " +/- ", err_iodine1
    print "CNR of 5% I vial: ", cnr_iodine5, " +/- ", err_iodine5

    #saving CNR data to be loaded later
    gold_data = np.array([cnr_gold1, err_gold1, cnr_gold5, err_gold5])
    gadolinium_data = np.array([cnr_gadolinium1, err_gadolinium1, cnr_gadolinium5, err_gadolinium5])
    iodine_data = np.array([cnr_iodine1, err_iodine1, cnr_iodine5, err_iodine5])
    cnr_data = np.array([gold_data,gadolinium_data,iodine_data])

    np.save(directory+"CNRdata_"+energy_bin+"_multiplexslice"+str(slice_no)+"_corrected2.npy",cnr_data)

def main_Kedge():
    '''
    After running main() method, do the K-edge subtraction
    :return:
    '''
    slice_no = 9

    radius = 1.593
    directory = "./images"
    dir_data = "./data_smoothing/"
    for i in range(15,16):
        img0 = np.load("SEC0_CTimage_slice" + str(i) + ".npy")
        img1 = np.load("SEC1_CTimage_slice" + str(i) + ".npy")
        img2 = np.load("SEC2_CTimage_slice" + str(i) + ".npy")
        img3 = np.load("SEC3_CTimage_slice" + str(i) + ".npy")
        img4 = np.load("SEC4_CTimage_slice" + str(i) + ".npy")
        img5 = np.load("SEC5_CTimage_slice" + str(i) + ".npy")

        img_I = img1 - img0
        img_Gd = img3 - img2
        img_Au = img5 - img4

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

        plot2DImage(ct_image_I, extent=(-radius,radius,-radius,radius), colourmap=plt.get_cmap("Greys_r"), label_x="x [cm]",
                    label_y="y [cm]", label_cb="HU", saveFile=directory + "/SEC-K_I/HU/Kedge_I_CTimage_slice"+str(i), plot=True,
                    imkwargs={"vmin": None, "vmax": None})
        plot2DImage(ct_image_Gd, extent=(-radius,radius,-radius,radius), colourmap=plt.get_cmap("Greys_r"), label_x="x [cm]",
                    label_y="y [cm]", label_cb="HU", saveFile=directory + "/SEC-K_Gd/HU/Kedge_Gd_CTimage_slice" + str(i), plot=True,
                    imkwargs={"vmin": None, "vmax": None})
        plot2DImage(ct_image_Au, extent=(-radius,radius,-radius,radius), colourmap=plt.get_cmap("Greys_r"), label_x="x [cm]",
                    label_y="y [cm]", label_cb="HU", saveFile=directory + "/SEC-K_Au/HU/Kedge_Au_CTimage_slice" + str(i), plot=True,
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
    print "Energy bin: K-edge SEC5 - SEC4"
    print "CNR of 1% Au vial: ", cnr_gold1, " +/- ", err_gold1
    print "CNR of 5% Au vial: ", cnr_gold5, " +/- ", err_gold5

    cnr_gadolinium1, err_gadolinium1, cnr_gadolinium5, err_gadolinium5 = calculateCNR(img_Gd, gadolinium1_roi,
                                                                                      gadolinium5_roi, noise_roi)
    print "Energy bin: K-edge SEC3 - SEC2"
    print "CNR of 1% Gd vial: ", cnr_gadolinium1, " +/- ", err_gadolinium1
    print "CNR of 5% Gd vial: ", cnr_gadolinium5, " +/- ", err_gadolinium5

    cnr_iodine1, err_iodine1, cnr_iodine5, err_iodine5 = calculateCNR(img_I, iodine1_roi,
                                                                      iodine5_roi, noise_roi)
    print "Energy bin: K-edge SEC1 - SEC0"
    print "CNR of 1% I vial: ", cnr_iodine1, " +/- ", err_iodine1
    print "CNR of 5% I vial: ", cnr_iodine5, " +/- ", err_iodine5

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
    radius = 8
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
        print "wrong vial type"

    for i in range(0, size):
        for j in range(0, size):
            if (i - x_c) ** 2 + (j - y_c) ** 2 < (radius ** 2):
                mask[j][i] = 1
    return mask

def getBackgroundRegion(size):
    #grabs the water region in middle of phantom
    mask = np.zeros([size, size])
    radius = 8
    for i in range(0, size):
        for j in range(0, size):
            # first ROI
            if (i-62) ** 2 + (j-60) ** 2 < (radius ** 2): #36, 32 #near gold 5%
                mask[j][i] = 1

    return mask

def end_plotCNR():
    # For plotting CC
    SIZE = 18
    dir_data = "/home/chelsea/Desktop/UVICnotes/data_multiplex/matlab/data/"
    dir_images = "/home/chelsea/Desktop/UVICnotes/data_multiplex/matlab/images/"

    # *******************************************************
    # For plotting SEC
    # load all CNR data
    gold_dataK, gadolinium_dataK, iodine_dataK = np.load(dir_data + "CNRdata_Kedge_SECmultiplex_corrected2.npy")
    gold_data1, gadolinium_data1, iodine_data1 = np.load(dir_data + "CNRdata_SEC1_multiplex_corrected2.npy")
    gold_data2, gadolinium_data2, iodine_data2 = np.load(dir_data + "CNRdata_SEC2_multiplex_corrected2.npy")
    gold_data3, gadolinium_data3, iodine_data3 = np.load(dir_data + "CNRdata_SEC3_multiplex_corrected2.npy")
    gold_data0, gadolinium_data0, iodine_data0 = np.load(dir_data + "CNRdata_SEC0_multiplex_corrected2.npy")
    gold_data4, gadolinium_data4, iodine_data4 = np.load(dir_data + "CNRdata_SEC4_multiplex_corrected2.npy")
    gold_data5, gadolinium_data5, iodine_data5 = np.load(dir_data + "CNRdata_SEC5_multiplex_corrected2.npy")
    gold_data6, gadolinium_data6, iodine_data6 = np.load(dir_data + "CNRdata_SEC6_multiplex_corrected2.npy")

    #cnr_gold1, err_gold1, cnr_gold5, err_gold5 = gold_data

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
    plt.title("CNR of Multiplexed Spectral CT \n with Gold Contrast - SEC")
    plt.savefig(dir_images+"CNR_multiplex_gold_SEC.png")
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
    plt.title("CNR of Multiplexed Spectral CT \n with Gadolinium Contrast - SEC")
    plt.savefig(dir_images+"CNR_multiplex_gadolinium_SEC.png")
    plt.show()

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
    plt.title("CNR of Multiplexed Spectral CT \n with Iodine Contrast - SEC")
    plt.savefig(dir_images+"CNR_multiplex_iodine_SEC.png")
    plt.show()


if __name__ == "__main__":
    #main()
    main_Kedge()
    end_plotCNR()
