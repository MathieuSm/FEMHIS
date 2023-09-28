#%% !/usr/bin/env python3
# Initialization

Version = '01'

# Define the script description
Description = """
    This script perform automatic segmentation using unet and random forest

    Author: Mathieu Simon
            ARTORG Center for Biomedical Engineering Research
            SITEM Insel
            University of Bern

    Date: June 2023
    """

#%% Imports
# Modules import

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import joblib
import argparse
import numpy as np
import pandas as pd
from keras import utils
import SimpleITK as sitk
from patchify import patchify
import matplotlib.pyplot as plt
from keras.models import load_model
from Utils import Time, SetDirectories
from skimage import io, feature, measure, color

#%% Functions
# Define functions

def SegmentBone(Array, Plot=False):

    """
    Segment bone structure
    :param Image: RGB numpy array dim r x c x 3
    :param Plot: Plot the results (bool)
    :return: Segmented bone image
    """

    Time.Process(1, 'Segment bone')
    
    # Mark areas where there is bone
    Filter1 = Array[:, :, 0] > 160
    Filter2 = Array[:, :, 1] > 160
    Filter3 = Array[:, :, 2] > 160
    Bone = ~(Filter1 & Filter2 & Filter3)

    if Plot:
        Shape = np.array(Array.shape) / max(Array.shape) * 10
        Figure, Axis = plt.subplots(1, 1, figsize=(Shape[1], Shape[0]))
        Axis.imshow(Array)
        Axis.axis('off')
        plt.subplots_adjust(0, 0, 1, 1)
        plt.show()

        Figure, Axis = plt.subplots(1, 1, figsize=(Shape[1], Shape[0]))
        Axis.imshow(Bone, cmap='binary')
        Axis.axis('off')
        plt.subplots_adjust(0, 0, 1, 1)
        plt.show()

    # Keep connected component
    Labels = measure.label(Bone)
    Areas = measure.regionprops_table(Labels, properties=['area'])['area']
    KArea = np.where(Areas > 1E6)[0]
    Bone = np.zeros(Labels.shape)
    for K in KArea:
        Bone += Labels == K + 1

    # Print elapsed time
    Time.Process(0)

    return Bone

def ExtractROIs(Array, Bone, N, ROIName):

    """
    Extract regions of interest of cortical bone according to the parameters given as arguments for the Main function.
    According to Grimal et al (2011), cortical bone representative volume element should be around 1mm side length and
    presents a BV/TV of 88% at least. Therefore, a threshold of 0.88 is used to ensure that the selected ROI reaches
    this value.

    Grimal, Q., Raum, K., Gerisch, A., &#38; Laugier, P. (2011)
    A determination of the minimum sizes of representative volume elements
    for the prediction of cortical bone elastic properties
    Biomechanics and Modeling in Mechanobiology (6), 925-937
    https://doi.org/10.1007/s10237-010-0284-9

    :param Array: 3D numpy array (2D + RGB)
    :param Bone: 2D numpy array of segmented bone (bool)
    :param N: Number of ROIs to extract (int)
    :param Plot: Plot the results (bool)
    :return: ROIs
    """

    # Fixed parameters
    Threshold = 0.88                # BV/TV threshold
    Pixel_S = 1.0460251046025104    # Pixel spacing
    ROI_S = 2000                    # ROI physical size

    # Record time
    Time.Process(1, str(N) + ' ROIs selection')

    # Set ROI pixel size
    ROISize = int(round(ROI_S / Pixel_S))

    # Define region between the two holes
    Left = int(Bone.shape[1] * 0.2)
    Right = int(Bone.shape[1] * 0.8)

    # Patchify image and keep ROIs with BV/TV > Threshold
    Step = (ROISize//8, ROISize//8)
    Patches = patchify(Bone[:,Left:Right], patch_size=(ROISize, ROISize), step=Step)
    Valid = np.sum(Patches, axis=(2,3)) / (ROISize**2) > Threshold

    # Select random ROIs
    Y, X = np.where(Valid)
    Step = (ROISize//8, ROISize//8, 3)
    Patches = patchify(Array[:,Left:Right], patch_size=(ROISize, ROISize, 3), step=Step)
    
    # Semi-random coordinates
    xMin = X[np.abs(X - 0.15 * (X.max() - X.min())).argmin()]
    xMid = X[np.abs(X - 0.5 * (X.max() - X.min())).argmin()]
    xMax = X[np.abs(X - 0.85 * (X.max() - X.min())).argmin()]

    yMin = int(np.median(Y[X == xMin]))
    yMid = int(np.median(Y[X == xMid]))
    yMax = int(np.median(Y[X == xMax]))
    Rand = [(xMin, yMin), (xMid, yMid), (xMax, yMax)]

    ROIs = []
    Xs = []
    Ys = []
    for i, (Rx, Ry) in enumerate(Rand):
        ROI = Patches[Ry, Rx][0].astype('uint8')
        Name = str(ROIName) + '_' + str(i) + '.png'
        io.imsave(Name, ROI)
        X0 = Rx * Step[0] + Left
        Y0 = Ry * Step[1]
        Xs.append([X0, X0 + ROISize])
        Ys.append([Y0, Y0 + ROISize])
        ROIs.append(ROI)

    Xs = np.array(Xs)
    Ys = np.array(Ys)
    ROIs = np.array(ROIs)

    # Print elapsed time
    Time.Process(0, str(N) + ' ROIs selected')


    return ROIs, Xs, Ys

def PlotROIs(Bone, Array, ROIName, Xs, Ys):

    # Bone segmentation
    Seg = np.zeros(Bone.shape + (4, ), int)
    Seg[:,:,1] = np.array(Bone * 255).astype(int)
    Seg[:,:,-1] = np.array(Bone * 255).astype(int)

    # Downsample arrays for plotting
    Factor = 4
    dArray = Array[::Factor, ::Factor]
    dSeg = Seg[::Factor, ::Factor]
    Xs = Xs / Factor
    Ys = Ys / Factor

    # Plot full image with ROIs location and overlay bone segmentation
    Shape = np.array(Array.shape[:-1]) / 1000
    Figure, Axis = plt.subplots(1, 1, figsize=(Shape[1], Shape[0]))
    Axis.imshow(dArray)
    Axis.imshow(dSeg, alpha=0.25)
    for i in range(len(Xs)):
        Axis.plot([Xs[i,0], Xs[i,1]], [Ys[i,0], Ys[i,0]], color=(1, 0, 0))
        Axis.plot([Xs[i,1], Xs[i,1]], [Ys[i,0], Ys[i,1]], color=(1, 0, 0))
        Axis.plot([Xs[i,1], Xs[i,0]], [Ys[i,1], Ys[i,1]], color=(1, 0, 0))
        Axis.plot([Xs[i,0], Xs[i,0]], [Ys[i,1], Ys[i,0]], color=(1, 0, 0))
    Axis.axis('off')
    plt.subplots_adjust(0, 0, 1, 1)
    plt.savefig(str(ROIName) + '.png')
    plt.close()

    return

def UNet_Prediction(UNet, Image):
            
    Size = UNet.input_shape[1]
    
    NPatches = np.ceil(np.array(Image.shape)[:2] / np.array((Size, Size))).astype(int)
    Pad = (NPatches * np.array((Size, Size)) - np.array(Image.shape)[:2]) // 2
    Padded = np.pad(Image, [[Pad[0],Pad[0]],[Pad[1],Pad[1]],[0,0]], mode='reflect')

    # Separate image into patches to fit UNet
    Step = np.array([256, 256, 3])
    Patches = patchify(Padded, [Size, Size, 3], step=Step)
    Seg = np.zeros(np.concatenate([Padded.shape[:-1], [4]]), float)
    for Xi, Px in enumerate(Patches):
        for Yi, Py in enumerate(Px):
            Pred = UNet.predict(Py / 255, verbose=0)[0]
            X1 = Xi*Step[0]
            X2 = Size + Xi*Step[0]
            Y1 = Yi*Step[1]
            Y2 = Size + Yi*Step[1]
            Seg[X1:X2, Y1:Y2] += Pred

    Norm = np.zeros(Seg.shape)
    for i in range(Padded.shape[0] // Step[0]):
        for j in range(Padded.shape[1] // Step[1]):
            iStart = i * Step[0]
            iStop = iStart + Step[0]
            jStart = j * Step[1]
            jStop = jStart + Step[1]
            Vals = Seg[iStart:iStop,jStart:jStop]
            Norm[iStart:iStop,jStart:jStop] = Vals / np.max(Vals)

    Norm = Norm[Pad[0]:-Pad[0], Pad[1]:-Pad[1]]

    return Norm

#%% Main
# Main part

def Main():

    # Set directory and data
    WD, DD, SD, RD = SetDirectories('FEMHIS')
    RD = RD / 'Segmentation'

    # Load reference ROI and compute color stats
    Ref = io.imread(DD / 'Reference.png')[:,:,:-1]
    LAB = color.rgb2lab(Ref / 255)
    Mean = np.mean(LAB, axis=(0,1))
    Std = np.std(LAB, axis=(0,1), ddof=1)

    # Load Unet and random forest classifier
    RFc = joblib.load(str(DD / 'RandomForest.joblib'))
    Unet = load_model(str(DD / 'UNet.hdf5'))

    # Initialize data frame
    Folders = [F.name for F in DD.iterdir() if F.is_dir()]
    Laterality = ['Right','Left']
    Indices = pd.MultiIndex.from_product([Folders, Laterality])
    # Cols = pd.MultiIndex.from_product([['Number','Area','Std','Density'], ['ROI 1', 'ROI 2', 'ROI 3']])
    # OcData = pd.DataFrame(columns=Cols, index=Indices, dtype=float)
    # HcData = pd.DataFrame(columns=Cols, index=Indices, dtype=float)
    # Cols = pd.MultiIndex.from_product([['Density'], ['ROI 1', 'ROI 2', 'ROI 3']])
    # ClData = pd.DataFrame(columns=Cols, index=Indices, dtype=float)

    OcData = pd.read_csv(RD / 'Osteocytes.csv', header=[0,1], index_col=[0,1])
    OcData.index = Indices
    HcData = pd.read_csv(RD / 'HaversianCanals.csv', header=[0,1], index_col=[0,1])
    HcData.index = Indices
    ClData = pd.read_csv(RD / 'CementLines.csv', header=[0,1], index_col=[0,1])
    ClData.index = Indices

    # Loop for each image 
    N = 3
    for Idx in Indices:

        # Create result directory
        ROIDir = RD / Idx[0]
        os.makedirs(ROIDir, exist_ok=True)

        # Segment images by thresholding to define bone
        IName = str(DD / Idx[0] / (Idx[0] + '_' + Idx[1][0] + '_M100.jpg'))
        Image = sitk.ReadImage(IName)
        Array = sitk.GetArrayFromImage(Image)
        Bone = SegmentBone(Array)

        # Based on bone pixels coordinates, select N ROIs with BV/TV > 0.88
        ROIName = ROIDir / (Idx[0] + '_' + Idx[1])
        ROIs, Xs, Ys = ExtractROIs(Array, Bone, N, ROIName)
        PlotROIs(Bone, Array, ROIName, Xs, Ys)

        # Normalize ROIs colors
        Norms = []
        for I in ROIs:
            LAB = color.rgb2lab(I / 255)
            X_Bar = np.mean(LAB, axis=(0,1))
            S_X = np.std(LAB, axis=(0,1), ddof=1)
            Norm = (LAB - X_Bar) / S_X * Std + Mean
            RGB = color.lab2rgb(Norm)
            RGB = np.round(RGB * 255).astype(int)
            Norms.append(RGB)
        Norms = np.array(Norms)

        # Get Unet prediction
        Prob = []
        Time.Process(1,'Unet prediction')
        for i, I in enumerate(Norms):
            Prob.append(UNet_Prediction(Unet, I))
            Time.Update((i+1) / len(ROIs))
        Time.Process(0)
        Prob = np.array(Prob)

        # Extract features for random forest fit
        Features = []
        Time.Process(1,'Extract features')
        for i, R in enumerate(Norms):
            Feature = feature.multiscale_basic_features(R, channel_axis=-1, sigma_min=2, num_sigma=3)
            Features.append(Feature)
            Time.Update(i / len(ROIs))
        Features = np.array(Features)
        Time.Process(0)

        # Reshape arrays to match random forest
        Prob = Prob.reshape(-1, Prob.shape[-1])
        Features = Features.reshape(-1, Features.shape[-1])
        RF_Feat = np.concatenate([Features, Prob], axis=1)

        # Segment ROIs and store results
        RFc.verbose = 2
        Prediction = RFc.predict(RF_Feat)
        RF_Pred = np.reshape(Prediction, ROIs.shape[:-1])
        for i, (R, P) in enumerate(zip(Norms, RF_Pred)):
            S = R.shape
            Seg = np.zeros((S[0],S[1],4), int)
            Cat = utils.to_categorical(P)
            Seg[:,:,:-1] = Cat[:,:,-3:] * 255
            BG = Cat[:,:,1].astype('bool')
            Seg[:, :,-1][~BG] = 255
            Seg[:,:,1][Seg[:,:,2] == 255] = 255

            IName = str(ROIName) + '_Seg' +str(i) + '.png'
            Figure, Axis = plt.subplots(1,1)
            Axis.imshow(R)
            Axis.imshow(Seg, alpha=0.5)
            Axis.axis('off')
            plt.subplots_adjust(0,0,1,1)
            plt.savefig(IName)
            plt.close(Figure)

            # Compute quantities of interest
            L = measure.label(P == 2)
            RP = pd.DataFrame(measure.regionprops_table(L, properties=['area']))
            OcData.loc[Idx,('Number','ROI ' + str(i+1))] = len(RP)
            OcData.loc[Idx,('Area','ROI ' + str(i+1))] = RP['area'].mean()
            OcData.loc[Idx,('Std','ROI ' + str(i+1))] = RP['area'].std()
            OcData.loc[Idx,('Density','ROI ' + str(i+1))] = np.sum(P==2) / P.size

            L = measure.label(P == 3)
            RP = pd.DataFrame(measure.regionprops_table(L, properties=['area']))
            HcData.loc[Idx,('Number','ROI ' + str(i+1))] = len(RP)
            HcData.loc[Idx,('Area','ROI ' + str(i+1))] = RP['area'].mean()
            HcData.loc[Idx,('Std','ROI ' + str(i+1))] = RP['area'].std()
            HcData.loc[Idx,('Density','ROI ' + str(i+1))] = np.sum(P==3) / P.size

            ClData.loc[Idx,('Density','ROI ' + str(i+1))] = np.sum(P==4) / P.size

    # Save results (ROI, Segmented ROI, quantities) in result directory
    OcData.to_csv(RD / 'Osteocytes.csv')
    HcData.to_csv(RD / 'HaversianCanals.csv')
    ClData.to_csv(RD / 'CementLines.csv')

    # OcData = pd.read_csv(ResultsDir / 'Osteocytes.csv', header=[0,1], index_col=[0,1,2])
    # HcData = pd.read_csv(ResultsDir / 'HaversianCanals.csv', header=[0,1], index_col=[0,1,2])
    # ClData = pd.read_csv(ResultsDir / 'CementLines.csv', header=[0,1], index_col=[0,1,2])


#%% If main
if __name__ == '__main__':

    # Initiate the parser with a description
    Parser = argparse.ArgumentParser(description=Description, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Add long and short argument
    ScriptVersion = Parser.prog + ' version ' + Version
    Parser.add_argument('-v', '--Version', help='Show script version', action='version', version=ScriptVersion)

    # Read arguments from the command line
    Arguments = Parser.parse_args()

    Main()
# %%
