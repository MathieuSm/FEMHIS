#%% !/usr/bin/env python3
# Initialization

Version = '01'

# Define the script description
Description = """
    This script perform statistical analysis of
    automatic segmentation results

    Author: Mathieu Simon
            ARTORG Center for Biomedical Engineering Research
            SITEM Insel
            University of Bern

    Date: September 2023
    """

#%% Imports
# Modules import

import os
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import statsmodels.formula.api as smf
from Utils import SetDirectories, Show
from scipy.stats.distributions import t
from matplotlib.colors import ListedColormap


#%% Functions
# Define functions

Show.ShowPlot = False

#%% Main
# Main part

def Main():

    # Set directory and data
    WD, DD, SD, RD = SetDirectories('FEMHIS')
    SD = RD / 'Statistics'
    RD = RD / 'Segmentation'

    # Load automatic segmentation data
    Samples = pd.read_csv(DD / 'SampleList.csv')
    FEMCOL = pd.read_csv(DD / 'FEMCOL.csv')
    Osteocytes = pd.read_csv(RD / 'Osteocytes.csv', header=[0,1], index_col=[0,1])
    Haversian = pd.read_csv(RD / 'HaversianCanals.csv', header=[0,1], index_col=[0,1])
    CementLines = pd.read_csv(RD / 'CementLines.csv', header=[0,1], index_col=[0,1])

    # Modify FEMCOL names and values
    Cols = FEMCOL.columns.values
    Cols[1] = 'Age / year'

    FEMCOL[Cols[4]] = FEMCOL[Cols[4]] / 1E3
    Cols[4] = 'Stiffness Mineralized / kN/mm'

    FEMCOL[Cols[12]] = FEMCOL[Cols[12]] * 1E2
    Cols[12] = 'Ultimate Strain / %'

    FEMCOL[Cols[19]] = FEMCOL[Cols[19]] * 1E2
    Cols[19] = 'Mineral weight fraction / %'

    FEMCOL[Cols[20]] = FEMCOL[Cols[20]] * 1E2
    Cols[20] = 'Organic weight fraction / %'

    FEMCOL[Cols[21]] = FEMCOL[Cols[21]] * 1E2
    Cols[21] = 'Water weight fraction / %'

    FEMCOL[Cols[22]] = FEMCOL[Cols[22]] * 1E2
    Cols[22] = 'Bone Volume Fraction / %'

    FEMCOL.columns = Cols

    # Build data frame
    C1, C2 = Samples.columns[:-1], FEMCOL.columns[4:]
    Cols = np.concatenate([C1, C2])
    Data = pd.DataFrame(columns=Cols, index=Samples.index)
    SID = Osteocytes.reset_index()['level_0'].unique()
    for i, R in Samples.iterrows():
        S, L, A, G = R[C1]
        Loc = str(S) + L

        if L == 'L':
            L = 'Left'
        elif L == 'R':
            L = 'Right'

        if A == 'XX':
            A = 0
            G = np.nan

        A = int(A)

        for j, v in enumerate([S, L, A, G]):
            Data.loc[i, C1[j]] = v

        if sum(FEMCOL['Sample ID'] == Loc) > 0:
            FC_Data = FEMCOL[FEMCOL['Sample ID'] == Loc]
            for C in C2:
                Data.loc[i,C] = FC_Data[C].values[0]

        if S in SID:
            Data.loc[i,'Osteocytes'] = np.mean(Osteocytes.loc[(S, L),'Density'])
            Data.loc[i,'Haversian'] = np.mean(Haversian.loc[(S, L),'Density'])
            Data.loc[i,'CementLines'] = np.mean(CementLines.loc[(S, L),'Density'])

    Data = Data.dropna()

    # Compute correlations and p-values
    sData = Data[Data.columns[4:]].astype(float)
    sData['Age / year'] = Data['Age'].astype(int)
    Corr = sData.corr(method=lambda x, y: pearsonr(x, y)[0])
    Pvalues = sData.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(len(sData.columns)) 

    N = 256
    Values = np.ones((N,4))
    Values[:int(N/2.5),1] = np.linspace(0, 1, int(N/2.5))
    Values[:int(N/2.5),2] = np.linspace(0, 1, int(N/2.5))
    Values[-int(N/2.5):,0] = np.linspace(1, 0, int(N/2.5))
    Values[-int(N/2.5):,1] = np.linspace(1, 0, int(N/2.5))
    CMP = ListedColormap(Values)

    Figure, Axis = plt.subplots(1,1, figsize=(9,12))
    Im = Axis.matshow(Corr, vmin=-1, vmax=1, cmap=CMP)
    Axis.xaxis.set_ticks_position('bottom')
    Axis.set_xticks(np.arange(len(Corr)))
    Axis.set_xticklabels(Corr.columns, rotation=45, ha='right')
    Axis.set_yticks(np.arange(len(Corr)))
    Axis.set_yticklabels(Corr.columns)
    Cb = plt.colorbar(Im, fraction=0.046, pad=0.04)
    Figure.savefig(SD / 'Correlations.png', dpi=Figure.dpi, bbox_inches='tight')
    plt.close(Figure)

    Cat = Pvalues.copy()
    Cat[Cat >= 0.05] = 4
    Cat[Cat < 0.001] = 1
    Cat[Cat < 0.01] = 2
    Cat[Cat < 0.05] = 3

    Figure, Axis = plt.subplots(1,1, figsize=(9,12))
    Im = Axis.matshow(Cat, vmin=1, vmax=4, cmap='viridis_r')
    Axis.xaxis.set_ticks_position('bottom')
    Axis.set_xticks(np.arange(len(Corr)))
    Axis.set_xticklabels(Corr.columns, rotation=45, ha='right')
    Axis.set_yticks(np.arange(len(Corr)))
    Axis.set_yticklabels(Corr.columns)
    Cb = plt.colorbar(Im, ticks=[1, 2, 3, 4], fraction=0.046, pad=0.04)
    Cb.ax.set_yticklabels(['<0.001', '<0.01', '<0.05','$\geq$0.05'])
    Figure.savefig(SD / 'Pvalues.png', dpi=Figure.dpi, bbox_inches='tight')
    plt.close(Figure)

    # Get significant variables
    F = Cat[['Osteocytes','Haversian']] < 4
    Osteocytes_Sig = F.index[F['Osteocytes']]
    Haversian_Sig = F.index[F['Haversian']]

    F = Cat['CementLines'] < 4
    CementLines_Sig = F.index[F]

    DataList = ['Osteocytes','Haversian','CementLines']
    Osteocytes_Sig = Osteocytes_Sig.drop(DataList, errors='ignore')
    Haversian_Sig = Haversian_Sig.drop(DataList, errors='ignore')
    CementLines_Sig = CementLines_Sig.drop(DataList, errors='ignore')

    # Add FEMCOL results
    for i, R in FEMCOL.iterrows():
        S, L = R[['Sample ID', 'Site']]

        if L == 'L':
            L = 'Left'
        elif L == 'R':
            L = 'Right'

        S = int(S[:-1])

        for C in Osteocytes_Sig:
            Osteocytes.loc[(S, L),C] = R[C]
        
        for C in Haversian_Sig:
            Haversian.loc[(S, L),C] = R[C]

        for C in CementLines_Sig:
            CementLines.loc[(S, L),C] = R[C]

    Osteocytes = Osteocytes.dropna()
    Haversian = Haversian.dropna()
    CementLines = CementLines.dropna()

    for i, Tissue in enumerate([Osteocytes, Haversian, CementLines]):

        # Create folder to store tissue results
        Folder = SD / DataList[i]
        os.makedirs(Folder, exist_ok=True)

        # Create data frame to analyse
        Yv = 'Density'
        Show.FName = str(Folder / 'Density_Hist.png')
        Show.Histogram(Tissue[Yv].values.ravel() * 1E2,
                       Density=True, Norm=True,
                       Labels=['Density (%)','Relative count (-)'])
        Show.FName = str(Folder / 'Density_QQ.png')
        Show.QQPlot(Tissue[Yv].values.ravel() * 1E2)

        if i == 2:
            C = Tissue.columns.drop('Density')
        else:
            C = Tissue.columns.drop(['Density','Number','Area','Std'],level=0)

        for Xv, _ in C:

            XSplit = Xv.split(' / ')
            XLabel = XSplit[0] + ' ('
            for Split in XSplit[1:]:
                XLabel += Split.split()[0] + '/'
            XLabel = XLabel[:-1] + ')'

            DataFrame = Tissue[[Xv, Yv]].copy()
            DataFrame = DataFrame.astype('float')
            X = DataFrame[Xv].values
            X = np.repeat(X, 3)
            Y = DataFrame[Yv].values.ravel() * 1E2
            Data = pd.DataFrame(np.vstack([X,Y]).T,columns=['X','Y'])

            # Add donor group
            Group = DataFrame.reset_index()['level_0'].values
            Group = np.repeat(Group,3)
            Data['Donor'] = Group

            # Add sample group
            Group = DataFrame.reset_index()['level_1'].values
            Group = np.repeat(Group,3)
            Data['Sample'] = Group

            # # Linear fixed effects model
            # OLS = Show.OLS(X, Y, Labels=[Xv,Yv])

            # # Linear mixed-effect model
            # LME = smf.mixedlm('Y ~ X',
            #                   data=Data, groups=Data['Donor']
            #                   ).fit(reml=False, method=['lbfgs'])
            # print(LME.summary())

            # # Investigate sample effect
            # LME = smf.mixedlm('Y ~ X',
            #                   data=Data, groups=Data['Sample']
            #                   ).fit(reml=False, method=['lbfgs'])
            # print(LME.summary())

            # 2 Levels LME
            LME = smf.mixedlm('Y ~ X',
                            data=Data, groups=Data['Donor'],
                            re_formula='1', vc_formula={'Sample': '0 + Sample'}
                            ).fit(reml=False, method=['lbfgs'])
            
            if LME.pvalues[1] < 0.05:

                # Check data normality assumptions
                FName = Xv.split(' / ')[0].replace(' ','_')
                FName = FName.replace('/','')
                Show.FName = str(Folder / (FName + '_Hist.png'))
                Show.Histogram(X,Density=True, Norm=True,
                            Labels=[XLabel,'Relative count (-)'])
                Show.FName = str(Folder / (FName + '_QQ.png'))
                Show.QQPlot(X)

                with open(str(Folder / (FName + '_Table.tex')), 'w') as F:
                    F.write(LME.summary().as_latex())

                # Plot 2 levels LME
                Show.FName = str(Folder / (FName + '_LME.png'))
                Show.MixedLM(Data, LME, Xlabel=XLabel, Ylabel='Density (%)')

                # Check random effects assumptions
                RE = pd.DataFrame(LME.random_effects).T
                RE.columns = ['Group','Left','Right']
                Show.FName = str(Folder / (FName + '_RE.png'))
                Show.BoxPlot([RE['Group'], RE['Left'].dropna(), RE['Right'].dropna()],
                            SetsLabels=['Donor','Left','Right'],
                            Labels=['','Random Effects'])

                Show.FName = str(Folder / (FName + '_Group_QQ.png'))
                Show.QQPlot(RE['Group'].values)
                Show.FName = str(Folder / (FName + '_Left_QQ.png'))
                Show.QQPlot(RE['Left'].dropna().values)
                Show.FName = str(Folder / (FName + '_Right_QQ.png'))
                Show.QQPlot(RE['Right'].dropna().values)

                # Check residuals
                Show.FName = str(Folder / (FName + '_Residuals.png'))
                Show.BoxPlot([LME.resid], Labels=['','Residuals'])
                Show.FName = str(Folder / (FName + '_Residuals_QQ.png'))
                Show.QQPlot(LME.resid.values)


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
