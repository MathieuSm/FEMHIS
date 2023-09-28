#%% #!/usr/bin/env python3
# Initialization

Version = '01'

Description = """
    This script aims to provide utility functions for the histology
    part of the FEXHIP project. Most functions are taken from different
    sources and adapted here
    
    Author: Mathieu Simon
            ARTORG Center for Biomedical Engineering Research
            SITEM Insel
            University of Bern

    Date: May 2023
    """

#%% Imports
# Modules import

import time
import numpy as np
import pandas as pd
import SimpleITK as sitk
from pathlib import Path
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from matplotlib.colors import ListedColormap
from scipy.stats.distributions import t, norm
# from mpl_toolkits.axes_grid1 import make_axes_locatable


#%% Functions
# Define some general functions

def SetDirectories(Name):

    CWD = str(Path.cwd())
    Start = CWD.find(Name)
    WD = Path(CWD[:Start], Name)
    Data = WD / '01_Data'
    Scripts = WD / '02_Scripts'
    Results = WD / '03_Results'

    return WD, Data, Scripts, Results

#%% Time functions
class Time():

    def __init__(self):
        self.Width = 15
        self.Length = 16
        self.Text = 'Process'
        self.Tic = time.time()
        pass
    
    def Set(self, Tic=None):
        
        if Tic == None:
            self.Tic = time.time()
        else:
            self.Tic = Tic

    def Print(self, Tic=None,  Toc=None):

        """
        Print elapsed time in seconds to time in HH:MM:SS format
        :param Tic: Actual time at the beginning of the process
        :param Toc: Actual time at the end of the process
        """

        if Tic == None:
            Tic = self.Tic
            
        if Toc == None:
            Toc = time.time()


        Delta = Toc - Tic

        Hours = np.floor(Delta / 60 / 60)
        Minutes = np.floor(Delta / 60) - 60 * Hours
        Seconds = Delta - 60 * Minutes - 60 * 60 * Hours

        print('\nProcess executed in %02i:%02i:%02i (HH:MM:SS)' % (Hours, Minutes, Seconds))

        return

    def Update(self, Progress, Text=''):

        Percent = int(round(Progress * 100))
        Np = self.Width * Percent // 100
        Nb = self.Width - Np

        if len(Text) == 0:
            Text = self.Text
        else:
            self.Text = Text

        Ns = self.Length - len(Text)
        if Ns >= 0:
            Text += Ns*' '
        else:
            Text = Text[:self.Length]
        
        Line = '\r' + Text + ' [' + Np*'=' + Nb*' ' + ']' + f' {Percent:.0f}%'
        print(Line, sep='', end='', flush=True)

    def Process(self, StartStop:bool, Text=''):

        if len(Text) == 0:
            Text = self.Text
        else:
            self.Text = Text

        if StartStop*1 == 1:
            print('')
            self.Tic = time.time()
            self.Update(0, Text)

        elif StartStop*1 == 0:
            self.Update(1, Text)
            self.Print()

Time = Time()
#%% Ploting functions
class Show():

    def __init__(self):
        self.FName = None
        self.ShowPlot = True

    def Slice(self, Image, Slice=None, Title=None, Axis='Z'):

        try:
            Array = sitk.GetArrayFromImage(Image)
            Dimension = Image.GetDimension()
        except:
            Array = Image
            Dimension = len(Array.shape)

        if Dimension == 3:
            
            if Axis == 'Z':
                if Slice:
                    Array = Array[Slice,:,:]
                else:
                    Array = Array[Array.shape[0]//2,:,:]
            if Axis == 'Y':
                if Slice:
                    Array = Array[:,Slice,:]
                else:
                    Array = Array[:,Array.shape[1]//2,:]
            if Axis == 'X':
                if Slice:
                    Array = Array[:,:,Slice]
                else:
                    Array = Array[:,:,Array.shape[2]//2]

        Figure, Axis = plt.subplots()
        Axis.imshow(Array,interpolation=None, cmap='binary_r')
        Axis.axis('Off')
        
        if (Title):
            Axis.set_title(Title)

        if (self.FName):
            plt.savefig(self.FName, bbox_inches='tight', pad_inches=0)

        if self.ShowPlot:
            plt.show()
        else:
            plt.close()

        return

    def BoxPlot(self, ArraysList, Labels=['', 'Y'], SetsLabels=None, Vertical=True):

        Width = 2.5 + len(ArraysList)
        Figure, Axis = plt.subplots(1,1, dpi=99, figsize=(Width,4.5))

        for i, Array in enumerate(ArraysList):
            RandPos = np.random.normal(i,0.02,len(Array))

            Axis.boxplot(Array, vert=Vertical, widths=0.35,
                        showmeans=False,meanline=True,
                        showfliers=False, positions=[i],
                        capprops=dict(color=(0,0,0)),
                        boxprops=dict(color=(0,0,0)),
                        whiskerprops=dict(color=(0,0,0),linestyle='--'),
                        medianprops=dict(color=(0,0,1)),
                        meanprops=dict(color=(0,1,0)))
            Axis.plot(RandPos - RandPos.mean() + i, Array, linestyle='none',
                      marker='o',fillstyle='none', color=(1,0,0), ms=5)
        
        Axis.plot([],linestyle='none',marker='o',fillstyle='none', color=(1,0,0), label='Data')
        Axis.plot([],color=(0,0,1), label='Median')
        Axis.set_xlabel(Labels[0])
        Axis.set_ylabel(Labels[1])

        if SetsLabels:
            Axis.set_xticks(np.arange(len(SetsLabels)))
            Axis.set_xticklabels(SetsLabels, rotation=0)
        else:
            Axis.set_xticks([])
        
        plt.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.125))
        plt.subplots_adjust(left=0.25, right=0.75)
        
        if (self.FName):
            plt.savefig(self.FName, bbox_inches='tight', pad_inches=0.02)
        if self.ShowPlot:
            plt.show()
        else:
            plt.close()

        return

    def Mask(self, Image, Mask, Label=''):

        Mask = Mask * 1.0
        Values = np.unique(Mask)
        Mask[Mask == Values[0]] = np.nan

        Figure, Axis = plt.subplots(1,1)
        Axis.imshow(Image)

        if len(Values) > 2:
            N = 256
            CValues = np.zeros((N, 4))
            CValues[:, 0] = np.linspace(0, 1, N)
            CValues[:, 1] = np.linspace(1, 0, N)
            CValues[:, 2] = np.linspace(1, 0, N)
            CValues[:, -1] = np.linspace(1.0, 1.0, N)
            CMP = ListedColormap(CValues)
            Plot = Axis.imshow(Mask, cmap=CMP, alpha=0.5)
            CBarAxis = Figure.add_axes([0.225, 0.08, 0.575, 0.025])
            plt.colorbar(Plot, cax=CBarAxis, orientation='horizontal', label=Label)

        else:
            Axis.imshow(Mask, cmap='brg_r', alpha=0.5)
        
        Axis.axis('off')
        plt.show()


        return

    def OLS(self, X, Y, Cmap=np.array(None), Labels=None, Alpha=0.95, Annotate=['N','R2','SE','Slope','Intercept']):

        if Labels == None:
            Labels = ['X', 'Y']
        
        # Perform linear regression
        Array = np.array([X,Y])
        if Array.shape[0] == 2:
            Array = Array.T
        Data = pd.DataFrame(Array,columns=['X','Y'])
        FitResults = smf.ols('Y ~ X', data=Data).fit()
        Slope = FitResults.params[1]

        # Build arrays and matrices
        Y_Obs = FitResults.model.endog
        Y_Fit = FitResults.fittedvalues
        N = int(FitResults.nobs)
        C = np.matrix(FitResults.normalized_cov_params)
        X = np.matrix(FitResults.model.exog)

        # Sort X values and Y accordingly
        Sort = np.argsort(np.array(X[:,1]).reshape(len(X)))
        X_Obs = np.sort(np.array(X[:,1]).reshape(len(X)))
        Y_Fit = Y_Fit[Sort]
        Y_Obs = Y_Obs[Sort]

        ## Compute R2 and standard error of the estimate
        E = Y_Obs - Y_Fit
        RSS = np.sum(E ** 2)
        SE = np.sqrt(RSS / FitResults.df_resid)
        TSS = np.sum((FitResults.model.endog - FitResults.model.endog.mean()) ** 2)
        RegSS = TSS - RSS
        R2 = RegSS / TSS
        R2adj = 1 - RSS/TSS * (N-1)/(N-X.shape[1]+1-1)

        ## Compute CI lines
        B_0 = np.sqrt(np.diag(np.abs(X * C * X.T)))
        t_Alpha = t.interval(Alpha, N - X.shape[1] - 1)
        CI_Line_u = Y_Fit + t_Alpha[0] * SE * B_0[Sort]
        CI_Line_o = Y_Fit + t_Alpha[1] * SE * B_0[Sort]

        ## Plots
        DPI = 100
        Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=DPI, sharey=True, sharex=True)

        if Cmap.any():
            Colors = plt.cm.winter((Cmap-min(Cmap))/(max(Cmap)-min(Cmap)))
            Scatter = Axes.scatter(X_Obs, Y_Obs, facecolor='none', edgecolor=Colors, marker='o',)
        else:
            Axes.plot(X_Obs, Y_Obs, linestyle='none', marker='o', color=(0,0,1), fillstyle='none')

        Axes.plot(X_Obs, Y_Fit, color=(1,0,0))
        Axes.fill_between(X_Obs, CI_Line_o, CI_Line_u, color=(0, 0, 0), alpha=0.1)

        if Slope > 0:

            YPos = 0.925
            if 'N' in Annotate:
                Axes.annotate(r'$N$  : ' + str(N), xy=(0.025, YPos), xycoords='axes fraction')
                YPos -= 0.075
            if 'R2' in Annotate:
                Axes.annotate(r'$R^2$ : ' + format(round(R2, 2), '.2f'), xy=(0.025, YPos), xycoords='axes fraction')
                YPos -= 0.075
            if 'SE' in Annotate:
                Axes.annotate(r'$SE$ : ' + format(round(SE, 2), '.2f'), xy=(0.025, YPos), xycoords='axes fraction')
            
            YPos = 0.025
            if 'Intercept' in Annotate:
                Intercept = str(FitResults.params[0])
                Round = 3 - Intercept.find('.')
                Intercept = round(FitResults.params[0], Round)
                CI = FitResults.conf_int().loc['Intercept'].round(Round)
                if Round <= 0:
                    Intercept = int(Intercept)
                    CI = [int(v) for v in CI]
                Text = r'Intercept : ' + str(Intercept) + ' (' + str(CI[0]) + ',' + str(CI[1]) + ')'
                Axes.annotate(Text, xy=(0.425, YPos), xycoords='axes fraction')
                YPos += 0.075

            if 'Slope' in Annotate:
                Round = 3 - str(FitResults.params[1]).find('.')
                Slope = round(FitResults.params[1], Round)
                CI = FitResults.conf_int().loc['X'].round(Round)
                if Round <= 0:
                    Slope = int(Slope)
                    CI = [int(v) for v in CI]
                Text = r'Slope : ' + str(Slope) + ' (' + str(CI[0]) + ',' + str(CI[1]) + ')'
                Axes.annotate(Text, xy=(0.425, YPos), xycoords='axes fraction')

        elif Slope < 0:

            YPos = 0.025
            if 'N' in Annotate:
                Axes.annotate(r'$N$  : ' + str(N), xy=(0.025, YPos), xycoords='axes fraction')
                YPos += 0.075
            if 'R2' in Annotate:
                Axes.annotate(r'$R^2$ : ' + format(round(R2, 2), '.2f'), xy=(0.025, YPos), xycoords='axes fraction')
                YPos += 0.075
            if 'SE' in Annotate:
                Axes.annotate(r'$SE$ : ' + format(round(SE, 2), '.2f'), xy=(0.025, YPos), xycoords='axes fraction')
            
            YPos = 0.925
            if 'Intercept' in Annotate:
                Intercept = str(FitResults.params[0])
                Round = 3 - Intercept.find('.')
                Intercept = round(FitResults.params[0], Round)
                CI = FitResults.conf_int().loc['Intercept'].round(Round)
                if Round <= 0:
                    Intercept = int(Intercept)
                    CI = [int(v) for v in CI]
                Text = r'Intercept : ' + str(Intercept) + ' (' + str(CI[0]) + ',' + str(CI[1]) + ')'
                Axes.annotate(Text, xy=(0.425, YPos), xycoords='axes fraction')
                YPos -= 0.075

            if 'Slope' in Annotate:
                Round = 3 - str(FitResults.params[1]).find('.')
                Slope = round(FitResults.params[1], Round)
                CI = FitResults.conf_int().loc['X'].round(Round)
                if Round <= 0:
                    Slope = int(Slope)
                    CI = [int(v) for v in CI]
                Text = r'Slope : ' + str(Slope) + ' (' + str(CI[0]) + ',' + str(CI[1]) + ')'
                Axes.annotate(Text, xy=(0.425, YPos), xycoords='axes fraction')
        
        Axes.set_xlabel(Labels[0])
        Axes.set_ylabel(Labels[1])
        plt.subplots_adjust(left=0.15, bottom=0.15)

        if (self.FName):
            plt.savefig(self.FName, bbox_inches='tight', pad_inches=0.02)
        if self.ShowPlot:
            plt.show()
        else:
            plt.close()

        return FitResults

    def Histogram(self, Array:np.array, Labels=[], Density=False, Norm=False, Bins=20):

        # Compute data values
        X = pd.DataFrame(Array)
        SortedValues = np.sort(X.T.values)[0]
        N = len(X)
        X_Bar = X.mean()
        S_X = np.std(X, ddof=1)

        # Figure plotting
        Figure, Axes = plt.subplots(1, 1)

        # Histogram
        Histogram, Edges = np.histogram(X, bins=Bins, density=True)
        Width = 0.9 * (Edges[1] - Edges[0])
        Center = (Edges[:-1] + Edges[1:]) / 2
        Axes.bar(Center, Histogram, align='center', width=Width,
                 edgecolor=(0,0,0), color=(1, 1, 1, 0), label='Histogram')

        # Density distribution
        if Density and N < 1E3:
            KernelEstimator = np.zeros(N)
            NormalIQR = np.sum(np.abs(norm.ppf(np.array([0.25, 0.75]), 0, 1)))
            DataIQR = np.abs(X.quantile(0.75)) - np.abs(X.quantile(0.25))
            KernelHalfWidth = 0.9 * N ** (-1 / 5) * min(np.abs([S_X, DataIQR / NormalIQR]))
            for Value in SortedValues:
                KernelEstimator += norm.pdf(SortedValues - Value, 0, KernelHalfWidth * 2)
            KernelEstimator = KernelEstimator / N
        
            Axes.plot(SortedValues, KernelEstimator, color=(1,0,0), label='Kernel density')
        
        # Corresponding normal distribution
        if Norm:
            TheoreticalDistribution = norm.pdf(SortedValues, X_Bar, S_X)
            Axes.plot(SortedValues, TheoreticalDistribution, linestyle='--',
                      color=(0,0,1), label='Normal distribution')
        
        if len(Labels) > 0:
            plt.xlabel(Labels[0])
            plt.ylabel(Labels[1])

        plt.legend(loc='best')

        if (self.FName):
            plt.savefig(self.FName, bbox_inches='tight', pad_inches=0.02)
        if self.ShowPlot:
            plt.show()
        else:
            plt.close(Figure)

    def QQPlot(self, DataValues, Alpha_CI=0.95):

        ### Based on: https://www.tjmahr.com/quantile-quantile-plots-from-scratch/
        ### Itself based on Fox book: Fox, J. (2015)
        ### Applied Regression Analysis and Generalized Linear Models.
        ### Sage Publications, Thousand Oaks, California.

        # Data analysis
        N = len(DataValues)
        X_Bar = np.mean(DataValues)
        S_X = np.std(DataValues,ddof=1)

        # Sort data to get the rank
        Data_Sorted = np.zeros(N)
        Data_Sorted += DataValues
        Data_Sorted.sort()

        # Compute quantiles
        EmpiricalQuantiles = np.arange(0.5, N + 0.5) / N
        TheoreticalQuantiles = norm.ppf(EmpiricalQuantiles, X_Bar, S_X)
        ZQuantiles = norm.ppf(EmpiricalQuantiles,0,1)

        # Compute data variance
        DataIQR = np.quantile(DataValues, 0.75) - np.quantile(DataValues, 0.25)
        NormalIQR = np.sum(np.abs(norm.ppf(np.array([0.25, 0.75]), 0, 1)))
        Variance = DataIQR / NormalIQR
        Z_Space = np.linspace(min(ZQuantiles), max(ZQuantiles), 100)
        Variance_Line = Z_Space * Variance + np.median(DataValues)

        # Compute alpha confidence interval (CI)
        Z_SE = np.sqrt(norm.cdf(Z_Space) * (1 - norm.cdf(Z_Space)) / N) / norm.pdf(Z_Space)
        Data_SE = Z_SE * Variance
        Z_CI_Quantile = norm.ppf(np.array([(1 - Alpha_CI) / 2]), 0, 1)

        # Create point in the data space
        Data_Space = np.linspace(min(TheoreticalQuantiles), max(TheoreticalQuantiles), 100)

        # QQPlot
        BorderSpace = max( 0.05*abs(Data_Sorted.min()), 0.05*abs(Data_Sorted.max()))
        Y_Min = Data_Sorted.min() - BorderSpace
        Y_Max = Data_Sorted.max() + BorderSpace
        Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
        Axes.plot(Data_Space, Variance_Line, linestyle='--', color=(1, 0, 0), label='Variance :' + str(format(np.round(Variance, 2),'.2f')))
        Axes.plot(Data_Space, Variance_Line + Z_CI_Quantile * Data_SE, linestyle='--', color=(0, 0, 1), label=str(int(100*Alpha_CI)) + '% CI')
        Axes.plot(Data_Space, Variance_Line - Z_CI_Quantile * Data_SE, linestyle='--', color=(0, 0, 1))
        Axes.plot(TheoreticalQuantiles, Data_Sorted, linestyle='none', marker='o', mew=0.5, fillstyle='none', color=(0, 0, 0))
        plt.xlabel('Theoretical quantiles (-)')
        plt.ylabel('Empirical quantiles (-)')
        plt.ylim([Y_Min, Y_Max])
        plt.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.15), prop={'size':10})
        plt.subplots_adjust(left=0.15, bottom=0.15)

        if (self.FName):
            plt.savefig(self.FName, bbox_inches='tight', pad_inches=0.02)
        if self.ShowPlot:
            plt.show()
        else:
            plt.close(Figure)

        return Variance

    def MixedLM(self, Data:pd.DataFrame, LME:smf.mixedlm, Alpha_CI=0.95, Xlabel='X', Ylabel='Y'):
        """
        Function used to plot mixed linear model results
        Plotting based on: https://www.azandisresearch.com/2022/12/31/visualize-mixed-effect-regressions-in-r-with-ggplot2/
        As bootstrap is expensive for CI band computation, compute
        CI bands based on FOX 2017

        Only implemented for 2 levels LME with nested random intercepts
        """

        # Compute conditional residuals
        Data['CR'] = LME.params[0] + Data['X']*LME.params[1] + LME.resid

        Min = Data['X'].min()
        Max = Data['X'].max()
        Range = np.linspace(Min, Max, len(Data))

        Y_Fit = LME.params[0] + Range * LME.params[1]
        Alpha = t.interval(Alpha_CI, len(Data) - len(LME.fe_params) - 1)

        RSS = np.sum(LME.resid ** 2)
        SE = np.sqrt(RSS / LME.df_resid)

        C = np.matrix(LME.cov_params())
        X = np.matrix([np.ones(len(Data)),np.linspace(Min, Max, len(Data))]).T
        
        if C.shape[0] > len(LME.fe_params):
            C = C[:len(LME.fe_params),:len(LME.fe_params)]

        B_0 = np.sqrt(np.diag(np.abs(X * C * X.T)))

        CI_Line_u = Y_Fit + Alpha[0] * SE * B_0
        CI_Line_o = Y_Fit + Alpha[1] * SE * B_0

        
        Figure, Axis = plt.subplots(1,2, sharex=True, sharey=True, figsize=(10,5))
        Axis[0].scatter(Data['X'], Data['Y'], c=Data['Donor'],
                        marker='o', cmap='winter')
        Axis[0].fill_between(np.linspace(Min, Max, len(Data)), CI_Line_o, CI_Line_u, color=(0,0,0,0.25), edgecolor='none')
        Axis[0].plot(Range, Y_Fit, color=(1,0,0))
        Axis[1].scatter(Data['X'], Data['CR'], c=Data['Donor'],
                        marker='o', cmap='winter')
        Axis[1].fill_between(np.linspace(Min, Max, len(Data)), CI_Line_o, CI_Line_u, color=(0,0,0,0.25), edgecolor='none')
        Axis[1].plot(Range, Y_Fit, color=(1,0,0))
        Axis[0].set_ylabel(Ylabel)
        for i in range(2):
            Axis[i].set_xlabel(Xlabel)
        Axis[0].set_title('Raw Data')
        Axis[1].set_title('Conditional Residuals')
        
        if (self.FName):
            plt.savefig(self.FName, bbox_inches='tight', pad_inches=0.02)
        if self.ShowPlot:
            plt.show()
        else:
            plt.close()

        return

Show = Show()
# %%