"""
Created: 06.05.2024
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import seaborn as sb
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from scipy.stats import spearmanr, pearsonr, f
from matplotlib.ticker import MultipleLocator

# Loading the data20240728_CRY1_Decay_Timeseries.py
CRY1_all = pd.read_excel('./CRY1_mClover_shNS1_shxT.xlsx')
CRY1_goodcells = pd.read_excel('./20240728_Good_Cells.xlsx')
CRY1_background_cells = pd.read_excel('./CRY1_mScarlet_shNS1_chxT_YFP.xlsx')

# Calculating the median background
CRY1_bg_intensities = CRY1_background_cells.iloc[:, 13:80]
CRY1_timepoints_median_bg = CRY1_bg_intensities.median(axis=0) # median background at each timepoint
CRY1_median_bg = np.median(CRY1_timepoints_median_bg) # median background from all timepoints
CRY1_median_bg_SD = np.std(CRY1_timepoints_median_bg) # standard deviation of the background

YFP = {'1':0.009625519349432576,'2':0.009300955415797372, '3':0.008673694099934993} #Photobleaching constants

### List of cells
single_cell = np.unique(CRY1_all['cell'][CRY1_all['cell'] == 5625])
unique_goodcells = np.unique(CRY1_goodcells['cell'])

### Models
# Linear regression
def line(t, slope, intercept):
    return slope * t + intercept

# Define the exponential decay model
def decay_without_offset(t, y0, lam):
    return y0 * np.exp(-lam * t)

# Define the exponential decay model, with offset
def decay_with_offset(t, y0, lam, C):
    return y0 * np.exp(-lam * t) + C

# Define the harmonic model
def harmonic_model(t, a, b, c):
    return a * np.sin(t) + b * np.cos(t) + c

### Cutoffs and such
R_squared_cutoff = 0.95
half_life_cutoff = 72
pval_F_test_cutoff = 0.01
background_cutoff = CRY1_median_bg + 2 * CRY1_median_bg_SD
starting_timepoint = 2
end_timepoint = 8
metrics_list = []
figure_counter = 0
index_counter = 0 
cell_counter = 0

for cell in unique_goodcells:
    CRY1_i = CRY1_all[CRY1_all['cell'] == cell]
    CRY1_i_rhythmicity = CRY1_goodcells[CRY1_goodcells['cell'] == cell]
    t_CHX = CRY1_i.tp_treatment.values[0] #timepoint of chx addition: 44 or 48
    expt = CRY1_i.exp.values[0] # Experiment number: 1, 2, or 3
    bg_christian = CRY1_i.neg_cell_int_chx.values[0] # Christian's background
    christian_half_life = CRY1_i.half_life.values[0] # Half-life estimate from Christian's analysis
    phi_chx = CRY1_i_rhythmicity.phi_chx.values[0] # Phase (in radians) at t_CHX. From our rhythmicity analysis
    intensities_i = CRY1_i.iloc[:, 13:80]
    intensities_transposed_i = intensities_i.T
    intensities_bg_sub_i = intensities_transposed_i.sub(CRY1_timepoints_median_bg, axis = 0) # Background subtraction
    # intensities_bg_sub_i = intensities_transposed_i - bg_christian
    rhythmic_i = intensities_bg_sub_i.iloc[0:(t_CHX-1),:]
    decay_i = intensities_bg_sub_i.iloc[(t_CHX-1):67,:] #decay data
    Xi_nan = np.arange(len(decay_i.index)).reshape(-1, 1) #with nans
    Yi_nan = decay_i.iloc[:, 0].values #Dep. variable, with nans     
    Yi_uncorrected = Yi_nan[~np.isnan(Yi_nan)] #remove nans

    ### Does the intensity at timepoint 2 hours after CHX addition clearly exceed background? If not, move onto next cell
    if Yi_uncorrected[2] <= background_cutoff:
        continue
    
    ### Photobleaching correction
    photobleaching_constant = YFP[str(expt)]
    timepoints = np.arange(0,len(Yi_uncorrected))
    Yi_corrected = Yi_uncorrected * np.exp(timepoints * photobleaching_constant) # photobleaching-corrected intensities
    Xi_removed_nan = Xi_nan[~np.isnan(Yi_nan)] #remove nans

    ### Taking only timepoints starting at 2 hours post-CHX (time of CHX addition = timepoint zero)
    Yi = Yi_corrected[starting_timepoint:]
    ln_Yi = np.log(Yi) # ln transformation
    Xi = Xi_removed_nan[starting_timepoint:] - starting_timepoint # Timepoint 2 becomes zero. Necessary for the exp. decay function, which starts at 0

    ### Linear regression on ln-transformed data
    ln_params, ln_cov = curve_fit(line, Xi.flatten(), ln_Yi, p0=(0.1, 0.1), maxfev=100000)
    ln_slope, ln_intercept = ln_params
    ln_slope_abs = abs(ln_slope)
    ln_intercept_converted = np.exp(ln_intercept)
    t_fit = np.linspace(min(Xi), max(Xi), 1000) # generate time data
    ln_y_fitline = line(t_fit, *ln_params)

    ### Exclude cells with a positive decay slope
    params_line, cov_line = curve_fit(line, Xi.flatten(), Yi, p0=(0.1, 0.1), maxfev=100000)
    slope_fit, intercept_fit = params_line
    if slope_fit > 0: 
        continue

    ### Fit the exponential decay model without offset
    params_decay_without_offset, cov_decay_without_offset = curve_fit(decay_without_offset, Xi.flatten(), Yi,  p0 = (ln_intercept_converted, ln_slope_abs), maxfev=100000)
    y0_fit1, lam_fit1 = params_decay_without_offset
    decay_without_offset_fit = decay_without_offset(t_fit, *params_decay_without_offset)

    ### Fit the exponentialy decay model with offset to the non-transformed data
    params_decay_with_offset, cov_decay_with_offset = curve_fit(decay_with_offset, Xi.flatten(), Yi, p0 = (ln_intercept_converted, ln_slope_abs, background_cutoff), maxfev=100000)
    y0_fit2, lam_fit2, c_fit2 = params_decay_with_offset
    decay_with_offset_fit = decay_with_offset(t_fit, *params_decay_with_offset)

    ### Calculate the half-lives
    half_life_decay_without_offset = np.log(2)/lam_fit1
    half_life_decay_with_offset = np.log(2)/lam_fit2

    if half_life_decay_with_offset > half_life_cutoff: # exclude cells with a half life greater than 3 days
        continue

    ### Calculate the R-squared values
    y_predicted_decay_without_offset = decay_without_offset(Xi.flatten(), *params_decay_without_offset)
    R_squared_decay_without_offset = r2_score(Yi, y_predicted_decay_without_offset)
    y_predicted_decay_with_offset = decay_with_offset(Xi.flatten(), *params_decay_with_offset)
    R_squared_decay_with_offset = r2_score(Yi, y_predicted_decay_with_offset)

    if R_squared_decay_with_offset < R_squared_cutoff: # exclude cells with a poor R-squared
        continue

    metrics_list.append([cell, half_life_decay_with_offset, lam_fit2, christian_half_life, phi_chx])
    
    # Plot the cells individually
    index_counter += 1
    # Create a new figure if the current subplot is the first one
    if index_counter % 12 == 1:
        fig = plt.figure(figsize=(6.4*3, 4.8*4))
        figure_counter += 1
    # Plot the data points
    subplot_index = (index_counter - 1) % 12 + 1
    ax = fig.add_subplot(4, 3, subplot_index)
    ax.scatter(Xi+2, Yi, color = "black", s = 12, alpha= 0.8)
    #ax.plot(t_fit, y_fitline, color = "gray", linestyle='--', lw=1) # Line fit
    ax.plot(t_fit+2, decay_without_offset_fit, color = "blue", linewidth = 1.5,
            label = f"$R^2 = {R_squared_decay_without_offset:.2f}$, t₁/₂ = {half_life_decay_without_offset:.2f} h") # Exponential decay
    ax.plot(t_fit+2, decay_with_offset_fit, color = "red", linewidth = 1.5, 
            label = f"$R^2 = {R_squared_decay_with_offset:.2f}$, t₁/₂ = {half_life_decay_with_offset:.2f} h") # Exponential decay with offset
    ax.set_title(f"Cell {cell}", fontsize=20)
    ax.set_xlabel("Time after CHX addition [h]", fontsize=20)
    ax.set_ylabel("Intensity [a.u.]", fontsize=20)
    ax.tick_params(axis='both', labelsize=16)
    ax.legend(framealpha=0, fontsize=16)
    ax.xaxis.set_major_locator(MultipleLocator(4))
    # Get the current tick positions and ensure the first tick is at 2
    ticks = ax.get_xticks()
    ticks = np.arange(2, ticks[-2], 4)
    ax.set_xticks(ticks)

    # If the maximum number of plots is reached. save the figure
    if index_counter % 12 == 0:
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.5)
        plt.savefig(f'20240728_CRY1_Decay_Fig{figure_counter}.png', dpi = 300)
        plt.close()  # Close the current figure to release memory

# Save the last figure if it's not yet saved
if (index_counter - 1) % 12 != 0:
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.5)
    plt.savefig(f'20240728_CRY1_Decay_Fig{figure_counter}.png', dpi = 300)