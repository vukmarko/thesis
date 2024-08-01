"""
Created: 07.03.2024
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from scipy.stats import f
from matplotlib.ticker import MaxNLocator, FixedLocator

# Loading the data
CRY1_all = pd.read_excel('./CRY1_mClover_shNS1_shxT.xlsx')

# Define the linear regression model
def linear_model(t, a, b, c):
    return a * np.sin(2 * np.pi * t / 24) + \
        b * np.cos(2 * np.pi * t/ 24) + c

# Define the nonlinear regression model
# (with/without decay)
def nonlinear_decay_model(t, A, tau, phi, c, lam):
    return (A * np.cos(2 * np.pi / tau * t - phi) + c) * np.exp(-lam*t)
def nonlinear_model(t, A, tau, phi, c):
    return A * np.cos(2 * np.pi / tau * t - phi) + c

def line(t, intercept):
    return intercept

R_squared_cutoff = 0.65
pval_cutoff_F = 0.01
unique_cells = np.unique(CRY1_all['cell'])
representative_cell_values = [5625, 5626, 5631, 5638, 5641, 5642, 5650, 5653, 5656]
representative_cells = np.unique(CRY1_all['cell'][np.isin(CRY1_all['cell'], representative_cell_values)])
list_quality_metrics = []
figure_counter = 0
index_counter = 0 # to accomodate add_subplot() limit of subplots

for cell in unique_cells:
    CRY1_i = CRY1_all[CRY1_all['cell'] == cell]
    t_CHX = CRY1_i.tp_treatment.values[0] #timepoint of chx addition
    intensities_i = CRY1_i.iloc[:, 13:80]
    intensities_transposed_i = intensities_i.T
    rhythmic_i = intensities_transposed_i.iloc[0:(t_CHX-1),:]
    decay_i = intensities_transposed_i.iloc[(t_CHX-1):67,:] #decay data

    Xi_nan = np.arange(len(rhythmic_i.index)).reshape(-1, 1) #with nans
    Yi_nan = rhythmic_i.iloc[:, 0].values #Dep. variable, with nans     
    Yi = Yi_nan[~np.isnan(Yi_nan)] #remove nans
    Xi = Xi_nan[~np.isnan(Yi_nan)] + 1 #remove nans and make the timepoints start at 1, not 0

    # fit linear model
    params, cov = curve_fit(linear_model, Xi.flatten(), Yi)
    a_fit, b_fit, c_fit = params
    amplitude = np.sqrt(a_fit**2 + b_fit**2)
    phase = np.arctan2(a_fit,b_fit) #arctan2() works better than arctan()
    initial = [amplitude, 24, phase, c_fit]

    # fit nonlinear model
    params2, cov2 = curve_fit(nonlinear_model, Xi.flatten(), Yi, 
                              p0 = initial, maxfev=100000)
    A_fit, tau_fit, phi_fit, c_fit2 = params2

    if tau_fit > 32:
        tau_fit2 = 18
        initial = [amplitude, tau_fit2, phase, c_fit]
        params2, cov2 = curve_fit(nonlinear_model, Xi.flatten(), Yi, 
                                  p0 = initial, maxfev=100000)

    if tau_fit < 12:
        tau_fit2 = 20
        initial = [amplitude, tau_fit2, phase, c_fit]
        params2, cov2 = curve_fit(nonlinear_model, Xi.flatten(), Yi, 
                                  p0 = initial, maxfev=100000)
    
    A_fit, tau_fit, phi_fit, c_fit2 = params2

    phi_fit = phi_fit % (2*np.pi) # Limit the phase to the (0,2pi) range

    # Check if the extracted phase is at the trough or at the peak
    # Second derivative is positive at the trough of a cosine function and negative at the peak
    def calc_second_derivative(t):
        result = - 4*(np.pi**2)/(tau_fit**2) * A_fit * np.cos(2*np.pi*t/tau_fit - phi_fit)
        return result
    phi_hours = phi_fit / (2*np.pi) * tau_fit
    second_derivative_at_phi = calc_second_derivative(phi_hours)
    if second_derivative_at_phi > 0:
        phi_fit = (phi_fit + np.pi) % (2*np.pi)

    # calculate phase at which CHX is applied
    #phi_chx = ((t_CHX - phi_fit / (2 * np.pi) * tau_fit) % tau_fit) * (2 * np.pi / tau_fit)
    phi_chx = ((2 * np.pi / tau_fit) * t_CHX - phi_fit) % (2 * np.pi)

    # fit a line to data (H0)
    params_line, cov_line = curve_fit(line, Xi.flatten(), Yi, 
                                      p0=(0.1), maxfev=100000)
    
    # Generate values for the fitted curve
    t_fit = np.linspace(min(Xi), max(Xi), 1000) # generate time data
    y_fit1 = linear_model(t_fit, *params)
    y_fit2 = nonlinear_model(t_fit, *params2)
    y_fit2_normalized = y_fit2 / c_fit2
    y_fitline = line(t_fit, *params_line)
    y_fitline_predicted = line(Xi.flatten(), *params_line)

    # Calculate R-squared value using scikit-learn
    y_predicted1 = linear_model(Xi.flatten(), *params)
    R_squared1 = r2_score(Yi, y_predicted1)
    y_predicted2 = nonlinear_model(Xi.flatten(), *params2)
    R_squared2 = r2_score(Yi, y_predicted2)

    # Model vs data
    # For timepoints 1 to 44, calculate the difference between the model and the data
    y_predicted2_normalized = y_predicted2 / c_fit2
    Yi_normalized = Yi / c_fit2
    model_vs_data = y_predicted2_normalized - Yi_normalized

    # calculate residuals & sum of squared errors for RHY and H0
    res_RHY = Yi - y_predicted2 #rhythmic model
    SSE_RHY = np.sum(res_RHY**2)
    res_LIN = Yi - y_fitline_predicted #nonrhythmic model (simpler model, H0)
    SSE_LIN = np.sum(res_LIN**2)

    # p value associated with F statistic
    dof_RHY = len(Yi) - len(params2) #degrees of freedom
    dof_LIN = len(Yi) - len(params_line)
    F = ((SSE_LIN - SSE_RHY) / (dof_LIN - dof_RHY)) / \
            (SSE_RHY/ dof_RHY)
    pval_model1 = 1 - f.cdf(F, dof_LIN, dof_RHY)
    signif = 'ns' if pval_model1 > pval_cutoff_F else '* rhy'

    # if cell is classified as nonrhythmic, try to fit an
    # exponentially decaying/increasing function (to account 
    # for the trend) to see if some of the nonrhythmic cells
    # can be "rescued"
    if signif == "ns" or (tau_fit > 32 or tau_fit < 12):
        # fit nonlinear model with exp. decay
        initial = [amplitude, 24, phase, c_fit, 0.1]
        params2, cov2 = curve_fit(nonlinear_decay_model, Xi.flatten(), 
                                  Yi, p0 = initial, maxfev=100000)
        A_fit, tau_fit, phi_fit, c_fit2, lam_fit = params2

        phi_fit = phi_fit % (2*np.pi) # Limit the phase to the (0,2pi) range

        # Check if the extracted phase is at the trough or at the peak
        # The second derivative is positive at the trough and negative at the peak
        def calc_second_derivative_exp(t):
            term1 = np.exp(-lam_fit * t)
            term2 = lam_fit * (4 * np.pi / tau_fit) * A_fit * np.sin(2 * np.pi * t / tau_fit - phi_fit)
            term3 = (lam_fit ** 2) * (A_fit * np.cos(2 * np.pi * t / tau_fit - phi_fit) + c_fit2)
            term4 = (4 * (np.pi ** 2)) / (tau_fit ** 2) * A_fit * np.cos(2 * np.pi * t / tau_fit - phi_fit)
    
            result = term1 * (term2 + term3 - term4)
            return result
        phi_hours = phi_fit / (2*np.pi) * tau_fit
        second_derivative_at_phi = calc_second_derivative_exp(phi_hours)
        if second_derivative_at_phi > 0:
            phi_fit = (phi_fit + np.pi) % (2*np.pi)

        # calculate phase at which CHX is applied
        phi_chx = ((2 * np.pi / tau_fit) * t_CHX - phi_fit) % (2 * np.pi)

        # values of the fitted "artificial" data
        y_fit2 = nonlinear_decay_model(t_fit, *params2)

        # calculate r-squared value using scikit-learn
        y_predicted2 = nonlinear_decay_model(Xi.flatten(), *params2)
        R_squared2 = r2_score(Yi, y_predicted2)

        # calculate residuals & sum of squared errors for new
        # RHY curve with exponential decay. Also for null hypothesis
        res_RHY = Yi - y_predicted2 #rhythmic model
        SSE_RHY = np.sum(res_RHY**2)

        # p value associated with F statistic
        dof_RHY = len(Yi) - len(params2) #degrees of freedom
        F = ((SSE_LIN - SSE_RHY) / (dof_LIN - dof_RHY)) / \
                (SSE_RHY/ dof_RHY)
        pval = 1 - f.cdf(F, dof_LIN, dof_RHY)
        signif = 'ns' if pval > pval_cutoff_F else '* rhy_exp'

    # exclude cell from analysis if period is too far away from
    # the circadian period (anything with tau<12 or tau>32)

    excl = 'excl' if (tau_fit < 16) or (tau_fit > 30) else 'in'

    classification = 'falling' if (phi_chx < np.pi) else 'rising'

    amplitude = abs(A_fit)

    relative_amplitude = abs(A_fit/c_fit2)

    list_quality_metrics.append([cell, classification, signif, R_squared2, excl, tau_fit, relative_amplitude, phi_fit])
    
        # Filtering and plotting cells meeting the criteria
    if (excl == 'in') and ((signif == "* rhy") or (signif == "* rhy_exp")) and (R_squared2 >= R_squared_cutoff):
        index_counter += 1
        # Create a new figure if the current subplot is the first one
        if index_counter % 12 == 1:
            fig = plt.figure(figsize=(6.4*3, 4.8*4))
            figure_counter += 1
        # Plot the data points
        subplot_index = (index_counter - 1) % 12 + 1
        ax = fig.add_subplot(4, 3, subplot_index)
        ax.scatter(Xi.flatten(), Yi, color = "black", s = 12)
        # Plot the fitted curve
        # ax.plot(t_fit, y_fitline, color = "gray", linestyle='--', lw=1)
        ax.set_title(f"Cell {cell} \n $\phi_{{CHX}}$ classification: {classification}", fontsize=20)
        ax.set_xlabel("Time [h]", fontsize=20)
        ax.set_ylabel("Intensity [a.u.]", fontsize=20)
        ax.tick_params(axis='both', labelsize=16)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        min_x_value = min(Xi.flatten())
        if min_x_value == 1:
            # Use MaxNLocator to determine the tick locations
            locator = MaxNLocator(nbins=6)
            ax.xaxis.set_major_locator(locator)
            # Get the tick locations and manually adjust the first tick
            ticks = ax.get_xticks()
            ticks = [tick for tick in ticks if tick >= 1]  # Remove ticks less than 1
            ticks.insert(0, 1)  # Ensure the first tick is 1
            # Set the new tick locations
            ax.xaxis.set_major_locator(FixedLocator(ticks))
        else:
            # Use the default tick locator
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ### Plot only the rhythmic cells

        if (signif == '* rhy'):
            ax.plot(t_fit, y_fit2, color = "blue", 
            label = f"$R^2 = {R_squared2:.2f}$, τ = {tau_fit:.1f}h")
        else:
            ax.plot(t_fit, y_fit2, color = "red", 
            label = f"$R^2 = {R_squared2:.2f}$, τ = {tau_fit:.1f}h")

        ax.legend(framealpha=0, fontsize=16)

         # If the maximum number of plots is reached. save the figure
        if index_counter % 12 == 0:
            fig.tight_layout()
            fig.subplots_adjust(hspace=0.5)
            plt.savefig(f'20240728_CRY1_model_coloring_Fig{figure_counter}.png', dpi = 300)
            plt.close()  # Close the current figure to release memory

# Save the last figure if it's not yet saved
if (index_counter - 1) % 12 != 0:
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.5)
    plt.savefig(f'20240728_CRY1_model_coloring_Fig{figure_counter}.png', dpi = 300)
        
        
df = pd.DataFrame(list_quality_metrics, columns=['cell', 'class','rhy_test', 'R^2', 'circadian period?', 'period', 'relative amplitude', 'phase'])
filtered_df = df[((df['rhy_test'].str.endswith('rhy')) | (df['rhy_test'].str.endswith('rhy_exp'))) &
                (df['R^2'] >= R_squared_cutoff) & 
                (df['circadian period?'] == 'in')]

constant_model_amplitudes = filtered_df[filtered_df['rhy_test'] == '* rhy']['relative amplitude']
exp_model_amplitudes = filtered_df[filtered_df['rhy_test'] == '* rhy_exp']['relative amplitude']
constant_model_periods = filtered_df[filtered_df['rhy_test'] == '* rhy']['period']
exp_model_periods = filtered_df[filtered_df['rhy_test'] == '* rhy_exp']['period']
constant_model_phase = filtered_df[filtered_df['rhy_test'] == '* rhy']['phase']
exp_model_phase = filtered_df[filtered_df['rhy_test'] == '* rhy_exp']['phase']

# plt.show()

# Number of cells left after filtering
#print(filtered_df)
num_rows, num_columns = filtered_df.shape
print("Cells after filtering:", num_rows)