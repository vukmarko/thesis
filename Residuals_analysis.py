"""
Created: 11.04.2024
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from scipy.stats import spearmanr, pearsonr, f, mannwhitneyu
from scipy import stats

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

def exp_function(t, lam):
    return np.exp(-lam*t)

time_classification = 40
pval_cutoff_F = 0.01
pval_cutoffs_t = [0.001, 0.01, 0.05]
R_squared_cutoff = 0.65
unique_cells = np.unique(CRY1_all['cell'])
#unique_cells = np.unique(CRY1_all['cell'][CRY1_all['cell'] == 5631])
list_quality_metrics = []
model_counter = 0
index_counter = 1 # to accomodate add_subplot() limit of subplots
fig = plt.figure(figsize=(6.4*3, 4.8*3))

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
        result = - 4*np.pi/(tau_fit**2) * A_fit * np.cos(2*np.pi*t/tau_fit - phi_fit)
        return result
    phi_hours = phi_fit / (2*np.pi) * tau_fit
    second_derivative_at_phi = calc_second_derivative(phi_hours)
    if second_derivative_at_phi > 0:
        phi_fit = (phi_fit + np.pi) % (2*np.pi)

    # calculate phase at which CHX is applied
    #phi_chx = ((t_CHX - phi_fit / (2 * np.pi) * tau_fit) % tau_fit) * (2 * np.pi / tau_fit)
    phi_chx = ((2 * np.pi / tau_fit) * t_CHX - phi_fit) % (2 * np.pi)
    phi_classification = ((2 * np.pi / tau_fit) * time_classification - phi_fit) % (2 * np.pi)

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
    model_vs_data = Yi_normalized - y_predicted2_normalized

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
    pval = 1 - f.cdf(F, dof_LIN, dof_RHY)
    signif = 'ns' if pval > pval_cutoff_F else '* rhy'

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
        phi_classification = ((2 * np.pi / tau_fit) * time_classification - phi_fit) % (2 * np.pi)

        # values of the fitted "artificial" data
        y_fit2 = nonlinear_decay_model(t_fit, *params2)
        y_fit2_normalized = y_fit2 / c_fit2

        # calculate r-squared value using scikit-learn
        y_predicted2 = nonlinear_decay_model(Xi.flatten(), *params2)
        R_squared2 = r2_score(Yi, y_predicted2)

        Yi_normalized = Yi / c_fit2
        y_predicted2_normalized = y_predicted2 / c_fit2
        model_vs_data = y_predicted2_normalized - Yi_normalized

        # calculate residuals & sum of squared errors for new
        # RHY curve with exponential decay 
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

    classification = 'falling' if (phi_classification < np.pi) else 'rising'

    list_quality_metrics.append([cell, classification, signif, R_squared2, excl, model_vs_data, Xi.flatten()])

        
df = pd.DataFrame(list_quality_metrics, columns=['cell', 'class','rhy_test', 'R^2', 'circadian period?', 'model_vs_data', 'timepoints'])
filtered_df = df[(df['rhy_test'].str.endswith('rhy') | df['rhy_test'].str.endswith('rhy_exp')) &
                (df['R^2'] >= R_squared_cutoff) & 
                (df['circadian period?'] == 'in')]

# filtered_df.to_excel('20240424_Marko_cells.xlsx', index = False)
counts = filtered_df['class'].value_counts()
count_rising = counts['rising']
count_falling = counts['falling']

# Initialize a DataFrame to store the expanded data
expanded_data = pd.DataFrame()

# Iterate over each row in the DataFrame
for index, row in filtered_df.iterrows():
    # Create a dictionary to map timepoints to model_vs_data values
    timepoints_to_values = {timepoint: np.nan for timepoint in range(1, 49)}
    for timepoint, value in zip(row['timepoints'], row['model_vs_data']):
        timepoints_to_values[int(timepoint)] = value
    # Convert the dictionary to a DataFrame and concatenate it with expanded_data
    expanded_data = pd.concat([expanded_data, pd.DataFrame(timepoints_to_values, index=[index])])

# Concatenate the expanded data with the original DataFrame
filtered_df_residuals = pd.concat([filtered_df.drop(columns=['model_vs_data', 'timepoints']), expanded_data], axis=1)

# Plot the violin plot using Seaborn
# Select only columns titled '1' through '48'
selected_columns = [i for i in range(10, 41) if i in filtered_df_residuals.columns]
filtered_df_selected = filtered_df_residuals[selected_columns].copy()
# Add the 'class' column to the selected DataFrame
filtered_df_selected['class'] = filtered_df_residuals['class']
# Melt the DataFrame to long format
melted_df = pd.melt(filtered_df_selected, id_vars=['class'], var_name='Timepoint', value_name='Value')


# t-test
timepoints = melted_df['Timepoint'].unique()

# Create an empty list to store t-test results
mw_test_results = []

# Iterate over each timepoint
for timepoint in timepoints:
    # Filter the DataFrame to include only the current timepoint
    timepoint_melted_df = melted_df[melted_df['Timepoint'] == timepoint]
    
    # Separate the data into rising and falling groups
    rising_values = timepoint_melted_df[timepoint_melted_df['class'] == 'rising']['Value']
    falling_values = timepoint_melted_df[timepoint_melted_df['class'] == 'falling']['Value']
    
    # Perform the t-test
    mw_statistic, mw_p_value = mannwhitneyu(rising_values, falling_values, alternative='two-sided')

    # Highlight timepoints with a significant p-value
    if mw_p_value <= pval_cutoffs_t[0]:
        signif_mw_test = '***'
    elif mw_p_value <= pval_cutoffs_t[1]:
        signif_mw_test = '**'
    elif mw_p_value <= pval_cutoffs_t[2]:
        signif_mw_test = '*'
    else:
        signif_mw_test = 'n.s.'
    
    # Store the results in the list
    mw_test_results.append({'Timepoint': timepoint,
                            'T-statistic': mw_statistic,
                            'P-value': mw_statistic,
                            'Significance': signif_mw_test})
    
# Convert the list of dictionaries to a DataFrame
mw_test_results_df = pd.DataFrame(mw_test_results)


# # Create an empty list to store t-test results
# t_test_results = []

# # Iterate over each timepoint
# for timepoint in timepoints:
#     # Filter the DataFrame to include only the current timepoint
#     timepoint_melted_df = melted_df[melted_df['Timepoint'] == timepoint]
    
#     # Separate the data into rising and falling groups
#     rising_values = timepoint_melted_df[timepoint_melted_df['class'] == 'rising']['Value']
#     falling_values = timepoint_melted_df[timepoint_melted_df['class'] == 'falling']['Value']
    
#     # Perform the t-test
#     t_statistic, p_value = stats.ttest_ind(rising_values, falling_values)

#     # Highlight timepoints with a significant p-value
#     if p_value <= pval_cutoffs_t[0]:
#         signif_t_test = '***'
#     elif p_value <= pval_cutoffs_t[1]:
#         signif_t_test = '**'
#     elif p_value <= pval_cutoffs_t[2]:
#         signif_t_test = '*'
#     else:
#         signif_t_test = 'n.s.'
    
#     # Store the results in the list
#     t_test_results.append({'Timepoint': timepoint,
#                             'T-statistic': t_statistic,
#                             'P-value': p_value,
#                             'Significance': signif_t_test})
    
# # Convert the list of dictionaries to a DataFrame
# t_test_results_df = pd.DataFrame(t_test_results)


### PLOTTING ###
hue_order = ["falling", "rising"]  # This ensures "falling" appears on the left
sb.violinplot(x='Timepoint', y='Value', hue='class', data=melted_df, 
               split=True, inner=None, palette={"rising": "green", "falling": "orange"},
               hue_order=hue_order)
datapoints = sb.stripplot(x='Timepoint', y='Value', hue='class', data=melted_df,
                        dodge=True, palette={"rising": "black", "falling": "black"}, 
                        hue_order=hue_order, size = 1.5, alpha = 1)
handles, labels = datapoints.get_legend_handles_labels() # Get the legend handles and labels
new_labels = [f'falling (n = {count_falling})', f'rising (n = {count_rising})']
plt.legend(handles[0:2], new_labels, fontsize = 16) # Create the legend without the stripplot labels
plt.xlabel('Timepoint [h]', fontsize = 20)
plt.ylabel('Residuals, normalized [a.u.]', fontsize = 20)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.title(f'Classification at timepoint {time_classification}', fontsize = 26)
plt.ylim(-1,1)

# Iterate over t_test_results_df to add significance labels
for index, row in mw_test_results_df.iterrows():
    plt.text(row['Timepoint'] - 10, 0.77, row['Significance'], ha='center', va='bottom', fontsize = 20)

# plt.show()

# Number of cells left after filtering
num_rows, num_columns = filtered_df.shape
print("Both models, cells after filtering:", num_rows)
#print("Model counter:", model_counter)

plt.savefig(f"20240728_CRY1_Residuals_Classification_at_{time_classification}h.png", dpi = 300)