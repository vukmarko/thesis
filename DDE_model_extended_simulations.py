import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator
from scipy.signal import find_peaks
from ddeint import ddeint 
from scipy.interpolate import interp1d

# Parameters extracted from the half-life vs phase plot. (my estimates of background, phase, and half-life)
a_fit = 0.03109224
b_fit = 0.0032573
c_fit = 0.13536525
rhy_deg_amp = np.sqrt(a_fit**2 + b_fit**2)
center_deg_rate = c_fit
max_deg_rate = center_deg_rate + (rhy_deg_amp/2)
min_deg_rate = center_deg_rate - (rhy_deg_amp/2)

def relative_amplitude(result):
    # Ensure the input is a 1-D array
    if isinstance(result, np.ndarray) and result.ndim == 2 and result.shape[1] == 1:
        result = result.flatten()
    elif not isinstance(result, np.ndarray) or result.ndim != 1:
        raise ValueError("Input must be a 1-D numpy array or a 2-D numpy array with a single column.")
    
    # Find the indices of the peaks
    peaks, _ = find_peaks(result)
    
    if len(peaks) < 2:
        raise ValueError("There are fewer than two peaks in the data.")
    
    # Initialize a list to store the relative amplitudes for each segment
    relative_amplitudes = []
    
    # Iterate through consecutive peaks
    for i in range(len(peaks) - 1):
        first_peak = peaks[i]
        second_peak = peaks[i + 1]
        
        # Extract the values between the two peaks
        values_between_peaks = result[first_peak:second_peak+1]
        
        # Calculate the mean of the values between the peaks
        mean_between_peaks = np.mean(values_between_peaks)
        
        # Calculate the mean of the maximum and the minimum values within this segment
        max_val = np.max(values_between_peaks)
        min_val = np.min(values_between_peaks)
        
        # Compute the relative amplitude for this segment
        rel_amplitude = (max_val - min_val) / mean_between_peaks
        relative_amplitudes.append(rel_amplitude)
    
    # Compute the overall relative amplitude (you can use mean, median, etc.)
    overall_relative_amplitude = np.mean(relative_amplitudes)
    
    return overall_relative_amplitude

def calculate_period(result, time):
    # Ensure the input is a 1-D array
    if isinstance(result, np.ndarray) and result.ndim == 2 and result.shape[1] == 1:
        result = result.flatten()
    elif not isinstance(result, np.ndarray) or result.ndim != 1:
        raise ValueError("Input must be a 1-D numpy array or a 2-D numpy array with a single column.")
    
    # Find the indices of the peaks
    peaks, _ = find_peaks(result)
    
    if len(peaks) < 2:
        raise ValueError("There are fewer than two peaks in the data.")
    
    # Use the first two peaks for simplicity
    #first_peak = peaks[0]
    #second_peak = peaks[1]
    #period = time[second_peak] - time[first_peak]

    # calculate mean and std of the peak-to-peak distance
    time_peaks = time[peaks]
    period = np.diff(time_peaks).mean()
    period_std = np.diff(time_peaks).std()    
    
    return [period, period_std]

def calculate_average_deg_rate(d_values, result_asymp, t_values, t_asymp):
    # Ensure result_asymp is 1-D
    result_asymp = np.squeeze(result_asymp)
    # Find peaks in result_asymp
    peaks, _ = find_peaks(result_asymp)

    # Select indices of the last 4 peaks
    last_4_peaks_indices = peaks[-4:]

    # Get corresponding time values from t_asymp
    t_values_last_4_peaks = t_asymp[last_4_peaks_indices]

    # Calculate the time limits for the last 3 oscillations
    t_start = t_values_last_4_peaks[-3]
    t_end = t_values_last_4_peaks[-1]

    # Find indices of t_values within t_value limits
    indices = [i for i, t in enumerate(t_values) if t >= t_start and t <= t_end]

    # Extract d_values within the time limits
    d_values_last_3_oscillations = [d_values[i] for i in indices]

    # Calculate the average of d_values last 3 oscillations
    average_d_values = np.mean(d_values_last_3_oscillations)

    return average_d_values

def normalize_oscillations(data):
    # Ensure the data is a 1-D array
    if isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] == 1:
        data = data.flatten()
    elif not isinstance(data, np.ndarray) or data.ndim != 1:
        raise ValueError("Input must be a 1-D numpy array or a 2-D numpy array with a single column.")
    
    # Find peaks in the data
    peaks, _ = find_peaks(data)
    
    # If no peaks are found or less than 2 peaks, return the original data
    if len(peaks) < 2:
        return data
    
    # Calculate mean between the peaks
    means = []
    for i in range(len(peaks) - 1):
        segment = data[peaks[i]:peaks[i + 1]]
        mean_segment = np.mean(segment)
        means.append(mean_segment)
    
    # Average mean between peaks
    mean_between_peaks = np.mean(means)
    
    # Normalize data by dividing by the mean between peaks
    normalized_data = data / mean_between_peaks

    return normalized_data

def sort_deg_rates(time, data):
    # Ensure time and data are numpy arrays
    time = np.array(time)
    data = np.array(data)
    
    # Get the sorted indices of the time array
    sorted_indices = np.argsort(time)
    
    # Use the sorted indices to sort the data arrays
    sorted_data = data[sorted_indices]
    sorted_time = time[sorted_indices]

    return sorted_time, sorted_data

# Define the function representing the DDE system
def delay_model_const_deg(y, t, parameters):
    c = parameters['c']
    d1 = parameters['d1']
    d2 = parameters['d2']
    ck = parameters['ck']
    tau = parameters['tau']

    x = y(t)
    x_delayed = y(t - tau)

    # Approximate the slope dx/dt
    increment = 1  # A time increment for numerical differentiation
    slope = (y(t) - y(t - increment)) / increment

    # Choose d based on the slope
    if slope > 0:
        d = d1
    elif slope < 0:
        d = d2
    else:
        d = center_deg_rate

    t_values1.append(t)
    d_values1.append(d)

    dxdt = (c / (ck + x_delayed)) ** 3 - d * x

    return dxdt

def delay_model_rhy_deg(y, t, parameters):
    c = parameters['c']
    d1 = parameters['d1']
    d2 = parameters['d2']
    ck = parameters['ck']
    tau = parameters['tau']

    x = y(t)
    x_delayed = y(t - tau)

    # Approximate the slope dx/dt
    increment = 1  # A time increment for numerical differentiation
    slope = (y(t) - y(t - increment)) / increment

    # Choose d based on the slope
    if slope < 0:
        d = d1
    elif slope > 0:
        d = d2
    else:
        d = center_deg_rate

    t_values2.append(t)
    d_values2.append(d)
 
    dxdt = (c / (ck + x_delayed)) ** 3 - d * x

    return dxdt

# Define the initial condition function
def history(t):
    return 1

# Define the time interval for the DDE solver
t0 = 0
tfinal = 1000
dt = 0.05
t = np.arange(t0, tfinal, dt)

# Generate non-linear degradation rate values using a power function
num_values = 30
power = 2  # Adjust the power to control the distribution
linear_space = np.linspace(0, 1, num=num_values) # Generate linearly spaced values between 0 and 1
transformed_space = linear_space ** power # Apply a power transformation to cluster values more towards the start
d1_values = center_deg_rate - (center_deg_rate - min_deg_rate) * transformed_space # Scale the transformed values to the desidarkgray range
d2_values = center_deg_rate + (max_deg_rate - center_deg_rate) * transformed_space 
d_range_x = np.linspace(1, num_values, num_values)

# Empty lists necesary for the for-loop
d_values1 = []
t_values1 = []
d_values2 = []
t_values2 = []
relative_amplitudes1 = []
relative_amplitudes2 = []
period_values1 = []
period_values2 = []
period_std_values1, period_std_values2 = [], []
d_differences1 = []
d_differences2 = []
d_averages_case1 = []
d_averages_case2 = []

# Loop to vary d1 and d2
for index in range(len(d1_values)):
    d1 = d1_values[index]
    d2 = d2_values[index]

    parameters = {
        'c': 1,
        'ck': 0.1,
        'd1': d1,
        'd2': d2,
        'tau': 7.58
    }

    # Call the ddeint function
    result1 = ddeint(delay_model_const_deg, history, t, fargs=(parameters,))
    result2 = ddeint(delay_model_rhy_deg, history, t, fargs=(parameters,))

    # Remove transient behavior, keep only asymptotic dynamics
    last_osc = 100  # keep last 100h 
    time_points = int(last_osc / dt)  # Calculate the number of time points that correspond to the last oscillation period
    result_asymp1 = result1[-time_points:, :]  # extract the result from the last 100 h
    result_asymp2 = result2[-time_points:, :]

    # Normalize all oscillations to their mean
    result_asymp1 = normalize_oscillations(result_asymp1)
    result_asymp2 = normalize_oscillations(result_asymp2) 

    # Get the relative amplitude
    rel_amplitude1 = relative_amplitude(result_asymp1)
    relative_amplitudes1.append(rel_amplitude1)
    d_differences1.append(d2 - d1)
    rel_amplitude2 = relative_amplitude(result_asymp2)
    relative_amplitudes2.append(rel_amplitude2)
    d_differences2.append(d2 - d1)
  
    # Get the period values
    t_asymp = t[-time_points:] #extract the time points from the last 100 h
    # t_asymp = np.arange(0, last_osc, dt) #start from 0
    period_1 = calculate_period(result_asymp1, t_asymp)[0]
    period_1_std = calculate_period(result_asymp1, t_asymp)[1]
    period_values1.append(period_1)
    period_std_values1.append(period_1_std)

    period_2 = calculate_period(result_asymp2, t_asymp)[0]
    period_2_std = calculate_period(result_asymp2, t_asymp)[1]
    period_values2.append(period_2)
    period_std_values2.append(period_2_std)

    ### Get the average degradation rate
    # interpolate d_values and keep datapoints for last 100h
    print(f'running simulation {index+1}/{len(d1_values)}')
    d_values1_interp = interp1d(t_values1, d_values1, kind='nearest', 
                                fill_value='extrapolate')
    d_values1_aligned = d_values1_interp(t)
    d1_last_100_hours = d_values1_aligned[-time_points:]
    d_average1 = calculate_average_deg_rate(
        d1_last_100_hours, result_asymp1, t_asymp, t_asymp)
    d_averages_case1.append(d_average1)   

    d_values2_interp = interp1d(t_values2, d_values2, kind='nearest', 
                                fill_value='extrapolate')
    d_values2_aligned = d_values2_interp(t)
    d2_last_100_hours = d_values2_aligned[-time_points:]
    d_average2 = calculate_average_deg_rate(
        d2_last_100_hours, result_asymp2, t_asymp, t_asymp)
    d_averages_case2.append(d_average2)      

    # # Extract the degradation rates and the corresponding times from the last 100 hours
    # t1_num_elements = len(t_values1) // 10
    # t1_last_100_hours = t_values1[-t1_num_elements:]
    # d1_num_elements = len(d_values1) // 10
    # d1_last_100_hours = d_values1[-d1_num_elements:]
    # d_average1 = calculate_average_deg_rate(d1_last_100_hours, result_asymp1, t1_last_100_hours, t_asymp)
    # d_averages_case1.append(d_average1)

    # t2_num_elements = len(t_values2) // 10
    # t2_last_100_hours = t_values2[-t2_num_elements:]
    # d2_num_elements = len(d_values2) // 10
    # d2_last_100_hours = d_values2[-d2_num_elements:]
    # d_average2 = calculate_average_deg_rate(d2_last_100_hours, result_asymp2, t2_last_100_hours, t_asymp)
    # d_averages_case2.append(d_average2)

relative_amplitudes_percent1 = (relative_amplitudes1 - relative_amplitudes1[0]) / relative_amplitudes1[0] * 100
relative_amplitudes_percent2 = (relative_amplitudes2 - relative_amplitudes2[0]) / relative_amplitudes2[0] * 100
period_percent1 = (period_values1 - period_values1[0]) / period_values1[0] * 100
period_percent2 = (period_values2 - period_values2[0]) / period_values2[0] * 100
d_differences_array1 = np.array(d_differences1)
d_differences_fraction1 = d_differences_array1 / center_deg_rate
d_differences_array2 = np.array(d_differences2)
d_differences_fraction2 = d_differences_array2 / center_deg_rate
period_values1, period_std_values1 = np.asarray(period_values1), np.asarray(period_std_values1)
period_values2, period_std_values2 = np.asarray(period_values2), np.asarray(period_std_values2)
d_averages_case1 = np.asarray(d_averages_case1)
d_averages_case2 = np.asarray(d_averages_case2)

### PLOTTING
threshold_for_LC = 0.025 # std of peak-to-peak period (h)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 5.5))


simulation_number_d1 = np.asarray(range(1, len(d1_values) + 1))
simulation_number_d2 = np.asarray(range(1, len(d2_values) + 1))
ax1.plot(simulation_number_d1, d1_values, 'o-', markersize = 3, color = '#bfbfbf', label = r'$d_{min}$')
ax1.plot(simulation_number_d2, d2_values, 'o-', markersize = 3, color = 'black', label = r'$d_{max}$')
ax1.plot(simulation_number_d1[period_std_values1 < threshold_for_LC], d_averages_case1[period_std_values1 < threshold_for_LC], 
         'o-', ms = 3, color = 'dodgerblue', label = r'$\overline{d}$, case 1')
ax1.plot(simulation_number_d2[period_std_values2 < threshold_for_LC], d_averages_case2[period_std_values2 < threshold_for_LC], 
         'o-', ms = 3, color = 'crimson', label = r'$\overline{d}$, case 2')
ax1.set_ylabel('Degradation rate [h⁻¹]', fontsize = 14)
ax1.set_xlabel('Simulation #', fontsize = 14)
ax1.set_xticks([1, 5, 10, 15, 20, 25, 30])
ax1.tick_params(axis='x', labelsize=14)
ax1.tick_params(axis='y', labelsize=14)
ax1.set_title('Degradation rate', fontsize = 14)
ax1.legend(loc='upper left', fontsize = 14)
#ax1.plot(d1_values, color = 'orange', label = 'd1')
#ax1.plot(d2_values, color = 'purple', label = 'd2')
#ax1.plot(d_averages_case1, linestyle = '-', color = 'black', label = 'Mean deg. rate, case 1')
## ax1.plot(d_averages_case2, linestyle = ':', color = 'black', label = 'Mean deg. rate, case 2')
#ax1.set_ylabel('Degradation rate')
#ax1.set_xlabel('Simulation')
#ax1.set_title('Degradation rate')
#ax1.legend()

ax2.plot(d_differences_fraction1[period_std_values1 < threshold_for_LC], 
         relative_amplitudes_percent1[period_std_values1 < threshold_for_LC],  
         'o-', markersize = 3, color = 'dodgerblue', label = 'Case 1')
ax2.plot(d_differences_fraction2[period_std_values2 < threshold_for_LC], 
         relative_amplitudes_percent2[period_std_values2 < threshold_for_LC], 
         'o-', markersize = 3, color = 'crimson', label = 'Case 2')
ax2.set_xlabel(r'(d$_{max}$ - d$_{min}$) / d$_{center}$', fontsize = 16)
ax2.set_ylabel('Change in rel. amplitude [%]', fontsize = 14)
ax2.set_title('Relative amplitude', fontsize = 14)
ax2.set_ylim(-32, 32)
ax2.legend(loc='upper left', fontsize = 14)
ax2.tick_params(axis='x', labelsize=14)
ax2.tick_params(axis='y', labelsize=14)
#ax2.plot(d_differences_fraction1, relative_amplitudes_percent1, linestyle = '-', color = 'black', label = 'Case 1')
#ax2.plot(d_differences_fraction2, relative_amplitudes_percent2, linestyle = ':', color = 'black', label = 'Case 2')
#ax2.set_xlabel('(d$_{2}$ - d$_{1}$) / d$_{center}$')
#ax2.set_ylabel('Change in relative amplitude [%]')
#ax2.set_title('Relative amplitude')
#ax2.legend()

ax3.plot(d_differences_fraction1[period_std_values1 < threshold_for_LC], 
         period_values1[period_std_values1 < threshold_for_LC], 
         'o-', ms = 3, color = 'dodgerblue', label = 'Case 1')
ax3.plot(d_differences_fraction2[period_std_values2 < threshold_for_LC], 
         period_values2[period_std_values2 < threshold_for_LC], 
         'o-', ms = 3, color = 'crimson', label = 'Case 2')
ax3.set_xlabel(r'(d$_{max}$ - d$_{min}$) / d$_{center}$', fontsize = 16)
ax3.set_ylabel('Period [h]', fontsize = 14)
ax3.set_title('Period', fontsize = 14)
ax3.set_ylim(24.2, 26)
ax3.legend(loc='upper left', fontsize = 14)
ax3.tick_params(axis='x', labelsize=14)
ax3.tick_params(axis='y', labelsize=14)
#ax3.plot(d_differences_fraction1, period_values1, linestyle = '-', color = 'black', label = 'Case 1')
#ax3.plot(d_differences_fraction2, period_values2, linestyle = ':', color = 'black', label = 'Case 2')
#ax3.set_xlabel('(d$_{2}$ - d$_{1}$) / d$_{center}$')
#ax3.set_ylabel('Period [h]')
#ax3.set_title('Period')
#ax3.legend()

fig.tight_layout()
fig.subplots_adjust(wspace=0.27)
plt.show()

# if dt == 0.05:
#     name = '005dt'
# elif dt == 0.01:
#     name = '001dt'
# elif dt == 0.10:
#     name = '010dt'

# fig.savefig(f'20240625_{name}_rel_amp_and_period_cases1and2_{num_values}simulations.png', dpi = 300)
# fig.savefig(f'20240625_{name}_rel_amp_and_period_cases1and2_{num_values}simulations.pdf', format='pdf')

# plt.savefig('20240728_005dt_30simulations_new_delay.png', dpi = 300)