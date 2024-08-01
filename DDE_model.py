import numpy as np
import matplotlib.pyplot as plt
from ddeint import ddeint
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

# Parameters extracted from the half-life vs phase plot. (my estimates of background, phase, and half-life)
a_fit = 0.03109224
b_fit = 0.0032573
c_fit = 0.13536525
rhy_deg_amp = np.sqrt(a_fit**2 + b_fit**2)
center_deg_rate = c_fit
max_deg_rate = center_deg_rate + (rhy_deg_amp/2)
min_deg_rate = center_deg_rate - (rhy_deg_amp/2)

d_values1 = []
t_values1 = []
d_values2 = []
t_values2 = []
d_values3 = []
t_values3 = []
d_values_oscillations = []

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
    t_start = t_values_last_4_peaks[-4]
    t_end = t_values_last_4_peaks[-1]

    # Find indices of t_values within t_value limits
    indices = [i for i, t in enumerate(t_values) if t >= t_start and t <= t_end]

    # Extract d_values within the time limits
    d_values_last_3_oscillations = [d_values[i] for i in indices]
    d_values_oscillations.append(d_values_last_3_oscillations)

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

def delay_model_rhy_deg_case1(y, t, parameters):
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

    t_values2.append(t)
    d_values2.append(d)

    dxdt = (c / (ck + x_delayed)) ** 3 - d * x

    return dxdt

def delay_model_rhy_deg_case2(y, t, parameters):
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

    t_values3.append(t)
    d_values3.append(d)

    dxdt = (c / (ck + x_delayed)) ** 3 - d * x

    return dxdt

# Define the initial condition function
def history (t):
    return 1
g = lambda t: 1

# Define the time interval for the DDE solver
t0 = 0
tfinal = 1000
dt = 0.05
t = np.arange(t0, tfinal, dt)

# Define parameters
parameters1 = {'c'  : 1,
              'ck'  : 0.1,
              'd1'  : center_deg_rate,  # Degradation rate when slope is positive
              'd2'  : center_deg_rate,  # Degradation rate when slope is negative
              'tau' : 7.58}

parameters2 = {'c'  : 1,
              'ck'  : 0.1,
              'd1'  : min_deg_rate,  # Degradation rate when slope is positive
              'd2'  : max_deg_rate,  # Degradation rate when slope is negative
              'tau' : 7.58}

parameters3 = {'c'  : 1,
              'ck'  : 0.1,
              'd1'  : center_deg_rate - 0.07 * center_deg_rate,  # Degradation rate when slope is positive
              'd2'  : center_deg_rate + 0.07 * center_deg_rate,  # Degradation rate when slope is negative
              'tau' : 7.58}

# Call the ddeint function
result1 = ddeint(delay_model_const_deg, history, t, fargs=(parameters1,))
result2 = ddeint(delay_model_rhy_deg_case1, history, t, fargs=(parameters2,))
result3 = ddeint(delay_model_rhy_deg_case2, history, t, fargs=(parameters3,))

# Remove transient behavior, keep only asymptotic dynamics
last_osc = 100 #keep last 100h 
time_points = int(last_osc/dt) # Calculate the number of time points that correspond to the last oscillation period
t_asymp = t[-time_points:] #extract the time points from the last 100 h
# t_asymp = np.arange(0, last_osc, dt) #start from 0
result_asymp1 = result1[-time_points:, :] #extract the result from the last 100 h
result_asymp2 = result2[-time_points:, :] #extract the result from the last 100 h
result_asymp3 = result3[-time_points:, :] #extract the result from the last 100 h

# interpolate d_values and keep datapoints for last 100h
d_values1_interp = interp1d(t_values1, d_values1, kind='nearest', 
                            fill_value='extrapolate')
d_values1_aligned = d_values1_interp(t)
d1_last_100_hours = d_values1_aligned[-time_points:]

d_values2_interp = interp1d(t_values2, d_values2, kind='nearest', 
                            fill_value='extrapolate')
d_values2_aligned = d_values2_interp(t)
d2_last_100_hours = d_values2_aligned[-time_points:]

d_values3_interp = interp1d(t_values3, d_values3, kind='nearest', 
                            fill_value='extrapolate')
d_values3_aligned = d_values3_interp(t)
d3_last_100_hours = d_values3_aligned[-time_points:]

## Extract the degradation rates and the corresponding times from the last 100 hours
#t1_num_elements = len(t_values1) // 10
#t1_last_100_hours = t_values1[-t1_num_elements:]
#d1_num_elements = len(d_values1) // 10
#d1_last_100_hours = d_values1[-d1_num_elements:]
#t1_sorted, d1_sorted = sort_deg_rates(t1_last_100_hours, d1_last_100_hours)
#t2_num_elements = len(t_values2) // 10
#t2_last_100_hours = t_values2[-t2_num_elements:]
#d2_num_elements = len(d_values2) // 10
#d2_last_100_hours = d_values2[-d2_num_elements:]
#t2_sorted, d2_sorted = sort_deg_rates(t2_last_100_hours, d2_last_100_hours)
#t3_num_elements = len(t_values3) // 10
#t3_last_100_hours = t_values3[-t3_num_elements:]
#d3_num_elements = len(d_values3) // 10
#d3_last_100_hours = d_values3[-d3_num_elements:]
#t3_sorted, d3_sorted = sort_deg_rates(t3_last_100_hours, d3_last_100_hours)

# Normalize all oscillations to their mean
result_asymp1 = normalize_oscillations(result_asymp1)
result_asymp2 = normalize_oscillations(result_asymp2)
result_asymp3 = normalize_oscillations(result_asymp3)

# Get the period and the relative amplitude
rel_amplitude1 = relative_amplitude(result_asymp1)
rel_amplitude2 = relative_amplitude(result_asymp2)
rel_amplitude3 = relative_amplitude(result_asymp3)
period1, sd_period1 = calculate_period(result_asymp1, t_asymp)
period2, sd_period2 = calculate_period(result_asymp2, t_asymp)
period3, sd_period3 = calculate_period(result_asymp3, t_asymp)

# Get the average degradation rate
# d_average1 = calculate_average_deg_rate(d1_last_100_hours, result_asymp1, t1_last_100_hours, t_asymp)
# d_average2 = calculate_average_deg_rate(d2_last_100_hours, result_asymp2, t2_last_100_hours, t_asymp)
#d_average3 = calculate_average_deg_rate(d3_last_100_hours, result_asymp3, t3_last_100_hours, t_asymp)

from collections import Counter
# Flatten the nested list
flat_list = [item for sublist in d_values_oscillations for item in sublist]

# Count occurrences of each value in the list
counter = Counter(flat_list)

# Print the counter
print(counter)

### PLOTTING
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4)) 

x_axis_tick_positions = [900, 924, 948, 972, 996]
x_axis_tick_labels = [0, 24, 48, 72, 96]

ax1.plot(t_asymp, result_asymp1, label=f'$x$ [a.u.], rel. amplitude = {rel_amplitude1:.2f},\n period = {period1:.1f} h', 
         color = 'k', linestyle='--')
ax1.plot(t_asymp, d1_last_100_hours, color = 'k', label='degradation rate $d$ [h⁻¹]')
#ax1.scatter(t1_last_100_hours, d1_last_100_hours, color = 'grey', s = 0.1)#, label = f'deg. rate, average = {d_average1:.4f}')
# ax1.plot(t1_sorted, d1_sorted, color = 'grey', linestyle = '--', label = f'deg. rate, average = {d_average1:.4f}')
ax1.set_xticks(x_axis_tick_positions)
ax1.set_xticklabels(x_axis_tick_labels)
ax1.set_xlabel('Time [h]')
ax1.set_ylabel('')
ax1.set_ylim(0,2.9)
ax1.set_title('Constant degradation')
ax1.legend(fontsize = 10)

ax2.plot(t_asymp, result_asymp2, c = 'dodgerblue', linestyle='--',
         label=f'$x$ [a.u.], rel. amplitude = {rel_amplitude2:.2f},\n period = {period2:.1f} h')
ax2.plot(t_asymp, d2_last_100_hours, color = 'dodgerblue', label='degradation rate $d$ [h⁻¹]')
#ax2.scatter(t2_last_100_hours, d2_last_100_hours, color = 'dodgerblue', alpha=0.7, s = 0.1)#, label = f'deg. rate, average = {d_average2:.4f}')
# ax2.plot(t2_sorted, d2_sorted, color = 'grey', linestyle = '--', label = f'deg. rate, average = {d_average2:.4f}')
ax2.set_xlabel('Time [h]')
ax2.set_xticks(x_axis_tick_positions)
ax2.set_xticklabels(x_axis_tick_labels)
ax2.set_ylabel('')
ax2.set_ylim(0,2.9)
ax2.set_title('Rhythmic degradation, case 1')
ax2.legend(fontsize = 10)

ax3.plot(t_asymp, result_asymp3, c= 'crimson', linestyle='--',
         label=f'$x$ [a.u.], rel. amplitude = {rel_amplitude3:.2f},\n period = {period3:.1f} h')
ax3.plot(t_asymp, d3_last_100_hours, color = 'crimson', label='degradation rate $d$ [h⁻¹]')
#ax3.scatter(t3_last_100_hours, d3_last_100_hours, color = 'crimson', alpha = 0.7, s = 0.1)#, label = f'deg. rate, average = {d_average3:.4f}')
# ax3.plot(t3_sorted, d3_sorted, color = 'grey', linestyle = '--', label = f'deg. rate, average = {d_average3:.4f}')
ax3.set_xlabel('Time [h]')
ax3.set_xticks(x_axis_tick_positions)
ax3.set_xticklabels(x_axis_tick_labels)
ax3.set_ylabel('')
ax3.set_ylim(0,2.9)
ax3.set_title('Rhythmic degradation, case 2')
ax3.legend(fontsize = 10)

plt.show()

# fig.savefig(f'20240728_timeseries_dt_point_005.png', dpi = 300)

# if dt == 0.05:
#     name = '005'
# elif dt == 0.01:
#     name = '001'
# elif dt == 0.10:
#     name = '010'
# fig.savefig(f'20240702_timeseries_dt_point_{name}.png', dpi = 600)
# fig.savefig(f'20240702_timeseries_dt_point_{name}.pdf', format='pdf')