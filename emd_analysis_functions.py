import emd
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import bycycle
from bycycle import features
from bycycle import plts
# You can find analysis.py on https://gitlab.com/marcoFabus/fabus2021_itemd
from analysis import *
from cyclepoints_custom import *
from extrema_custom import *
from compute_shape_features_custom import *
import scipy.io
import copy


# Returns n_methods sets of IMFs, depending on the amount of methods
def run_mask_methods(trace, method_partials):
    imfs_methods = []
    masks = []
    for method in method_partials:
        # I assume that the partial functions return the masks used. So that config['ret_mask_freq'] is set to True
        # This is why the partial functions return both the imfs as well as the used frequency masks
        imfs_methods.append(method(trace)[0])
        masks.append(method(trace)[1])
    return imfs_methods, masks


# apply frequency transform on a list of methods, each having their own set of IMF's
# returns the IPIFIA of all the methods' IMF's
def freqtr_methods(imfs_methods, srate):
    ipifia_all = []
    for method in imfs_methods:
        IP, IF, IA = emd.spectra.frequency_transform(method, srate, 'nht')
        ipifia_all.append(np.array([IP,IF, IA]))
    return ipifia_all


# perform the hilbert transform to find the frequency of a set of imf's
# returns n_imfs frequency bins that represent each imf best
def calc_imf_freqs(ipifia_array, freq_edges):
    IP = ipifia_array[0]
    IF = ipifia_array[1]
    IA = ipifia_array[2]
    f, hht = emd.spectra.hilberthuang(IF, IA, freq_edges, mode='amplitude', sum_time=False, sum_imfs=False)
    return f[hht.sum(axis=1).argmax(axis=0)], f, hht


# uses calc_imf_freqs multiple times on a list of IPIFIA arrays for each sifting method 
def calc_imf_freqs_all(ipifia_all, freq_edges):
    freqs_imfs_all = []
    theta_indices_all = []
    hht_all = []
    for ipifia_method in ipifia_all:
        freqs_imfs, f, hht = calc_imf_freqs(ipifia_method, freq_edges)
        freqs_imfs_all.append(freqs_imfs)
        hht_all.append(hht)
        theta_indices = extract_thetas(freqs_imfs)
        theta_indices_all.append(theta_indices)
    return freqs_imfs_all, theta_indices_all, hht_all


# returns the indices of IMF's that are assigned to a frequency bin within the theta range
# theta_indices is a list with n_method lists, which in turn contain the selected imf indices
def extract_thetas(freqs_imfs):
    theta_indices = np.nonzero( [freq > 4 and freq < 12 for freq in freqs_imfs])[0].tolist()
    return theta_indices

# returns a list of length n_methods
# each element in that list is an IMF. Either a single theta IMF or multiple IMF's summed up.
# also the frequency of the selected imfs is returned
# this function is used when running on the object space task data, as it sums all the IMFs within theta range.
def select_imfs(imfs_all, freqs_imfs_all, indices_all, freq_edges, hht_all, srate):
    selected_imfs_all = []
    selected_freqs_all = []
    # Set of IMF's, after summing up all the theta IMF's in a method
    # So for example, if IT mask sift has IMF-3 and IMF-4 in theta range,
    # I will add IMF-3 + IMF-4 to the set of IMF's, and remove IMF-3 and IMF-4 from the set.
    # ae stands for after extraction, which refers to the process in the line above
    imfs_all_ae = copy.deepcopy(imfs_all)
    selected_hhts = []
    for nth_method, imfs in enumerate(imfs_all):
        samples = len(imfs[:,0])
        summed_imf = np.zeros(samples)
        imfs_to_remove = []
        for count, i in enumerate(copy.deepcopy(indices_all[nth_method])):
            zcs =  emd.imftools.zero_crossing_count(imfs[:,i])/samples*srate
            freq = freqs_imfs_all[nth_method][i]
            if abs(zcs-freq) < 20:
                summed_imf += imfs[:,i]
                if len(indices_all[nth_method]) >1:
                    imfs_to_remove.append(i)
                else:
                    selected_hhts.append(hht_all[nth_method][:,:, i])
            # filter out noisy IMF's that have too many zero crossings for the frequency they were assigned to
            else:
                del indices_all[nth_method][count]
        imfs_all_ae[nth_method] = np.delete(imfs_all_ae[nth_method], imfs_to_remove ,1)
        
        if len(indices_all[nth_method]) >1:
            imfs_all_ae[nth_method] = np.insert(imfs_all_ae[nth_method], indices_all[nth_method][0], summed_imf, axis=1)
            ipifia_summed = freqtr_methods([summed_imf], srate)[0]
            _, _, summed_hht = calc_imf_freqs(ipifia_summed, freq_edges)
            selected_hhts.append(summed_hht[:,:,0])
        elif len(indices_all[nth_method]) == 0:
            selected_hhts.append([])
        selected_imfs_all.append(summed_imf)
        selected_freqs = [freqs_imfs_all[nth_method][i] for i in indices_all[nth_method]]
        selected_freqs_all.append(selected_freqs)
    return imfs_all_ae, selected_imfs_all, selected_freqs_all, indices_all, selected_hhts
        
# from a list of pmsi sets and imf indices, compute the pmsi of each n'th IMF in each set of IMF's.
def pmsi_all(imfs_all_ae, theta_indices):
    pmsi_all = []
    for method_imfs, method_thetas in zip(imfs_all_ae, theta_indices):
        if len(method_thetas) > 1:
            pmsi_single_method = PMSI(method_imfs, method_thetas[0], method='both')
        elif len(method_thetas) == 0:
            pmsi_single_method = np.nan
        else:
            imf = method_thetas[0]
            if imf == len(method_imfs[0]) - 1:
                # if the selected imf was the last IMF of the set, set the PMSI as the PMSI to the IMF above, multiplied by 2
                pmsi_single_method = PMSI(method_imfs, imf, method='above') * 2
            else:
                pmsi_single_method = PMSI(method_imfs, imf, method='both')
        pmsi_all.append(pmsi_single_method)
    return pmsi_all


def plot_psmi_all(pmsi_all, theta_indices_all, selected_freqs_all, method_names, title):
    f, ax = plt.subplots(figsize=(18,5))
    nth_method = 0
    xaxis = np.arange(len(method_names))
    for method_pmsis, method_theta_indices in zip(pmsi_all, theta_indices_all):
        nth_index = 0
        for pmsi, theta_index in zip(method_pmsis, method_theta_indices):
            bar_heights = [0] * len(method_names)
            bar_heights[nth_method] = pmsi
            label = 'IMF-' + str(theta_index+1) +": " + str( round(selected_freqs_all[nth_method][nth_index],2) )
            plt.bar(xaxis + nth_index*1/len(method_pmsis)*0.4, bar_heights, width = 1/len(method_pmsis)*0.4, label = label)
            nth_index+=1
        nth_method += 1
    plt.xticks(xaxis, method_names)
    ax.legend()
    plt.title(title)
    plt.show()

def plot_pt_imfs(selected_imfs_all, theta_indices_all, srate, method_names):
    subplot = 0
    rows = len(selected_imfs_all)
    fig3,axs3 = plt.subplots(rows,1, figsize=(20,20))
    for i, imf in enumerate(selected_imfs_all):
        df_cyclepoints = compute_cyclepoints_custom(imf[:3*srate], srate, f_range = None)
        bycycle.plts.plot_cyclepoints_df(df_cyclepoints, imf[:3*srate], srate, plot_zerox=False, fig=fig3, ax=axs3[subplot])
        title = method_names[i] + " IMF('s) used: " + str(np.array(theta_indices_all[i]) +1)
        axs3[subplot].set_title(title)
        subplot += 1

def plot_output_methods(selected_imfs_all, srate):
    plots = [plt.plot(imf[:5*srate]) for imf in selected_imfs_all]
    return plots

def single_trial_analysis(trial, srate, maskmethods_list, ensemblemethods_list, freq_edges):
    imfs_methods, _ = run_mask_methods(trial, maskmethods_list)
    for ensemble_config in ensemblemethods_list:
        imfs_methods.append(ensemble_config(trial))
    freq_stats = freqtr_methods(imfs_methods, srate)
    freqs_imfs_all, theta_indices_all, hht_all = calc_imf_freqs_all(freq_stats, freq_edges)
    imfs_methods_ae, selected_imfs_all, selected_freqs_all, theta_indices_ae, selected_hhts = select_imfs(imfs_methods, freqs_imfs_all, theta_indices_all, freq_edges, hht_all, srate)
    pmsis_all = pmsi_all(imfs_methods_ae, theta_indices_ae)
    return imfs_methods_ae, selected_imfs_all, selected_freqs_all, theta_indices_all, selected_hhts, pmsis_all



def trials_analysis(trials_list, maskmethods_list, ensemblemethods_list, method_names, srate, freq_edges):
    # The extracted theta waves of each method, per trial. so the first element will be a set of n theta imf's, where n is the amount of methods
    selected_imfs_trials = []
    # The new set of IMF's where the theta imf's in a method are summed up
    imfs_ae_trials = []
    # HHT's of the theta extracted waves
    hhts_trials = []
    selected_freqs_trials = []
    trials = len(trials_list)
    methods = len(method_names)
    pmsis_trials = np.zeros((trials, methods))
    for nth_trial, trial in enumerate(trials_list):
        imfs_methods_ae, selected_imfs_all, selected_freqs_all, _, selected_hhts, pmsis_all = single_trial_analysis(trial, srate, maskmethods_list, ensemblemethods_list, freq_edges)
        hhts_trials.append(selected_hhts)
        selected_imfs_trials.append(selected_imfs_all)
        imfs_ae_trials.append(imfs_methods_ae)
        selected_freqs_trials.append(selected_freqs_all)
    return selected_imfs_trials, imfs_ae_trials, selected_freqs_trials, hhts_trials, pmsis_trials

def plot_pmsi(pmsis_trials, method_names, title):
    fig2, axs2 = plt.subplots(1, figsize = (25,10))
    means = np.nanmean(pmsis_trials, axis = 0)
    errors = np.nanstd(pmsis_trials, axis=0)
    xaxis = np.arange(len(method_names))
    axs2.set_ylabel('PMSI', size=20)
    axs2.set_xticks(xaxis)
    axs2.set_xticklabels(method_names)
    axs2.set_title(title, size=30)
    axs2.yaxis.grid(True)
    axs2.bar(xaxis, means, alpha=0.5, ecolor='black', capsize=15, yerr=errors)
    axs2.xaxis.set_tick_params(labelsize=15)
    axs2.yaxis.set_tick_params(labelsize=15)

# apply a sine function 'order' times on a time series
# a higher order increases IF variability
def iterated_sine(points, order, normalize = True):
    for _ in range(order):
        points = np.sin(points)
    if normalize == False:
        return points
    else:
        return points / np.max(points)

# returns a list selected_imfs_all of length n_methods
# each element in that list is an IMF. Either a single theta IMF or multiple IMF's summed up.
# also the frequency of the selected imfs is returned
# this function is used to select IMFs from EMD outputs on generated signals. The IMF selected will be 
# the IMF closest to the target_freq. When add_imf_above is set to True, the IMF above the selected
# IMF will be summed with the selected IMF. 
def select_imfs_target(imfs_all, freqs_imfs_all, indices_all, hht_all, srate, target_freq, add_imf_above = False):
    selected_imfs_all = []
    selected_freqs_all = []
    selected_hhts = []
    for nth_method, imfs in enumerate(imfs_all):
        samples = len(imfs[:,0])
        for count, i in enumerate(copy.deepcopy(indices_all[nth_method])):
            zcs =  (emd.imftools.zero_crossing_count(imfs[:,i])/samples)*srate
            freq = freqs_imfs_all[nth_method][i]
            # filter out noisy IMF's that have too many zero crossings for the frequency they were assigned to
            if abs(zcs-freq) > 20:
                indices_all[nth_method].remove(i)
        if len(indices_all[nth_method]) == 1:
            selected_imf = imfs[:, indices_all[nth_method][0]]
            selected_hht = hht_all[nth_method][:,:, indices_all[nth_method][0]]
        elif len(indices_all[nth_method]) > 1:
            candidate_freqs = [freqs_imfs_all[nth_method][i] for i in indices_all[nth_method]]
            diff = abs( np.array(candidate_freqs) - target_freq)
            closest = np.argmin(diff)
            indices_all[nth_method] = [ indices_all[nth_method][closest] ]
            selected_imf = imfs[:, indices_all[nth_method][0]]
            selected_hht = hht_all[nth_method][:,:, indices_all[nth_method][0]]
        elif len(indices_all[nth_method]) == 0:
            selected_imf = np.zeros_like(imfs_all[nth_method])
            selected_hht = np.zeros_like(hht_all[0][:,:, 0])
        if add_imf_above == True and len(indices_all[nth_method]) > 0:
            selected_imf += imfs[:, indices_all[nth_method][0]-1]
        
        selected_imfs_all.append(selected_imf)
        selected_freqs = [freqs_imfs_all[nth_method][i] for i in indices_all[nth_method]]
        selected_freqs_all.append(selected_freqs)
        selected_hhts.append(selected_hht)
    return selected_imfs_all, selected_freqs_all, indices_all, selected_hhts

# use an amount of EMD method configurations on a single generated signal. returns the sets of imfs and other characteristics of each methods' output 
def gensignal_analysis(trial, srate, maskmethods_list, ensemblemethods_list, freq_edges, target_freq, add_imf_above = False):
    imfs_methods, _ = run_mask_methods(trial, maskmethods_list)
    for ensemble_config in ensemblemethods_list:
        imfs_methods.append(ensemble_config(trial))
    freq_stats = freqtr_methods(imfs_methods, srate)
    freqs_imfs_all, theta_indices_all, hht_all = calc_imf_freqs_all(freq_stats, freq_edges)
    selected_imfs_all, selected_freqs_all, theta_indices, selected_hhts = select_imfs_target(imfs_methods, freqs_imfs_all, theta_indices_all, hht_all, srate, target_freq, add_imf_above)
    pmsis_all = pmsi_all(imfs_methods, theta_indices)
    return imfs_methods, selected_imfs_all, selected_freqs_all, theta_indices, selected_hhts, pmsis_all

# convert a signal to its phase-aligned waveform
def calc_pa_wf(signal, srate, amppercentile = 10, npoints=48):
    IP, IF, IA = emd.spectra.frequency_transform(signal, srate, 'nht')
    thresh = np.percentile(IA, amppercentile)
    mask = IA > thresh
    mask_cycles = emd.cycles.get_cycle_vector(IP, return_good=True, mask=mask)
    pa_signal = emd.cycles.phase_align(IP, signal, cycles=mask_cycles, npoints = npoints)
    return np.nanmean(pa_signal[0], axis=1)

# compute the phase-aligned IF by finding cycles and averaging over the IFs in these cycles.
def calc_pa_IF(signal, srate, amppercentile = 10, npoints=48):
    IP, IF, IA = emd.spectra.frequency_transform(signal, srate, 'nht')
    thresh = np.percentile(IA, amppercentile)
    mask = IA > thresh
    mask_cycles = emd.cycles.get_cycle_vector(IP, return_good=True, mask=mask)
    pa_if_signal = emd.cycles.phase_align(IP, IF, cycles = mask_cycles, npoints=npoints)
    return np.nanmean(pa_if_signal[0], axis=1)

# compute the frequency distortion of a phase-aligned IF
def calc_fd(pa_if, target_freq):
    return (np.max(pa_if) - np.min(pa_if)) / target_freq

# construct a theta-phase modulating component, and a gamma-amplitude modulated component summed.
def construct_thetagamma(f_p, f_a, srate, data_length, n_sin, A_fpmax = 1, nonmodulatedamplitude=2):
    npnts = srate*data_length #number of points to generate,
    t  = np.arange(0,npnts)/srate #time vector
    # fp is the theta/phase frequency
    # fa is the gamma/amplitude frequency
    # A_fpmax is the max amplitude of the theta component
    theta_pure = A_fpmax*np.sin(2*np.pi*t*f_p)
    if n_sin == 0:
        theta = theta_pure
    else:
        theta = iterated_sine(theta_pure, n_sin)
    A_fpmax = 1 #Maximal amplitude of fp
    # A_fa is the theta-modulated amplitude of the gamme component
    A_fa=(0.2*(theta+1)+nonmodulatedamplitude*0.1)
    gamma_pure = np.sin(2*np.pi*t*f_a)
    gamma = A_fa * gamma_pure
    return theta + gamma, theta, gamma

# add white noise to a signal
def add_noise(signal, sigma):
    gauss = sigma * np.random.randn(1,len(signal))
    return np.add(signal, gauss[0])

# construct a synthetic signal with a theta, gamma and white noise component.
def construct_synth(f_p, f_a, srate, data_length, n_sin, sigma, A_fpmax = 1, nonmodulatedamplitude=2):
    tg, _, _ = construct_thetagamma(f_p, f_a, srate, data_length, n_sin, A_fpmax = A_fpmax, nonmodulatedamplitude=nonmodulatedamplitude)
    return add_noise(tg, sigma)

# compute pearson's r correlation between two phase-aligned IFs
def compute_corr(paIF_gt, paIF):
    return scipy.stats.pearsonr(paIF_gt, paIF)[0]

def corrs_increasing_noise(truth_paif, thetagamma, srate, noise_bins, method_names, maskmethods_gen, ensemblemethods_gen, freq_edges, f_p, iterations, add_imf_above=False):
    """

    Computes the correlation of an EMD output to the ground truth component under varying noise. 
    The ground truth component is part of a signal with a gamma component and a noise component.

    Parameters
    ----------
    truth_paif :
        Phase-aligned IF of ground truth component.
    srate :
        Sample rate of signal in Hz.
    noise_bins :
        an array of standard deviations of noise.
    method_names : names of EMD methods to use.
    maskmethods_gen : 
        List of configurations for all (iterated) mask EMD methods.
    ensemblemethods_gen :
        List of configurations for ensembleEMD methods.
    freq_edges  :
        Edges of frequency bins to set domain for spectral analysis.
    f_p :
        The frequency of the phase-modulating frequency (part of the true theta component).
    add_imf_above:
        Determines whether to select one IMF from each set, or two. If False, only the IMF with an estimated
        frequency closest to f_p will be compared to the ground truth. If True, the IMF above the selected
        IMF will be added to the selected IMF for further analysis.
    Returns
    -------
    corrs_mean : The average corr for each method after n iterations.
    corrs_std : The average corr error for each method after n iterations.
        

    """
    
    corrs = np.zeros((len(noise_bins), len(method_names), iterations))
    for iteration in range(iterations):
        for row, sigma in enumerate(noise_bins):
            lfp = add_noise(thetagamma, sigma) 
            _, selected_imfs_all, _, _, _, _ = gensignal_analysis(lfp, srate, maskmethods_gen, ensemblemethods_gen, freq_edges, f_p, add_imf_above)
            for col, imf in enumerate(selected_imfs_all):
                try:
                    imf_pa_IF = calc_pa_IF(imf, srate)
                    corr = scipy.stats.pearsonr(truth_paif, imf_pa_IF)[0]
                except:
                    corr = np.nan
                corrs[row,col,iteration] = corr
        print('\r i = {} , {}'.format(iteration, (corrs[:,:,iteration]) ), end="")
    corrs_mean = np.nanmean(corrs, axis=2)
    corrs_std = np.nanstd(corrs, axis=2)
    return corrs_mean, corrs_std

def corrs_increasing_fd(n_sin_bins, method_names, maskmethods_gen, ensemblemethods_gen, freq_edges, iterations, f_p, f_a, srate, data_length, fixed_sigma, A_fpmax = 1, nonmodulatedamplitude=2, add_imf_above=False):
    """
    Computes the correlation of an EMD output to the ground truth component with varying degrees of frequency distortion. 
    The ground truth component is part of a signal with a gamma component and a noise component.
    ----------
    n_sin_bins :
        An array of amounts of sine iterations
    method_names : names of EMD methods to use.
    maskmethods_gen : 
        List of configurations for all (iterated) mask EMD methods.
    ensemblemethods_gen :
        List of configurations for ensembleEMD methods.
    freq_edges  :
        Edges of frequency bins to set domain for spectral analysis.
    f_p :
        The frequency of the phase-modulating frequency (part of the true theta component).
    f_a :
        The frequency of the amplitude-modulated frequency (part of the gamma component).
    srate :
        Sample rate of signal in Hz.
    data_length : 
        Length of generated signals.
    fixed_sigma : 
        Standard deviation of white noise component in generated signal.
    A_fpmax : 
        The max amplitude of the phase-modulating (theta) component
    nonmodulatedamplitude : 
        The amplitude of gamma component that is not modulated by theta phase.
    add_imf_above:
        Determines whether to select one IMF from each set, or two. If False, only the IMF with an estimated
        frequency closest to f_p will be compared to the ground truth. If True, the IMF above the selected
        IMF will be added to the selected IMF for further analysis.
    Returns
    -------
    corrs_mean : The average corr for each method after n iterations.
    corrs_std : The average corr error for each method after n iterations.
        

    """
    corrs = np.zeros((len(n_sin_bins), len(method_names), iterations))
    for iteration in range(iterations):
        for row, n_sin in enumerate(n_sin_bins):
            n_sin = int(n_sin)
            # construct synthetic signal
            lfp = construct_synth(f_p, f_a, srate, data_length, n_sin, fixed_sigma, A_fpmax = A_fpmax, nonmodulatedamplitude=nonmodulatedamplitude)
            truth_paif = calc_pa_IF(lfp, srate)
            _, selected_imfs_all, _, _, _, _ = gensignal_analysis(lfp, srate, maskmethods_gen, ensemblemethods_gen, freq_edges, f_p, add_imf_above)
            for col, imf in enumerate(selected_imfs_all):
                try:
                    imf_pa_IF = calc_pa_IF(imf, srate)
                    corr = scipy.stats.pearsonr(truth_paif, imf_pa_IF)[0]
                except:
                    corr = np.nan
                corrs[row,col,iteration] = corr
        print('\r i = {} , {}'.format(iteration, (corrs[:,:,iteration]) ), end="")
    corrs_mean = np.nanmean(corrs, axis=2)
    corrs_std = np.nanstd(corrs, axis=2)
    return corrs_mean, corrs_std
    
def pmsis_increasing_noise(thetagamma, srate, noise_bins, method_names, maskmethods_gen, ensemblemethods_gen, freq_edges, f_p, iterations, add_imf_above=False):
    """

    Computes the correlation of an EMD output to the ground truth component under varying noise. 
    The ground truth component is part of a signal with a gamma component and a noise component.

    Parameters
    ----------
    thetagamma :
        signal with a theta and a gamma component summed up.
    srate :
        Sample rate of signal in Hz.
    noise_bins :
        an array of standard deviations of noise.
    method_names : names of EMD methods to use.
    maskmethods_gen : 
        List of configurations for all (iterated) mask EMD methods.
    ensemblemethods_gen :
        List of configurations for ensembleEMD methods.
    freq_edges  :
        Edges of frequency bins to set domain for spectral analysis.
    f_p :
        The frequency of the phase-modulating frequency (part of the true theta component).
    add_imf_above:
        Determines whether to select one IMF from each set, or two. If False, only the IMF with an estimated
        frequency closest to f_p will be compared to the ground truth. If True, the IMF above the selected
        IMF will be added to the selected IMF for further analysis.
    Returns
    -------
    pmsis_mean : The average pmsis for each method after n iterations.
    pmsis_std : The average pmsis error for each method after n iterations.
        

    """
    pmsis = np.zeros((len(noise_bins), len(method_names), iterations))
    for iteration in range(iterations):
        for row, sigma in enumerate(noise_bins):
            lfp = add_noise(thetagamma, sigma) 
            _, _, _, _, _, pmsis_all = gensignal_analysis(lfp, srate, maskmethods_gen, ensemblemethods_gen, freq_edges, f_p, add_imf_above)
            pmsis[row,:,iteration] = np.array(pmsis_all)
    pmsis_mean = np.nanmean(pmsis, axis=2)
    pmsis_std = np.nanstd(pmsis, axis=2)
    return pmsis_mean, pmsis_std

def pmsis_increasing_fd(n_sin_bins, method_names, maskmethods_gen, ensemblemethods_gen, freq_edges, iterations, f_p, f_a, srate, data_length, fixed_sigma, A_fpmax = 1, nonmodulatedamplitude=2, add_imf_above=False):
    """

    Computes the PMSI of selected IMF or the sum of 2 IMFs of different EMD methods on generated signals with varying frequency distortion in theta component.  
    ----------
    n_sin_bins :
        An array of amounts of sine iterations
    method_names : names of EMD methods to use.
    maskmethods_gen : 
        List of configurations for all (iterated) mask EMD methods.
    ensemblemethods_gen :
        List of configurations for ensembleEMD methods.
    freq_edges  :
        Edges of frequency bins to set domain for spectral analysis.
    f_p :
        The frequency of the phase-modulating frequency (part of the true theta component).
    f_a :
        The frequency of the amplitude-modulated frequency (part of the gamma component).
    srate :
        Sample rate of signal in Hz.
    data_length : 
        Length of generated signals.
    fixed_sigma : 
        Standard deviation of white noise component in generated signal.
    A_fpmax : 
        The max amplitude of the phase-modulating (theta) component
    nonmodulatedamplitude : 
        The amplitude of gamma component that is not modulated by theta phase.
    add_imf_above:
        Determines whether to select one IMF from each set, or two. If False, only the IMF with an estimated
        frequency closest to f_p will be compared to the ground truth. If True, the IMF above the selected
        IMF will be added to the selected IMF for further analysis.
    Returns
    -------
    pmsis_mean : The average pmsis for each method after n iterations.
    pmsis_std : The average pmsis error for each method after n iterations.
        

    """
    pmsis = np.zeros((len(n_sin_bins), len(method_names), iterations))
    for iteration in range(iterations):
        for row, n_sin in enumerate(n_sin_bins):
            n_sin = int(n_sin)
            # construct synthetic signal
            lfp = construct_synth(f_p, f_a, srate, data_length, n_sin, fixed_sigma, A_fpmax = A_fpmax, nonmodulatedamplitude=nonmodulatedamplitude)
            _, _, _, _, _, pmsis_all = gensignal_analysis(lfp, srate, maskmethods_gen, ensemblemethods_gen, freq_edges, f_p, add_imf_above)
            pmsis[row,:,iteration] = np.array(pmsis_all)
    pmsis_mean = np.nanmean(pmsis, axis=2)
    pmsis_std = np.nanstd(pmsis, axis=2)
    return pmsis_mean, pmsis_std

