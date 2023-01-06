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
def run_mask_methods(trace, srate, method_partials):
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
def select_imfs(imfs_all, freqs_imfs_all, indices_all, freq_edges, hht_all, srate, pmsi_only = False):
    selected_imfs_all = []
    selected_freqs_all = []
    # Set of IMF's, after summing up all the theta IMF's in a method
    # So for example, if IT mask sift has IMF-3 and IMF-4 in theta range,
    # I will add IMF-3 + IMF-4 to the set of IMF's, and remove IMF-3 and IMF-4 from the set.
    imfs_all_ae = copy.deepcopy(imfs_all)
    selected_hhts = []
    for nth_method, imfs in enumerate(imfs_all):
        samples = len(imfs[:,0])
        summed_imf = np.zeros(samples)
        for count, i in enumerate(copy.deepcopy(indices_all[nth_method])):
            imfs_to_remove = []
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
        for imf_index in imfs_to_remove:
            imfs_all_ae[nth_method] = np.delete(imfs_all_ae[nth_method], imf_index ,1)
        
        if len(indices_all[nth_method]) >1:
            imfs_all_ae[nth_method] = np.insert(imfs_all_ae[nth_method],-1, summed_imf, axis=1)
            ipifia_summed = freqtr_methods([summed_imf], srate)[0]
            _, _, summed_hht = calc_imf_freqs(ipifia_summed, freq_edges)
            selected_hhts.append(summed_hht[:,:,0])
        elif len(indices_all[nth_method]) == 0:
            selected_hhts.append([])
        selected_imfs_all.append(summed_imf)
        selected_freqs = [freqs_imfs_all[nth_method][i] for i in indices_all[nth_method]]
        selected_freqs_all.append(selected_freqs)
    if pmsi_only == True:
        return imfs_all_ae, indices_all
    return imfs_all_ae, selected_imfs_all, selected_freqs_all, indices_all, selected_hhts
        
# plots the hht and the selected imf's for visualisation
def plot_imfs_methods(imfs_methods, method_names, freqs_imfs_all, theta_indices_all, freq_centres, selected_hhts, srate):
    assert len(imfs_methods) == len(method_names)
    subplot = 0
    nth_method = 0
    samples = len(imfs_methods[0])
    time_centres = np.arange(samples)-.5
    rows = len(imfs_methods)
    fig3,axs3 = plt.subplots(rows,2, figsize=(20,20))
    fig3.tight_layout(pad=5)

    for imf, method_name, hht in zip(imfs_methods, method_names, selected_hhts):
        label = "IMF-" + str(theta_indices_all[nth_method][0]+1) + ", " + str(freqs_imfs_all[nth_method][theta_indices_all[nth_method][0]])
        if len(theta_indices_all[nth_method]) >1:
            for theta_index in theta_indices_all[nth_method][1:]:
                label += " + IMF-" + str( theta_index +1) + ", " + str(freqs_imfs_all[nth_method][theta_index])
        emd.plotting.plot_hilberthuang(selected_hhts[nth_method][:3*srate, :], time_centres/srate, freq_centres,
                               cmap='viridis', time_lims = (0,3), freq_lims=(1,20), log_y=False, fig=fig3, ax=axs3[subplot, 1])

        axs3[subplot,0].plot(np.arange(0,len(imf[:3*srate]))/srate, imf[:3*srate]/srate, label = label)
        axs3[subplot,0].legend()
        axs3[subplot,0].set_xlabel('Time (s)')
        axs3[subplot,0].set_ylabel('\u03BCv')
        axs3[subplot,0].set_title(method_name)
        subplot += 1
        nth_method += 1
    plt.show()

def pmsi_all(imfs_all_ae, theta_indices):
    pmsi_all = []
    # nth_method = 0
    for method_imfs, method_thetas in zip(imfs_all_ae, theta_indices):
        if len(method_thetas) > 1:
            pmsi_single_method = PMSI(method_imfs, -1, method='both')
        elif len(method_thetas) == 0:
            pmsi_single_method = np.nan
        else:
            imf = method_thetas[0]
            if imf == len(method_imfs[0]) - 1:
                imf = -1
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

def calc_rd_ratio(selected_imfs_all, srate):
    rds_all = []
    for method_imfs in selected_imfs_all:
        rds_method = []
        for imf in method_imfs:
            df_shapes = compute_shape_features_custom(imf, srate, f_range = None)
            ratio = (df_shapes['volt_rise'] / df_shapes['volt_decay']).mean()
            rds_method.append(ratio)
        rds_all.append(rds_method)
    return rds_all

def plot_output_methods(selected_imfs_all, srate):
    plots = [plt.plot(imf[:5*srate]) for imf in selected_imfs_all]
    return plots

def single_trial_analysis(trial, srate, maskmethods_list, ensemblemethods_list, freq_edges, pmsi_only = False):
    imfs_methods, masks_methods = run_mask_methods(trial, srate, maskmethods_list)
    for ensemble_config in ensemblemethods_list:
        imfs_methods.append(ensemble_config(trial))
    freq_stats = freqtr_methods(imfs_methods, srate)
    freqs_imfs_all, theta_indices_all, hht_all = calc_imf_freqs_all(freq_stats, freq_edges)
    if pmsi_only == False:
        imfs_methods_ae, selected_imfs_all, selected_freqs_all, theta_indices_ae, selected_hhts = select_imfs(imfs_methods, freqs_imfs_all, theta_indices_all, freq_edges, hht_all, srate)
        pmsis_all = pmsi_all(imfs_methods_ae, theta_indices_ae)
        return imfs_methods_ae, selected_imfs_all, selected_freqs_all, theta_indices_all, selected_hhts, pmsis_all
    else:
        imfs_methods_ae, theta_indices_ae = select_imfs(imfs_methods, freqs_imfs_all, theta_indices_all, freq_edges, hht_all, srate, pmsi_only = True)
        return pmsi_all(imfs_methods_ae, theta_indices_ae)



def trials_analysis(trials_list, maskmethods_list, ensemblemethods_list, method_names, srate, freq_edges, pmsi_only = False):
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
        if pmsi_only == False:
            imfs_methods_ae, selected_imfs_all, selected_freqs_all, _, selected_hhts, pmsis_all = single_trial_analysis(trial, srate, maskmethods_list, ensemblemethods_list, freq_edges)
            hhts_trials.append(selected_hhts)
            selected_imfs_trials.append(selected_imfs_all)
            imfs_ae_trials.append(imfs_methods_ae)
            selected_freqs_trials.append(selected_freqs_all)
            
        else:
            pmsis_all = single_trial_analysis(trial, srate, maskmethods_list, ensemblemethods_list, freq_edges, pmsi_only = True)
        pmsis_trials[nth_trial] = np.array(pmsis_all)
    if pmsi_only == False:
        return selected_imfs_trials, imfs_ae_trials, selected_freqs_trials, hhts_trials, pmsis_trials
    else:
        return pmsis_trials

def plot_wf_hht(selected_imfs_all, selected_freqs_all, selected_hhts, method_names, freq_centres, srate ):
    fig, axs = plt.subplots(2, 5, figsize=(35,8))
    fig.tight_layout(pad=5) 
    time_centres = np.arange(2*srate)-.5
    xaxis = np.arange(0,2*srate)/srate
    nth_method = 0
    for imf, method_name, hht in zip(selected_imfs_all, method_names, selected_hhts):
        if len(selected_freqs_all[nth_method]) > 0:
            label = method_name + ', ' + str( round(selected_freqs_all[nth_method][0],2) ) +'Hz'
            if len(selected_freqs_all[nth_method]) > 1:
                for freq in selected_freqs_all[nth_method][1:]:
                    label += ' and ' + str( round(freq,2) ) + 'Hz'
            emd.plotting.plot_hilberthuang(hht, time_centres/srate, freq_centres,
                            cmap='viridis', time_lims = (0,2), freq_lims=(1,20), log_y=False, fig=fig, ax=axs[1, nth_method])
            axs[0, nth_method].plot(xaxis, imf[:2*srate], label = label)
            axs[0, nth_method].legend()
            axs[0, nth_method].set_xlabel('Time (s)')
            axs[0, nth_method].set_ylabel('\u03BCv')
            axs[0, nth_method].set_title(method_name)
        nth_method += 1

def plot_pmsi(pmsis_trials, method_names):
    fig2, axs2 = plt.subplots(1, figsize = (25,10))
    means = np.nanmean(pmsis_trials, axis = 0)
    errors = np.nanstd(pmsis_trials, axis=0)
    xaxis = np.arange(len(method_names))
    axs2.set_ylabel('PMSI')
    axs2.set_xticks(xaxis)
    axs2.set_xticklabels(method_names)
    axs2.set_title('Pseudo mode splitting-index')
    axs2.yaxis.grid(True)
    axs2.bar(xaxis, means, yerr=errors)
