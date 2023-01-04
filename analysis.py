#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Various tools for analysis used by other scripts to
reproduce results from Fabus et al (2021).

Routines:
    corr_ttest
    running_filt
    PMSI
    
@author: MSFabus

"""

import numpy as np
from statsmodels.sandbox.stats.multicomp import multipletests
from scipy import stats
from collections import deque
from bisect import insort, bisect_left
from itertools import islice

    
def corr_ttest(x1, x2, verbose=False, method='two-sided', alpha=0.05, p_lim=0.05):
    """
    Calculates Welch's t-test across 2nd dimension of imputted arrays
    and bonferroni corrects the p values.

    Parameters
    ----------
    x1 : array
        Array 1.
    x2 : array
        Array 2.
    verbose : bool, optional
        Return all p values with rejected rows or just rejected rows.
        The default is False.
    method : string, optional
        'one-sided' or 'two-sided'. The default is 'two-sided'.
    alpha : float, optional
        bonferroni alpha. The default is 0.05.
    p_lim : float, optional
        P value limit for significance. The default is 0.05.

    Returns
    -------
    1D array
        Vector with 1 where H0 was rejected, nan otherwise.

    """
    N = x1.shape[1]
    pvalue = np.zeros(N)*np.nan
    t = np.zeros(N)*np.nan
    H0_rej = np.zeros(N)*np.nan
    
    for i in range(N):
        t[i], pvalue[i] = stats.ttest_ind(x1[:,i], x2[:,i], nan_policy='omit',
                                        equal_var=False)
        
    if method == 'one-sided':
        pvalue[t<0] = 1
        alpha /= 2

    reject, corrpvalue, alphacSidak, alphacBonf = multipletests(pvalue, alpha=alpha, method='bonferroni')
    
    H0_rej[corrpvalue >= p_lim] = np.nan
    H0_rej[corrpvalue < p_lim] = 1
    
    if verbose:
        return [pvalue, corrpvalue, H0_rej]
    else:
        return H0_rej
    
    
def running_filt(x, window_size, mode='mean'):
    """
    Computes running filter of input vector.

    Parameters
    ----------
    x : 1D array
        Input array.
    window_size : int
        Window size for filter
    mode : string, optional
        Type of filter: mean or median. The default is 'mean'.

    Returns
    -------
    1D array
        Filtered array.

    """
    
    from scipy.ndimage.filters import uniform_filter1d
    if mode == 'mean':
        res = uniform_filter1d(x, size=window_size)
        return res
    
    if mode == 'median':
        x= iter(x); d = deque(); s = []; result = []
        for item in islice(x, window_size):
            d.append(item)
            insort(s, item)
            result.append(s[len(d)//2])
        m = window_size // 2
        for item in x:
            old = d.popleft()
            d.append(item)
            del s[bisect_left(s, old)]
            insort(s, item)
            result.append(s[m])
        return result


def PMSI(imf, m, method='both'):
    """
    Computes pseudo-mode mixing index of an intrinsic mode function.

    Parameters
    ----------
    imf : 2D array
        Set of IMFs.
    m : int
        Mode to calculate the PMSI of.
    method : string, optional
        Calculate PMSI as sum of PMSI between mode m and both above / below 
        modes, or only above / below mode. The default is 'both'.

    Returns
    -------
    float
        PMSI calculated.

    """
    if method == 'both':
        abs1 = (imf[:, m].dot(imf[:, m]) + imf[:, m-1].dot(imf[:, m-1]))
        pmsi1 = np.max([np.dot(imf[:, m], imf[:,m-1]) / abs1, 0])
        abs2 = (imf[:, m].dot(imf[:, m]) + imf[:, m+1].dot(imf[:, m+1]))
        pmsi2 = np.max([np.dot(imf[:, m], imf[:,m+1]) / abs2, 0])
        return pmsi1 + pmsi2
    
    if method == 'above':
        abs1 = (imf[:, m].dot(imf[:, m]) + imf[:, m-1].dot(imf[:, m-1]))
        pmsi1 = np.max([np.dot(imf[:, m], imf[:,m-1]) / abs1, 0])
        return pmsi1
    
    if method == 'below':
        abs2 = (imf[:, m].dot(imf[:, m]) + imf[:, m+1].dot(imf[:, m+1]))
        pmsi2 = np.max([np.dot(imf[:, m], imf[:,m+1]) / abs2, 0])
        return pmsi2