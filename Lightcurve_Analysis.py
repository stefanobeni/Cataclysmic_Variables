"""
Created on Thu Aug 25 12:08:03 2022

@author: Stefano Beni
    Code takes raw ZTF data light curve and uses the Lomb-Scargle Algorithm to first detect periods and then measure
    the period length.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks
from astropy.timeseries import LombScargle
import astropy.units as u

df_original = pd.read_csv("Systems/DIR_NN Ser/ZTF_bjd.csv")

# data = np.genfromtxt("~/PycharmProjects/WD_Project/Public Archive/Single_System_Finder/NN Ser/ZTF_bjd.csv",
# delimiter=',')
# The above turns data directly into numpy array from file (what i do is instead take the csv file and
# THEN convert columns to np)
print(df_original.columns)

filters = df_original.filtercode
for uniquefilt in np.unique(filters):
    print(uniquefilt)
    mask = filters == uniquefilt
    df = df_original[mask]
    # now we do a lomb scargle on each unique filter

    t = df.bjd.to_numpy() * u.day  # array needs to be numpy array to be able to do * u.{unit}
    y = df.mag.to_numpy() * u.mag
    dy = df.magerr.to_numpy() * u.mag

    ax = plt.gca()
    ax.invert_yaxis()
    plt.scatter(t, y, s=15, color="#219EBC")
    plt.errorbar(t, y, dy, ls='', color="#219EBC")
    plt.title("Lightcurve [NN Ser, " + uniquefilt + "]")
    plt.xlabel("Modified Julian Date")
    plt.ylabel("Magnitude")

    plt.show()
    plt.close()

    #####
    '''
    def sinusoid(x, a, b, c):
    return a*np.sin(b*x) + c


    popt, pcov = curve_fit(sinusoid, t, y, p0=[1, 0.676, 16.5])

    t_model = np.linspace(min(t), max(t), 100)
    y_model = sinusoid(t_model, popt[0], popt[1], popt[2])

    plt.plot(t_model, y_model, color='r')
    plt.show()
    '''
    #####
    ls = LombScargle(t, y, dy, nterms=1)

    maxp_minutes = 200 * u.min
    maxp_days = maxp_minutes.to(u.day)  # 0.946
    minf = 1 / maxp_days
    minp_minutes = 50 * u.min
    minp_days = minp_minutes.to(u.day)
    maxf = 1 / minp_days

    frequency, power = ls.autopower(nyquist_factor=1000)  # , minimum_frequency=minf, maximum_frequency=maxf)  # 40)

    peaks, _ = find_peaks(power, height=0.05)
    plt.plot(frequency, power, color="#219EBC")
    plt.plot(frequency[peaks], power[peaks], "x", color='#FB8500')
    plt.title("Lomb-Scargle Periodogram [NN Ser, " + uniquefilt + "]")
    plt.show()
    plt.close()

    # #####################################################################################################

    # print(ls.false_alarm_probability(np.max(power), method='bootstrap'))
    peaks_tmp = np.copy(peaks)

    bf = np.empty(6) * u.day
    best_frequencies = 1/bf

    i = 0
    print(frequency[np.argmax(power)])
    # Selects the largest peak, then removes it from list and then chooses the new highest
    for _ in best_frequencies:
        if len(peaks_tmp) > 0:
            best_frequencies[i] = frequency[peaks_tmp][np.argmax(power[peaks_tmp])]  # #################################
            # print(power[peaks_tmp])
            peaks_tmp = np.delete(peaks_tmp, np.argmax(power[peaks_tmp]))
            i += 1

    print(best_frequencies)

    ######################################################################
    fig, ax = plt.subplots(nrows=2, ncols=3)
    i = 0

    '''
    plt.plot((t_fit * best_frequencies[i]), y_fit, color='r')
    plt.scatter((t * best_frequencies[i]) % 1, y)
    plt.errorbar((t * best_frequencies[i]) % 1, y, dy, ls='')
    '''

    # Runs through subplot cells
    while i < 6:
        # y_fit = ls.model(t_fit, best_frequencies[i])

        plt.title('Phase Folded Lightcurves')

        a = 0
        # ax[1, 2].plot(t_fit, y_fit, color='r')

        while a < 2:
            b = 0
            while b < 3:
                t_fit = np.linspace(0, 1 / best_frequencies[i])  # * u.day
                '''
                y_fit = ls.model(t_fit, best_frequencies[i])
                ax[a, b].set_title("Frequency of " + str(best_frequencies[i]))
                ax[a, b].plot((t_fit * best_frequencies[i]), y_fit, color='r')
                ax[a, b].scatter((t * best_frequencies[i]) % 1, y)
                ax[a, b].errorbar((t * best_frequencies[i]) % 1, y, dy, ls='')
                ax[a, b].set(xlabel='Phase', ylabel='Magnitude')
                '''
                y_fit = ls.model(t_fit, best_frequencies[i])
                p_min = (1 / best_frequencies[i]).to(u.min)  # #######################

                fig.suptitle("Phase Folded Lightcurves")
                ax[a, b].set_title("f=" + str("{0:.3f}".format(best_frequencies[i]))
                                   + ";  p=" + str("{0:.3f}".format(p_min)))
                ax[a, b].plot((t_fit * best_frequencies[i]), y_fit, color='#FB8500')
                ax[a, b].scatter((t * best_frequencies[i]) % 1, y, s=15, alpha=0.9, color="#219EBC")
                ax[a, b].errorbar((t * best_frequencies[i]) % 1, y, dy, ls='', color="#219EBC")
                ax[a, b].invert_yaxis()
                ax[a, b].set(xlabel='Phase', ylabel='Magnitude')
                i += 1
                b += 1
            a += 1
        # ax.set_xlim(0, 0.5)
        # ax.set_ylim(16, 18.5)
    plt.tight_layout()
    plt.gcf().set_size_inches(12, 8)
    plt.show()
