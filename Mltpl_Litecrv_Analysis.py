"""
Created on Thu Aug 25 12:05:34 2022

@author: Stefano Beni
    Code takes raw ZTF data light curve and uses the Lomb-Scargle Algorithm to first detect periods and then measure
    the period length. Multiple light curves can be analysed with the use of a .csv file.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks
from astropy.timeseries import LombScargle
import astropy.units as u
from astropy.stats import sigma_clip
import os

os.chdir("Systems")
data_access = pd.read_csv("Book.csv")

for index, row in data_access.iterrows():
    print(data_access.Name[index])

    df_original = np.empty(1)

    try:
        df_original = pd.read_csv("DIR_" + str(data_access.Name[index]) + "/ZTF_bjd.csv")  # Binaries_BJD/
    except Exception as err:
        print("System does not have light curve")
        print("File does not exist DIR_" + str(data_access.Name[index]) + "/ZTF_bjd.csv")

    try:
        os.mkdir("LA_" + str(data_access.Name[index]))  # LA = Lightcurve Analysis
        os.chdir("LA_" + str(data_access.Name[index]))
    except Exception as err:
        print(err)

    f = open("Lightcurve_Analysis_Specs_" + str(data_access.Name[index]), "a")

    # Another option:
    # data = np.genfromtxt("~/PycharmProjects/WD_Project/Public Archive/Single_System_Finder/NN Ser/ZTF_bjd.csv",
    # delimiter=',')
    # genfromtxt runs data directly into numpy array from file
    # (what I do is instead take the csv file and THEN convert columns to np)

    if len(df_original) > 1:
        filters = df_original.filtercode
        for uniquefilt in np.unique(filters):
            print(uniquefilt)
            f.write(str(uniquefilt) + "\n")

            mask = filters == uniquefilt
            df = df_original[mask]

            # now we run the Lomb-Scargle algorithm on each unique filter

            t = df.bjd.to_numpy() * u.day  # array needs to be numpy array to be able to do * u.{unit}
            y = df.mag.to_numpy() * u.mag
            dy = df.magerr.to_numpy() * u.mag

            # check if system is outbursting
            '''
            print(min(y))
            print(np.mean(y))
            print(np.mean(y) - min(y))
            print(np.mean(sigma_clip(y, cenfunc='mean')))
            print(np.mean(sigma_clip(y, cenfunc='mean')) - min(y))
            '''
            if np.mean(sigma_clip(y, cenfunc='mean')) - min(y) < 6 * u.mag:
                ax = plt.gca()
                ax.invert_yaxis()
                plt.scatter(t, y, s=15, color="#219EBC")
                plt.errorbar(t, y, dy, ls='', color="#219EBC")
                plt.title("Lightcurve [" + data_access.Name[index] + ", " + uniquefilt + "]")
                plt.xlabel("Barycentric Julian Date [d]")
                plt.ylabel("Magnitude")
                # plt.xlim(58300, 58620)

                plt.savefig("Lightcurve_" + uniquefilt + ".pdf")
                plt.show()
                plt.close()

                '''
                def sinusoid(x, a, b, c):
                return a*np.sin(b*x) + c
            
                popt, pcov = curve_fit(sinusoid, t, y, p0=[1, 0.676, 16.5])
            
                t_model = np.linspace(min(t), max(t), 100)
                y_model = sinusoid(t_model, popt[0], popt[1], popt[2])
            
                plt.plot(t_model, y_model, color='r')
                '''

                # ####################################################################################################

                ls = LombScargle(t, y, dy)
                maxp_minutes = 78 * u.min  # 78 ############################# Maximum measured period
                maxp_days = maxp_minutes.to(u.day)
                minf = 1/maxp_days
                minp_minutes = 35 * u.min  # 1 ############################# Minimum measured period
                minp_days = minp_minutes.to(u.day)
                maxf = 1/minp_days

                # Setting limits does not always increase fitting power, but in general it does.
                # Correct limits are more efficient than higher Nyquist factor at finding correct system frequency.
                frequency, power = ls.autopower(minimum_frequency=minf, maximum_frequency=maxf)  # nyquist_factor=1000,
                # plt.plot(frequency, power)
                # #plt.show()

                # ####################################################################################################

                peaks, _ = find_peaks(power)  # , height=0.1)

                fa_prob = ls.false_alarm_probability(np.max(power), method='bootstrap')
                print("False alarm probability is: " + str(fa_prob))
                f.write("False alarm probability is: " + str(fa_prob) + "\n")
                print("Maximum power of periodogram is: " + str(max(power)))
                f.write("Maximum power of periodogram is: " + str(max(power)) + "\n")
                peaks_tmp = np.copy(peaks)

                best_frequencies = np.empty(6) * 1/u.day
                best_powers = np.empty(6)

                # print(frequency[np.argmax(power)])
                period_minutes = (1/frequency[np.argmax(power)]).to(u.min)
                print("Best period is: " + str("{0:.3f}".format(period_minutes)))
                f.write("Best period is: " + str("{0:.3f}".format(period_minutes)) + "\n\n")

                i = 0
                # Selects largest peak, then removes it from list and then chooses the new highest
                for _ in best_frequencies:
                    if len(peaks_tmp) > 0:
                        best_frequencies[i] = frequency[peaks_tmp][np.argmax(power[peaks_tmp])]  # [peaks_tmp])]
                        best_powers[i] = np.max(power[peaks_tmp])
                        # print(power[peaks_tmp])
                        peaks_tmp = np.delete(peaks_tmp, np.argmax(power[peaks_tmp]))
                        i += 1
                # print(best_frequencies)
                print()

                plt.plot(frequency, power, color="#219EBC")
                plt.plot(best_frequencies, best_powers, 'x', color='#FB8500')
                plt.xlabel("Frequency [1/d]")
                plt.ylabel("Power")
                plt.title("Lomb-Scargle Periodogram [" + data_access.Name[index] + ", " + uniquefilt + "]")

                plt.savefig("LS_Periodogram_" + uniquefilt + ".pdf")
                plt.show()
                plt.close()

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
                    # plt.title("Phase Folded Lightcurves [" + data_access.Name[index] + ", " + uniquefilt + "]")

                    a = 0
                    # ax[1, 2].plot(t_fit, y_fit, color='r')

                    while a < 2:
                        b = 0
                        while b < 3:
                            t_fit = np.linspace(0, 1 / best_frequencies[i])  # * u.day
                            # ls = LombScargle(t, y, dy)
                            y_fit = ls.model(t_fit, best_frequencies[i])
                            p_min = (1/best_frequencies[i]).to(u.min)  # #######################

                            fig.suptitle("Phase Folded Lightcurves [" + data_access.Name[index] + ", "
                                         + uniquefilt + "]")
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
                plt.savefig("Phase_Folded_Lightcurves_" + uniquefilt + ".pdf")
                plt.show()
                plt.close()
            else:
                print("System is outbursting in " + uniquefilt)
                f.write("System is outbursting in " + uniquefilt + "\n")
    f.close()
    os.chdir("..")
os.chdir("..")
