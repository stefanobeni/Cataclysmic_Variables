# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 16:39:06 2022

@author: James Munday
    Original code takes a single set of coordinates and uses them to look for a system that matches those
    coordinates in the ZTF dataset. Code then creates .csv files that store the ZTF data viewed in the R, G, and
    I filters. Requires a IRSA account.
@edited: Stefano Beni
    Improved code can take multiple coordinates at once using a .csv file as input. Edited code also makes use of
    a more appropriate set of coordinates (BJD instead of MJD). Data from all the required systems is stored in a new
    folder called 'Binaries'.
"""

import csv
import datetime
import jdcal
import pandas as pd

import matplotlib.pyplot as plt

import requests
import numpy as np
import os

from astropy.coordinates import SkyCoord
from astropy.coordinates import Angle
from astropy.time import Time

from astropy import units as u
from astroquery.gaia import Gaia


class ZTF(object):
    # create request to ZTF UNFINISHED
    def create_url(ID, CIRCLE, BAND=False, BANDNAME=False, MAG=False, NUM_OBS=False, TIME=False,
                   BAD_CATFLAGS_MASK=False, COLLECTION=False, FORMAT=False):  # POS,
        # unfinished business here
        url = "https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?"
        if ID:
            url += "ID="
            url += str(ID)
            url += "&"

        if CIRCLE:
            RA, Dec, radius = CIRCLE[0], CIRCLE[1], CIRCLE[2]  # radius in deg
            url += "POS=CIRCLE "
            url += str(RA)
            url += " "
            url += str(Dec)
            url += " "
            url += str(radius)
            url += "&"

        # if POS:
        #    POS_ID=str(POS)

        if BAND:  # in nanometres
            minwl, maxwl = BAND[0] / 100, BAND[1] / 100
            url += str(minwl)
            url += "e-7 "
            url += str(maxwl)
            url += "e-7&"

        if BANDNAME:
            url += "BANDNAME="
            url += str(BANDNAME)
            url += "&"

        if MAG:
            minmag, maxmag = MAG[0], MAG[1]
            url += "MAG="
            url += str(minmag)
            url += " "
            url += str(maxmag)
            url += "&"

        if NUM_OBS:
            url += "NUM_OBS="
            url += str(NUM_OBS)
            url += "&"

        if TIME:
            "enter TIME   as  [minT maxT] or as [0 2y]"
            minT = TIME[0]
            maxT = TIME[1]
            if "last" in TIME:
                if "y" in TIME:
                    yplace = TIME.find("y")
                    lastplace = TIME.find("last")
                    numyears = TIME[3:yplace]

                    dt = datetime.datetime.now()
                    mjd = sum(jdcal.gcal2jd(dt.year, dt.month, dt.day)) - 2400000.5

                    MJDmax = mjd
                    MJDmin = mjd - float(numyears) * 365
                else:
                    raise ValueError

            else:
                MJDmax = minT
                MJDmin = maxT
            url += "TIME="
            url += str(MJDmin)
            url += " "
            url += str(MJDmax)
            url += "&"

        if BAD_CATFLAGS_MASK:
            url += "BAD_CATFLAGS_MASK=32768&"

        if COLLECTION:
            None

        return url + "FORMAT=CSV"

    # grab the data

    def get_data(url, auth):
        response = requests.get(url, auth=auth)
        return response

    # save the data to new directory

    def save_data(count, raw_data):

        filename = "ZTF.csv"
        with open(filename, 'w') as f:
            writer = csv.writer(f)
            for line in raw_data.iter_lines():
                writer.writerow(line.decode('utf-8').split(','))

        ZTF.plot_photometry(os.getcwd(), filename)

    def plot_photometry(cwd, filename, saveit=True):
        data = np.genfromtxt(filename, skip_header=1, unpack=True, encoding=None)  # dtype=None)

        with open("ZTF.csv", 'r') as dest_f:
            data_iter = csv.reader(dest_f, quotechar='"')
            next(data_iter)
            data = [data for data in data_iter]

        oid, expid, hjd, mjd, mag, magerr, catflags, filtercode, ra, dec, chi, sharp, filefracday, field, ccdid, qid, limitmag, magzp, magzprms, clrcoeff, clrcounc, exptime, airmass, programid = np.asarray(
            data).T

        if saveit == True:
            plt.clf()

        for filt in np.unique(filtercode):

            mask = ((filtercode == filt) & (((catflags == "0")) | (catflags == "2") | (catflags == "16") | (
                    airmass.astype(np.float64) < 2.5) | (magzprms.astype(np.float64) < 0.04)))

            # remove any flagged data (page 79) https://web.ipac.caltech.edu/staff/fmasci/ztf/ztf_pipelines_deliverables.pdf

            plt.scatter(mjd.astype(float)[mask], mag.astype(float)[mask], label=filt)
            plt.errorbar(mjd.astype(float)[mask], mag.astype(float)[mask], yerr=magerr.astype(float)[mask], ls=' ')
            if saveit == True:
                np.savetxt("ZTF_" + str(filt) + ".csv",
                           np.array([mjd.astype(float)[mask], mag.astype(float)[mask], magerr.astype(float)[mask]]).T,
                           fmt="%s")
        if saveit == True:
            plt.legend(loc="upper right")
            plt.xlabel("MJD")
            plt.ylabel("mag")
            plt.gca().invert_yaxis()
            plt.autoscale()
            plt.savefig("ZTFphot.pdf")
            plt.close()


class PTF(object):
    def queryPTF(RA, Dec, rad='1'):
        # you can use astropy IRSA to do this, but I found that it timed out more often for some reason.
        # WISE currently uses this, but I might change in future to a manual search
        # https://astroquery.readthedocs.io/en/latest/ipac/irsa/irsa.html

        RA = RA.split(":")
        Dec = Dec.split(":")

        if len(RA[0]) == 1:
            RA[0] = "0" + RA[0]
        if len(RA[1]) == 1:
            RA[1] = "0" + RA[1]
        CheckSec = RA[2].split(".")
        if len(CheckSec[0]) == 1:
            RA[2] = "0" + RA[2]

        if len(Dec[0]) == 1:
            Dec[0] = "0" + Dec[0]
        if len(Dec[1]) == 1:
            Dec[1] = "0" + Dec[1]
        CheckSec = Dec[2].split(".")
        if len(CheckSec[0]) == 1:
            Dec[2] = "0" + Dec[2]

        requests_string = 'https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-query?catalog=ptf_lightcurves&spatial=cone&radius=' + rad + '&radunits=arcsec&objstr=' + \
                          RA[0] + 'h+' + RA[1] + 'm+' + RA[2] + 's+' + Dec[0] + 'd+' + Dec[1] + 'm+' + Dec[
                              2] + 's&outfmt=1'
        res = requests.get(
            requests_string)  # selcols=ra,dec,w1mpro,w1sigmpro,w1snr,w2mpro,w2sigmpro,w2snr,w3mpro,w3sigmpro,w3snr,w4mpro,w4sigmpro,w4snr')

        with open("PTF.txt", "w") as f:
            f.write(res.text)

    def splitRandG_PTF():
        if "PTF.txt" in os.listdir(os.getcwd()):
            b = np.loadtxt("PTF.txt", unpack=True, skiprows=101, dtype=str)
            try:
                qual = b[-4] == "1"
                obsmjd = b[0].astype(np.float64)[qual]
                mag_autocorr = b[1].astype(np.float64)[qual]
                magerr_auto = b[2].astype(np.float64)[qual]

                r = b[8] == "2"
                g = b[8] == "1"
                if len(obsmjd[r]) > 0:
                    np.savetxt("rPTF.csv", np.array([obsmjd[r], mag_autocorr[r], magerr_auto[r]]).T)
                if len(obsmjd[g]) > 0:
                    np.savetxt("gPTF.csv", np.array([obsmjd[g], mag_autocorr[g], magerr_auto[g]]).T)

            except:
                None


def Coords_Doc(RAdeg, Decdeg, RA, Dec, system_name):  # by Stefano
    f = open("Coords_" + system_name + ".txt", "w")

    f.write("Right Ascension in degrees: " + str(RAdeg) + "\n")
    f.write("Declination in degrees: " + str(Decdeg) + "\n")
    f.write("Right ascension in hours:minutes:seconds: " + str(RA) + "\n")
    f.write("Declination in degrees:arcminutes:arcseconds: " + str(Dec) + "\n")

    f.close()
    return None


def get_gaia_coords(ra, dec):  # by Stefano
    Gaia.MAIN_GAIA_TABLE = "gaiaedr3.gaia_source"
    Gaia.ROW_LIMIT = 3

    # ra=08:06:22.84, dec=+15:27:31.5
    # ra=280, dec=-60

    coord = SkyCoord(ra=ra, dec=dec, unit=(u.hourangle, u.degree), frame='icrs')
    radius = u.Quantity(5.0, u.arcsec)
    j = Gaia.cone_search(coord, radius, columns=["ra", "dec"])
    r = j.get_results()
    # r.pprint()

    ra_gaia_deg = r["ra"][0]
    dec_gaia_deg = r["dec"][0]

    return ra_gaia_deg, dec_gaia_deg


def Final_ZTF(RADec, RAdeg, Decdeg, RA, Dec, system_name):
    ztfUsername = '****'
    # your IRSA logon details    https://irsa.ipac.caltech.edu/    or
    # https://irsa.ipac.caltech.edu/account/signon/login.do?josso_back_to=https://irsa.ipac.caltech.edu/frontpage/
    ztfPassword = '****'
    # your password

    # getPTF
    Coords_Doc(RAdeg, Decdeg, RA, Dec, system_name)
    if Decdeg > -34:
        PTF.queryPTF(RA, Dec)
        PTF.splitRandG_PTF()
        # getZTF
        urlZTF = ZTF.create_url(None, [RAdeg, Decdeg, 3 / 3600], BAD_CATFLAGS_MASK=True)  # radius in deg
        try:
            ZTF.save_data(RADec, ZTF.get_data(urlZTF, (ztfUsername, ztfPassword)))
        except Exception as e:
            print(e)

# Final_ZTF is the procedure to access the ZTF data. 3/3600 is saying that I want a 3 arcsecond
# search radius. This number can be changed, particularly if you have a star with very high proper motion.
# If you go for too high a number, you will include other stars with the one you want.


############################################################################
# Main addition by Stefano Beni

# Runs through csv file to read coordinates of all required systems.
def run_thru_csv(file_name):
    try:
        os.mkdir("Binaries")
        os.chdir("Binaries")
    except Exception as err:
        print(err)

    k = open("Failures", "a")

    total_fails = 0
    mjd_to_bjd_fails = 0
    for index, row in file_name.iterrows():
        fail_count = 0
        System_Name = row['Name']
        print(System_Name)
        k.write(str(System_Name) + "\n")
        RADec = row['RADec']
        RA = "NA"
        Dec = "NA"
        RAdeg = "NA"
        Decdeg = "NA"

        # noinspection PyBroadException
        try:
            RA = RADec.split(" ")[0]
            Dec = RADec.split(" ")[1]
            print(RA)
            k.write("RA: " + str(RA) + "\n")
            print(Dec)
            k.write("Dec: " + str(Dec) + "\n")
        except:
            print("COORDINATES NOT AVAILABLE")
            fail_count += 1

        try:
            RAdeg = get_gaia_coords(RA, Dec)[0]
            print(RAdeg)
            k.write("RAdeg: " + str(RAdeg) + "\n")
        except:
            print("FAILED TO GET GAIA RIGHT ASCENSION")
            fail_count += 1

        try:
            Decdeg = get_gaia_coords(RA, Dec)[1]
            print(Decdeg)
            k.write("Decdeg: " + str(Decdeg) + "\n")
        except:
            print("FAILED TO GET GAIA DECLINATION")
            fail_count += 1
        print(str(fail_count) + "\n")
        k.write("fail_count: " + str(fail_count) + "\n\n")

        # Stores data in folders

        try:
            os.mkdir("DIR_" + str(System_Name))
        except Exception as err:
            print(err)
        os.chdir("DIR_" + str(System_Name))

        if fail_count == 0:
            Final_ZTF(RADec, RAdeg, Decdeg, RA, Dec, System_Name)
            from astropy.coordinates import EarthLocation

            ZTFloc = EarthLocation.from_geodetic(lat=33.3563, lon=-116.8650,
                                                 height=1712)
            # the earth location of the palomar telescope, where ZTF data is from
            try:
                df = pd.read_csv("ZTF.csv")

                mjd = df.mjd.to_numpy()[:-1]  # np.array([0, 0, 0, 0, 0, 0, 0])  # your mjd times
                bjd = jd_corr(mjd, RAdeg, Decdeg, ZTFloc, jd_type='bjd').value  # converted to bjd

                i = 0
                for element in df.mjd:
                    if i < len(df.mjd) - 1:
                        df.iloc[i, df.columns.get_loc('mjd')] = bjd[i]
                        i += 1
                df.rename(columns={'mjd': 'bjd'}, inplace=True)
                df.to_csv('ZTF_bjd.csv')
            except:
                print("COULD NOT UPDATE MJD TO BJD")
                mjd_to_bjd_fails += 1
                total_fails += 1
        else:
            total_fails += 1
        os.chdir("..")
    print("Unable to convert mjd to bjd " + str(mjd_to_bjd_fails) + " times.")
    print(str(total_fails) + " systems out of 104 were NOT extracted")
    f.write(str(total_fails) + " systems out of 104 were NOT extracted")
    f.close()
    return None


############################################################################


# You can use the functions below, ra_dec_deg_to_hr AND ra_dec_hr_to_deg, to convert how you prefer.
# I've put everything here for completeness


def ra_dec_hr_to_deg(ra, dec):
    # string formats "00:00:00 +00:00:00"
    c = SkyCoord(ra, dec, frame='icrs', unit=(u.hourangle, u.deg))
    return c.ra.deg, c.dec.deg


def ra_dec_deg_to_hr(RAdeg, Decdeg):
    RA = Angle(RAdeg / 15 * u.deg).to_string(unit=u.degree, sep=':')
    Dec = Angle(Decdeg * u.deg).to_string(unit=u.degree, sep=':')
    return RA, Dec


# REMEMBER IF YOU DO PERIOD SEARCHING TO CONVERT MJD TO BJD, like

def jd_corr(mjd, ra, dec, loc, jd_type='bjd'):
    # in this script, I work in BJD UTC. Never is it BJD TDB. I want to change this
    target = SkyCoord(ra, dec, unit=(u.deg, u.deg), frame='icrs')
    jd = Time(mjd, format='mjd', scale='utc', location=loc)
    if jd_type == 'bjd':
        corr = jd.light_travel_time(target, kind='barycentric')
    elif jd_type == 'hjd':
        corr = jd.light_travel_time(target, kind='heliocentric')
    new_jd = jd + corr
    return new_jd


#############################################################################

binaries = pd.read_csv("AM_CVn_population_Proper_AM_CVns.csv")

run_thru_csv(binaries)

#############################################################################
