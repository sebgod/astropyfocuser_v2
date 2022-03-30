from copy import copy
import cv2 as cv
import multiprocessing as mp
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.stats import gaussian_sigma_to_fwhm
from astropy.table import QTable
from astropy.visualization import simple_norm

from pathlib import Path

from photutils.aperture import CircularAperture, ApertureStats
from photutils.centroids import centroid_quadratic
from photutils.detection import find_peaks, IRAFStarFinder
from photutils.background import MADStdBackgroundRMS

from typing import Dict, Tuple

testDataPath = Path.cwd().joinpath('testdata')

frameSequenceNames = [
    'frame_2022-03-26-1135_6_Luminance_00001.fits',
    'frame_2022-03-26-1137_6_Luminance_00002.fits',
    'frame_2022-03-26-1139_6_Luminance_00003.fits',
    'frame_2022-03-26-1156_5_Luminance_00011.fits',
    'frame_2022-03-26-1158_5_Luminance_00012.fits',
    'frame_2022-03-26-1200_5_Luminance_00013.fits',
    'frame_2022-03-26-1202_5_Luminance_00014.fits',
    'frame_2022-03-26-1204_5_Luminance_00015.fits',
    'frame_2022-03-26-1208_7_Luminance_00017.fits',
    'frame_2022-03-26-1212_7_Luminance_00019.fits',
    'frame_2022-03-26-1214_7_Luminance_00020.fits',
    'frame_2022-03-26-1216_7_Luminance_00021.fits',
    'frame_2022-03-26-1218_7_Luminance_00022.fits',
    'frame_2022-03-26-1220_7_Luminance_00023.fits',
    'frame_2022-03-26-1230_7_Luminance_00028.fits',
    'frame_2022-03-26-1233_6_Luminance_00029.fits',
    'frame_2022-03-26-1235_6_Luminance_00030.fits',
    'frame_2022-03-26-1237_6_Luminance_00031.fits',
]

import time

def time_it(f):
    time_it.active = 0

    def tt(*args, **kwargs):
        time_it.active += 1
        t0 = time.time()
        tabs = '\t'*(time_it.active - 1)
        name = f.__name__
        print('{tabs}Executing <{name}>'.format(tabs=tabs, name=name))
        res = f(*args, **kwargs)
        print('{tabs}Function <{name}> execution time: {time:.3f} seconds'.format(
            tabs=tabs, name=name, time=time.time() - t0))
        time_it.active -= 1
        return res
    return tt

def open_test_file(filePath: Path) -> np.ndarray:
    hdu = fits.open(filePath)[0]
    return hdu.data # hdu.data[200:3944, 200:2622]

def background_substraction(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    bkgrms = MADStdBackgroundRMS()
    cv.medianBlur(data, 3, data)
    std = bkgrms(data)
    median = np.median(data)
    max_x = data.shape[1]
    max_y = data.shape[0]
    return (data - median)[30:max_y-30,30:max_x-30], std

def find_stars_using_peak_brightness(bkg_subst: np.ndarray, threshold: float, box_size: int = 5) -> QTable:
    tbl = find_peaks(bkg_subst, threshold, box_size=box_size, centroid_func=centroid_quadratic)
    tbl.remove_columns(['x_peak', 'y_peak'])
    tbl['peak_value'].info.format = '%.8g'  # for consistent table output
    return tbl

def find_duplicates(tbl : QTable, delta: float = 0.3):
    tbl.sort(['y_centroid', 'x_centroid'])
    tbl_len = len(tbl)
    rows_to_delete = set()
    last_y_val = float('-inf')
    last_y_idx = -1
    for i in range(0, tbl_len):
        row = tbl[i]
        curr_y = row['y_centroid']
        # remove integer based centroid values as they look dodgy
        if np.isnan(curr_y) or (curr_y == int(curr_y)):
            rows_to_delete.add(i)
        elif (last_y_val - delta) < curr_y < (last_y_val + delta):
            curr_x = row['x_centroid']
            # check if corresponding value matches too
            for j in range(last_y_idx, i):
                other = tbl[j]['x_centroid']
                if (other - delta) < curr_x < (other + delta):
                    rows_to_delete.add(i)
                    break
        else:
            last_y_val = curr_y
            last_y_idx = i

    tbl.remove_rows(sorted(rows_to_delete))

def calc_fwhm(tbl : QTable, data : np.ndarray, radius = 4.) -> QTable:
    centroids = np.transpose((tbl['x_centroid'], tbl['y_centroid']))
    aperture = CircularAperture(centroids, radius)

    aperstats = ApertureStats(data, aperture)
    stars = aperstats.to_table()
    star_count = len(stars)
    to_remove = []
    for i in range(0, star_count):
        fwhm = stars[i]['fwhm']
        if np.isnan(fwhm):
            to_remove.append(i)
    
    stars.remove_rows(to_remove)
    stars.show_in_browser()
    return stars

def find_brightest_stars(data : np.ndarray) -> QTable:
    bkg_subst, std = background_substraction(data)
    threshold = 30. * std

    tbl = find_stars_using_peak_brightness(bkg_subst, threshold)

    find_duplicates(tbl, delta = 1.1)

    stars = calc_fwhm(tbl, bkg_subst)
    
    return stars

def display_stars(tbl: QTable, data: np.ndarray):
    positions = np.transpose((tbl['x_centroid'], tbl['y_centroid']))
    peak_apertures = CircularAperture(positions, r=4.)
    norm = simple_norm(data, 'sqrt', percent=99.9)
    plt.imshow(data, cmap='viridis', origin='lower', norm=norm,
            interpolation='nearest')
    peak_apertures.plot(color='#0547f9', lw=1.5)
    plt.xlim(0, data.shape[1] - 1)
    plt.ylim(0, data.shape[0] - 1)
    plt.show()

def analyse_frame(filePath: Path) -> Dict[str, any]:
    testData = open_test_file(filePath)
    stars = find_brightest_stars(testData)
    return {
        'frame': filePath.name,
        'stars': len(stars),
        'median_peak_value': np.median(stars['max']),
        'median_fwhm': np.median(stars['fwhm'])
    }

if __name__ == '__main__':
    # matplotlib.use("TkAgg")
    pool = mp.Pool()

    absFilePathes = [testDataPath / name for name in frameSequenceNames]
    results = pool.map(analyse_frame, absFilePathes)
    pool.close()

    frame_metrics = QTable(
        names=['frame', 'stars', 'median_peak_value', 'median_fwhm'],
        # dtype=[str, np.int32, np.float32, np.float32],
        rows=results,
        copy=False
    )

    print(frame_metrics)
        
    frame_metrics.show_in_browser()