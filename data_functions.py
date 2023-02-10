import math
from pathlib import Path
from scipy import signal
from sklearn import preprocessing
import os
import numpy as np
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tifffile import imread
import pandas as pd
from typing import NamedTuple
from keras.preprocessing.image import ImageDataGenerator

blurry_image_boundary_width = 0

if os.name == 'nt':  # windows
    global_data_folder = r"F:/FromTelework/data/processed"
    global_result_folder = r"F:/FromTelework/data/processed/results"
    data_csv_postfix = "work"  # or "scinet"
else:  # linux
    global_data_folder = r"/project/hmi_deeplearning/data/"
    global_result_folder = r"/project/hmi_deeplearning/results"
    data_csv_postfix = "scinet"


def msc(x, xref=None):
    """Multiplicative Scatter Correction converted from
            Matlab code by Cleiton A. Nunes

    x_msc = msc(x,xref)

    input
    x (samples x variables)      spectra to correct
    xref (1 x variables)         reference spectra (in general mean(x) is used)

    Output
    x_msc (samples x variables)  corrected spectra
    """
    m, n = x.shape

    if xref is None:
        rs = x.mean(axis=0)
    else:
        rs = xref

    cw = np.ones((1, n))
    mz = np.hstack((np.ones((n,1)), np.reshape(rs, (n, 1))))
    mm, nm = mz.shape
    wmz = mz * (np.transpose(cw) @ np.ones((1, nm)))  # '*' elementwise multiply and '@' matrix multiply
    wz = x * (np.ones((m, 1)) @ cw)
    z = np.transpose(wmz) @ wmz
    z = z.astype(np.float)
    u, s, v = np.linalg.svd(z)
    sd = s  # instead of np.transpose(np.diag(s))
    cn = pow(10, 12)
    ms = sd[0] / math.sqrt(cn)
    cs = [max(sdi, ms) for sdi in sd]
    cz = u @ (np.diag(cs)) @ np.transpose(v)
    zi = np.linalg.inv(cz)
    b = zi @ np.transpose(wmz) @ np.transpose(wz)
    B = np.transpose(b)
    x_msc = x
    p = np.reshape(B[:, 0], (B.shape[0],1))
    x_msc = x_msc - (p @ np.ones((1, mm)))
    p = np.reshape(B[:, 1], (B.shape[0],1))
    x_msc = x_msc / (p @ np.transpose(np.ones((mm, 1))))
    return x_msc


def snv(input_data):
    """Define a new array and populate it with the corrected data by Daniel Pelliccia"""
    data_snv = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        data_snv[i, :] = (input_data[i, :] - np.mean(input_data[i, :])) / np.std(input_data[i, :], ddof=1)
    return data_snv


def zsnv(input_data):
    from scipy import stats
    data_snv = snv(input_data)
    data_zsnv = stats.zscore(data_snv)
    return data_zsnv


def snv_row(input_data):
    """Define a new array and populate it with the corrected data by Daniel Pelliccia"""
    data_snv = (input_data - np.mean(input_data)) / np.std(input_data, ddof=1)
    return data_snv

def difference(data, gap, direction):
    """
    DIFFERENCE implements the first order derivative.

        out = Difference( data, gap, direction)

        The first "gap" points are discard.
        Thus, the size of out is (n-gap).
    """
    if len(data.shape) != 2 or gap <= 0:
        raise ValueError('not supported dimension. should be 2')
    out = None
    if min(data.shape) == 1:
            out = data[gap:] - data[:-gap]
    else:
        if direction == 'column':
            out = data[gap:, :] - data[:-gap, :]
        elif direction == 'row':
            out = data[:, gap:] - data[:, :-gap]
        else:
            raise ValueError('not supported direction')
    return out


def detrend_polyfit(X, order=2):
    """
    DETRENDPOLYFIT Removes a non-linear trend from each column of a matrix. The nonlinearity is
       estimated with polynomial fit.

        Y = detrendpolyfit(X, *varargin)

        X is the input matrix.
        varargin is an integer number representing polynomial order. Its default value is 2.

        Returns values subtracted polynomial fit of order varargin from the input data X
    """
    if len(X.shape) < 2:
        X = np.reshape(X, (1, len(X)))
    Y = np.zeros(X.shape)
    for i in range(X.shape[1]):
        x = X[:, i]
        rng = [z+1 for z in range(len(x))]
        p = np.polyfit(np.transpose(rng), x[:], order)
        Y[:, i] = x[:]-np.polyval(p, np.transpose(rng))
    return Y


def deriv_norrisgap(X, gap, direction):
    """
    first derivative using Norris Gap
    :param X: m x n matrix
    :param gap: gap < n
    :param direction: 'row' or 'column'
    :return: Xout
    """
    if len(X.shape) != 2:
        raise ValueError('not supported dimension. should be 2')

    gap = gap + 1
    if min(X.shape) == 1:
        out = X[gap:] - X[:-gap]
    else:
        if direction == 'column':
            out = X[gap:, :] - X[:-gap, :]
        elif direction == 'row':
            out = X[:, gap:] - X[:, :-gap]

    out = out / gap
    Xout = rearrange_norrisgap(X, out)
    return Xout


def rearrange_norrisgap(X, out):
    """Re-arrage norris gap output"""
    M1, N1 = X.shape
    M2, N2 = out.shape
    if M2 < M1:
        # vertical direction
        offset = M1 - M2
        numPts = M2
        direction = 0
    elif N2 < N1:
        # horizontal direction
        offset = N1 - N2
        numPts = N2
        direction = 1
    else:
        raise ValueError('Error in Derivative_NorisGap')
    pos1 = np.int32(offset / 2)
    pos2 = np.int32(pos1 + numPts)
    Xout = np.float64(np.zeros_like(X))
    if direction == 0:
        Xout[pos1:pos2, :] = out
    elif direction == 1:
        Xout[:, pos1:pos2] = out
    return Xout


def expandarr(x,k):
    #make it work for 2D or nD with axis
    kadd = k
    if np.ndim(x) == 2:
        kadd = (kadd, np.shape(x)[1])
    return np.r_[np.ones(kadd)*x[0],x,np.ones(kadd)*x[-1]]

def movmoment(x, k, windowsize=3, lag='lagged'):
    """non-central moment
    Parameters
    ----------
    x : array
       time series data
    windsize : int
       window size
    lag : 'lagged', 'centered', or 'leading'
       location of window relative to current position

    Returns
    -------
    mk : array
        k-th moving non-central moment, with same shape as x
    Notes
    -----
    If data x is 2d, then moving moment is calculated for each
    column.
    """

    from scipy import signal
    windsize = windowsize
    #if windsize is even should it raise ValueError
    if lag == 'lagged':
        #lead = -0 + windsize #windsize//2
        lead = -0# + (windsize-1) + windsize//2
        sl = slice((windsize-1) or None, -2*(windsize-1) or None)
    elif lag == 'centered':
        lead = -windsize//2  #0#-1 #+ #(windsize-1)
        sl = slice((windsize-1)+windsize//2 or None, -(windsize-1)-windsize//2 or None)
    elif lag == 'leading':
        #lead = -windsize +1#+1 #+ (windsize-1)#//2 +1
        lead = -windsize +2 #-windsize//2 +1
        sl = slice(2*(windsize-1)+1+lead or None, -(2*(windsize-1)+lead)+1 or None)
    else:
        raise ValueError

    avgkern = (np.ones(windowsize)/float(windowsize))
    xext = expandarr(x, windsize-1)
    # Note: expandarr increases the array size by 2*(windsize-1)
    # sl = slice(2*(windsize-1)+1+lead or None, -(2*(windsize-1)+lead)+1 or None)
    # print(sl)

    if xext.ndim == 1:
        return np.correlate(xext**k, avgkern, 'full')[sl]
        # return np.correlate(xext**k, avgkern, 'same')[windsize-lead:-(windsize+lead)]
    else:
        # print(xext.shape)
        # print(avgkern[:,None].shape)

        # try first with 2d along columns, possibly ndim with axis
        return signal.correlate(xext**k, avgkern[:,None], 'full')[sl, :]


def movmean(x, windowsize=3, mode="same"):
    """moving window mean
    Parameters
    ----------
    x : array
       time series data
    windsize : int
       window size
    mode : string
        "same" - output array has same size as input
        "valid" - output array has reduced ize from input containing non-truncated values
    Returns
    -------
    mk : array
        moving mean, with same shape as x
    Notes
    -----
    When mode="same"
    The window size is automatically truncated at the endpoints
    when there are not enough elements to fill the window.
    When the window is truncated, the average is taken over
    only the elements that fill the window.
    """
    if type(x) is not np.ndarray:
        raise ValueError("input should be numpy ndarray!")

    lag = 'centered'
    ret = movmoment(x, 1, windowsize=windowsize, lag=lag)

    # to handle truncated window operations correctly
    n = windowsize//2
    if x.ndim == 2 and min(x.shape) > 1:
        for i in range(n):
            ret[i, :] = np.mean(x[:(i + n + 1), :], axis=0)
            ret[-(i + 1), :] = np.mean(x[-(i + n + 1):, :], axis=0)
    elif x.ndim == 1 or (x.ndim == 2 and min(x.shape) == 1):
        for i in range(n):
            ret[i] = np.mean(x[:(i + n + 1)])
            ret[-(i + 1)] = np.mean(x[-(i + n + 1):])
        pass
    else:
        raise ValueError("input dimension should be <= 2!")

    # remove values calculated with truncated window size in "valid" mode
    if mode == "valid":
        if x.ndim == 2 and min(x.shape) > 1:
            ret = ret[n:-n,:]
        elif x.ndim == 1 or (x.ndim == 2 and min(x.shape) == 1):
            ret = ret[n:-n]
    return ret


def msc_diff_ma(x, gap=7):
    """MSC+Difference+Moving average"""
    res = msc(x)
    res = difference(res, gap, 'row')
    res = movmean(res.T, gap, 'same').T
    return res


def msc_diff(x, gap=7):
    """MSC+Difference+Moving average"""
    res = msc(x)
    res = difference(res, gap, 'row')
    return res


def msc_ma(x, gap=7):
    """MSC+Moving average"""
    res = msc(x)
    res = movmean(res.T, gap, 'same').T
    return res


def snvd_diff_ma(x, gap=7):
    """SNV+Detrending+Difference+Moving average"""
    res = snv(x)
    res = signal.detrend(res)
    res = difference(res, gap, 'row')
    res = movmean(res.T, gap, 'same').T
    return res


def snvd_diff(x, gap=7):
    """SNV+Detrending+Difference+Moving average"""
    from statsmodels.tsa.tsatools import detrend
    res = snv(x).astype(np.float64)
    res1 = detrend(res, axis=1)
    res2 = difference(res1, gap, 'row')
    return res2


def snvd_ma(x, gap=7):
    """SNV+Detrending++Moving average"""
    res = snv(x)
    res = signal.detrend(res)
    res = movmean(res.T, gap, 'same').T
    return res


def preprocess_spectra(nptable, preprocessing_method=0):
    if preprocessing_method == 0:
        spectra = snv(nptable)
    elif preprocessing_method == 1:
        spectra = snvd_ma(nptable, 7)
    elif preprocessing_method == 2:
        spectra = snvd_diff(nptable, 7)
    elif preprocessing_method == 3:
        spectra = snvd_diff_ma(nptable, 7)
    elif preprocessing_method == 4:
        spectra = msc_ma(nptable, 7)
    elif preprocessing_method == 5:
        spectra = msc_diff(nptable, 7)
    elif preprocessing_method == 6:
        spectra = msc_diff_ma(nptable, 7)
    elif preprocessing_method == -1:
        spectra = nptable
    return np.array(spectra, dtype=float)


def preprocess_method_name(preprocessing_method=0):
    if preprocessing_method == 0:
        name = "SNV"
    elif preprocessing_method == 1:
        name = "SNVD_MA7"
    elif preprocessing_method == 2:
        name = "SNVD_DIFF7"
    elif preprocessing_method == 3:
        name = "SNVD_DIFF7_MA7"
    elif preprocessing_method == 4:
        name = "MSC_MA7"
    elif preprocessing_method == 5:
        name = "MSC_DIFF7"
    elif preprocessing_method == 6:
        name = "MSC_DIFF7_MA7"
    elif preprocessing_method == -1:
        name = "Raw"
    return name


class SplittedImageDataGenerator(NamedTuple):
    """ImageDataGenerators organized for machine learning (classification)"""
    raw_data_path: list
    train: ImageDataGenerator
    validation: ImageDataGenerator
    test: ImageDataGenerator


# For data organization for machine learning
class SplittedData:  # (NamedTuple):
    """Data (np.ndarray) organized for machine learning (classification)"""
    x_train: np.ndarray
    x_validation: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_validation: np.ndarray
    y_test: np.ndarray
    # index_train: np.ndarray
    # index_valid: np.ndarray
    # index_test: np.ndarray


class FusionNetData:  # (NamedTuple):
    """Data necessary to train Fusion-Net
    Attributes:
        spectral: SplittedData containing spectral data
        shape: SplittedData containing morphological properties
        image: SplittedData containing 2D images
    """
    spectral: SplittedData
    shape: SplittedData
    image: SplittedData


def d_resize(image, max_dim=64):  # to resize image
    """resize image to 64x64 by padding: image should be 2D, not 3D, 4D, ..."""
    resized_img = []
    h, w = image.shape[:2]
    top_pad = (max_dim - h) // 2
    bottom_pad = max_dim - h - top_pad
    left_pad = (max_dim - w) // 2
    right_pad = max_dim - w - left_pad
    padding = [(top_pad, bottom_pad), (left_pad, right_pad)]
    # print([max_dim, image.shape])
    image_new = np.pad(np.squeeze(image), padding, mode='constant', constant_values=0)
    return(image_new)


def get_table_aotf_deadlive(data_type=0):
    filename = rf"{global_data_folder}/AOTF_deadlive_Hali/classification/single_cells_{data_csv_postfix}.csv"
    table = pd.read_csv(filename)
    if data_type == 4:
        table = table[table[" Species"] >= data_type]
    elif 4 > data_type > 0:
        table = table[table[" Species"] == data_type]
    elif data_type != 0:
        raise Exception(f"Not valid data_type ({data_type})")
    return table


def read_data_spectral_aotf_deadlive(table=None, split_indice=None, culture_type=0, y_all=None):
    """Get spectral data for training of FPI species classification model"""
    classes = 2
    nbands = 89
    startcol = 17
    index_train_unlabeled = np.array([])
    index_test_unlabeled = np.array([])

    if len(split_indice) == 3:
        index_train, index_valid, index_test = split_indice
    elif len(split_indice) == 5:
        index_train, index_valid, index_test, \
            index_train_unlabeled, index_test_unlabeled = split_indice
    else:
        raise ValueError(f"split_indice needs to have 3 or 5 elements but current one has {len(split_indice)}.")
    nrows = table.shape[0]

    nptable = table.to_numpy()
    if y_all is None:
        y_single_all = nptable[:, -1]  # 1 <= class_label <= 4
        y_all = np_utils.to_categorical(y_single_all-1, classes)  # number of columns: classes

    y_train = y_all[index_train, :]
    y_test = y_all[index_test, :]
    y_valid = y_all[index_valid, :]
    if len(split_indice) == 5:
        y_train_unlabeled = y_all[index_train_unlabeled, :]
        y_test_unlabeled = y_all[index_test_unlabeled, :]

    # get all spectra
    x_all_spec = nptable[:, startcol:(startcol+nbands)].astype(np.float64)

    # preprocess
    x_all_spec = snv(x_all_spec)  # snvd_diff(x_all_spec, 7)

    # this is preprocessing and should be out of this function


    x_all_spec = np.expand_dims(x_all_spec, axis=2)
    x_spec_train = x_all_spec[index_train, :]
    x_spec_test = x_all_spec[index_test, :]
    x_spec_valid = x_all_spec[index_valid, :]

    if len(split_indice) == 5:
        x_spec_train_unlabeled = x_all_spec[index_train_unlabeled, :]
        x_spec_test_unlabeled = x_all_spec[index_test_unlabeled, :]

    # this is preprocessing and should be out of this function
    spec_scaler = preprocessing.StandardScaler(with_mean=True, with_std=True).fit(x_spec_train[:, :, 0])
    x_spec_train[:, :, 0] = spec_scaler.transform(x_spec_train[:, :, 0])
    x_spec_valid[:, :, 0] = spec_scaler.transform(x_spec_valid[:, :, 0])
    x_spec_test[:, :, 0] = spec_scaler.transform(x_spec_test[:, :, 0])

    if len(split_indice) == 5:
        x_spec_train_unlabeled[:, :, 0] = spec_scaler.transform(x_spec_train_unlabeled[:, :, 0])
        x_spec_test_unlabeled[:, :, 0] = spec_scaler.transform(x_spec_test_unlabeled[:, :, 0])

    spectra = SplittedData()
    spectra.x_train = x_spec_train
    spectra.x_validation = x_spec_valid
    spectra.x_test = x_spec_test
    spectra.y_train = y_train
    spectra.y_validation = y_valid
    spectra.y_test = y_test
    if len(split_indice) == 5:
        spectra.x_train_unlabeled = x_spec_train_unlabeled
        spectra.x_test_unlabeled = x_spec_test_unlabeled
        spectra.y_unlabeled = y_train_unlabeled
        spectra.y_unlabeled = y_test_unlabeled

    if split_indice is None:
        spectra.index_train = index_train
        spectra.index_valid = index_valid
        spectra.index_test = index_test
        if len(split_indice) == 5:
            spectra.index_train_unlabeled = index_train_unlabeled
            spectra.index_test_unlabeled = index_test_unlabeled
    return spectra, spec_scaler


def read_data_shape_aotf_deadlive(table=None, split_indice=None, culture_type=0, y_all=None):
    """Get shape data for training of FPI species classification model"""
    classes = 2
    startcol = 8
    endcol = 17
    index_train_unlabeled = np.array([])
    index_test_unlabeled = np.array([])

    if len(split_indice) == 3:
        index_train, index_valid, index_test = split_indice
    elif len(split_indice) == 5:
        index_train, index_valid, index_test, \
            index_train_unlabeled, index_test_unlabeled = split_indice
    else:
        raise ValueError(f"split_indice needs to have 3 or 5 elements but current one has {len(split_indice)}.")
    nrows = table.shape[0]

    nptable = table.to_numpy()
    if y_all is None:
        y_single_all = nptable[:, -1]  # 1 <= class_label <= 4
        y_all = np_utils.to_categorical(y_single_all-1, classes)  # number of columns: classes

    y_train = y_all[index_train, :]
    y_test = y_all[index_test, :]
    y_valid = y_all[index_valid, :]

    if len(split_indice) == 5:
        y_train_unlabeled = y_all[index_train_unlabeled, :]
        y_test_unlabeled = y_all[index_test_unlabeled, :]

    # get all shape
    x_all_shape = nptable[:, startcol:endcol].astype(np.float64)

    x_all_shape = np.expand_dims(x_all_shape, axis=1)
    x_shape_train = x_all_shape[index_train, :]
    x_shape_test = x_all_shape[index_test, :]
    x_shape_valid = x_all_shape[index_valid, :]

    if len(split_indice) == 5:
        x_shape_train_unlabeled = x_all_shape[index_train_unlabeled, :]
        x_shape_test_unlabeled = x_all_shape[index_test_unlabeled, :]

    shape_scaler = preprocessing.StandardScaler().fit(x_shape_train[:, 0, :])
    x_shape_train[:, 0, :] = shape_scaler.transform(x_shape_train[:, 0, :])
    x_shape_valid[:, 0, :] = shape_scaler.transform(x_shape_valid[:, 0, :])
    x_shape_test[:, 0, :] = shape_scaler.transform(x_shape_test[:, 0, :])
    if len(split_indice) == 5:
        x_shape_train_unlabeled[:, 0, :] = shape_scaler.transform(x_shape_train_unlabeled[:, 0, :])
        x_shape_test_unlabeled[:, 0, :] = shape_scaler.transform(x_shape_test_unlabeled[:, 0, :])

    shape_props = SplittedData()
    shape_props.x_train = x_shape_train
    shape_props.x_validation = x_shape_valid
    shape_props.x_test = x_shape_test
    shape_props.y_train = y_train
    shape_props.y_validation = y_valid
    shape_props.y_test = y_test
    if len(split_indice) == 5:
        shape_props.x_train_unlabeled = x_shape_train_unlabeled
        shape_props.x_test_unlabeled = x_shape_test_unlabeled
        shape_props.y_train_unlabeled = y_train_unlabeled
        shape_props.y_test_unlabeled = y_test_unlabeled
    return shape_props, shape_scaler


def read_data_species_aotf_deadlive(table=None, split_indice=None, culture_type=0, y_all=None):
    """Get species data for training of FPI species classification model"""
    classes = 2
    nspecies = 6
    startcol = -2
    index_train_unlabeled = np.array([])
    index_test_unlabeled = np.array([])

    if len(split_indice) == 3:
        index_train, index_valid, index_test = split_indice
    elif len(split_indice) == 5:
        index_train, index_valid, index_test, \
            index_train_unlabeled, index_test_unlabeled = split_indice
    else:
        raise ValueError(f"split_indice needs to have 3 or 5 elements but current one has {len(split_indice)}.")
    nrows = table.shape[0]

    nptable = table.to_numpy()

    species_single_all = nptable[:, startcol]
    species_all = np_utils.to_categorical(species_single_all - 1, nspecies)
    species_train = species_all[index_train, :]
    species_test = species_all[index_test, :]
    species_valid = species_all[index_valid, :]

    if len(split_indice) == 5:
        species_train_unlabeled = y_all[index_train_unlabeled, :]
        species_test_unlabeled = y_all[index_test_unlabeled, :]

    species = SplittedData()
    species.x_train = species_train
    species.x_validation = species_valid
    species.x_test = species_test
    if len(split_indice) == 5:
        species.x_train_unlabeled = species_train_unlabeled
        species.x_test_unlabeled = species_test_unlabeled
    return species


def imagescaler_fit_transform(x_image_train):
    dims = x_image_train.shape
    reshaped = x_image_train.reshape(dims[0], dims[1]*dims[2])
    image_scaler = preprocessing.StandardScaler().fit(reshaped)
    ret = image_scaler.transform(reshaped).reshape(dims)
    return image_scaler, ret


def imagescaler_transform(x_image_train, image_scaler):
    dims = x_image_train.shape
    reshaped = x_image_train.reshape(dims[0], dims[1]*dims[2])
    ret = image_scaler.transform(reshaped).reshape(dims)
    return ret


def read_data_image_aotf_deadlive(table=None, split_indice=None, culture_type=0,
                                  y_all=None, is_image_scale=False):
    """Get image data for training of FPI species classification model"""
    classes = 2
    image_size = 64
    index_train_unlabeled = np.array([])
    index_test_unlabeled = np.array([])

    if len(split_indice) == 3:
        index_train, index_valid, index_test = split_indice
    elif len(split_indice) == 5:
        index_train, index_valid, index_test, \
            index_train_unlabeled, index_test_unlabeled = split_indice
    else:
        raise ValueError(f"split_indice needs to have 3 or 5 elements but current one has {len(split_indice)}.")

    nrows = table.shape[0]

    nptable = table.to_numpy()
    if y_all is None:
        y_single_all = nptable[:, -1]  # 1 <= class_label <= 4
        y_all = np_utils.to_categorical(y_single_all-1, classes)  # number of columns: classes

    y_train = y_all[index_train, :]
    y_test = y_all[index_test, :]
    y_valid = y_all[index_valid, :]
    if len(split_indice) == 5:
        y_train_unlabeled = y_all[index_train_unlabeled, :]
        y_test_unlabeled = y_all[index_test_unlabeled, :]

    # get all image data
    x_all_image = read_image_data_from_table(table, image_size=image_size)
    x_image_train = x_all_image[index_train]
    x_image_test = x_all_image[index_test]
    x_image_valid = x_all_image[index_valid]

    if len(split_indice) == 5:
        x_image_train_unlabeled = x_all_image[index_train_unlabeled, :]
        x_image_test_unlabeled = x_all_image[index_test_unlabeled, :]

    if is_image_scale:
        image_scaler, x_image_train = imagescaler_fit_transform(x_image_train)
        x_image_valid = imagescaler_transform(x_image_valid, image_scaler=image_scaler)
        x_image_test = imagescaler_transform(x_image_test, image_scaler=image_scaler)
        if len(split_indice) == 5:
            x_image_train_unlabeled = imagescaler_transform(x_image_train_unlabeled, image_scaler=image_scaler)
            x_image_test_unlabeled = imagescaler_transform(x_image_test_unlabeled, image_scaler=image_scaler)


    images = SplittedData()
    images.x_train = x_image_train
    images.x_validation = x_image_valid
    images.x_test = x_image_test
    images.y_train = y_train
    images.y_validation = y_valid
    images.y_test = y_test

    if len(split_indice) == 5:
        images.x_train_unlabeled = x_image_train_unlabeled
        images.x_test_unlabeled = x_image_test_unlabeled
        images.y_train_unlabeled = y_train_unlabeled
        images.y_test_unlabeled = y_test_unlabeled

    if is_image_scale:
        return images, image_scaler
    else:
        return images


def read_data_fusionnet_aotf_deadlive(data_type=0):
    """Get data for fusion-net training"""
    table = get_table_aotf_deadlive(data_type=data_type)
    nrows = table.shape[0]

    # split data
    index_all = list(range(nrows))
    index_train, index_test = train_test_split(index_all, test_size=0.15)
    index_train, index_valid = train_test_split(index_train, test_size=0.15)

    spectra, spec_scaler = read_data_spectral_aotf_deadlive(table, [index_train, index_valid, index_test])
    shape_props, shape_scaler = read_data_shape_aotf_deadlive(table, [index_train, index_valid, index_test])
    images = read_data_image_aotf_deadlive(table, [index_train, index_valid, index_test])
    species = read_data_species_aotf_deadlive(table, [index_train, index_valid, index_test])

    fusionnet_data = FusionNetData()
    fusionnet_data.spectral = spectra
    fusionnet_data.shape = shape_props
    fusionnet_data.image = images
    fusionnet_data.species = species
    fusionnet_data.index_train = index_train
    fusionnet_data.index_valid = index_valid
    fusionnet_data.index_test = index_test
    return fusionnet_data, spec_scaler, shape_scaler


def read_data_fusionnet_aotf_deadlive_imagescaler(data_type=0):
    """Get data for fusion-net training for dead/live with image scaler"""
    table = get_table_aotf_deadlive(data_type=data_type)
    nrows = table.shape[0]

    # split data
    index_all = list(range(nrows))
    index_train, index_test = train_test_split(index_all, test_size=0.15)
    index_train, index_valid = train_test_split(index_train, test_size=0.15)

    spectra, spec_scaler = read_data_spectral_aotf_deadlive(table, [index_train, index_valid, index_test])
    shape_props, shape_scaler = read_data_shape_aotf_deadlive(table, [index_train, index_valid, index_test])
    images, image_scaler = read_data_image_aotf_deadlive(table, [index_train, index_valid, index_test],
                                                         is_image_scale=True)
    species = read_data_species_aotf_deadlive(table, [index_train, index_valid, index_test])

    fusionnet_data = FusionNetData()
    fusionnet_data.spectral = spectra
    fusionnet_data.shape = shape_props
    fusionnet_data.image = images
    fusionnet_data.species = species
    fusionnet_data.index_train = index_train
    fusionnet_data.index_valid = index_valid
    fusionnet_data.index_test = index_test
    return fusionnet_data, spec_scaler, shape_scaler, image_scaler


def min_max_normalize(img):
    """Normalize data with its min and max
     Args:
          img: 2D or 3D array
     Output:
          2D or 3D array
    """
    out = (img - img.min()) / (img.max() - img.min())
    return out


def read_image_data_from_table(table, image_size=128):
    nrows = table.shape[0]
    X = []
    print("reading image data...")
    for i in range(nrows):
        if len(table.shape) < 2:
            img_name = rf"{table.iat[4].strip()}/{table.iat[5].strip()}"
        else:
            img_name = rf"{table.iat[i, 4].strip()}/{table.iat[i, 5].strip()}"
        print(img_name)
        img_data = imread(img_name)
        if img_data is not None:
            img_data = min_max_normalize(img_data)
            img_data = d_resize(img_data, image_size)
            X.append(img_data)
    print("reading image data done")
    X = np.asarray(X)
    return X[:, :, :, np.newaxis]
