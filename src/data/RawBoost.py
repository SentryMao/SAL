import copy

import numpy as np
from scipy import signal

"""
   Hemlata Tak, Madhu Kamble, Jose Patino, Massimiliano Todisco, Nicholas Evans.
   RawBoost: A Raw Data Boosting and Augmentation Method applied to Automatic 
   Speaker Verification Anti-Spoofing.
   In Proc. ICASSP 2022, pp:6382--6386.
"""


def process_Rawboost_feat(feat, sr, algo):
    if algo == 1:
        feat = LnL_convolutive_noise(feat, sr)
    elif algo == 2:
        feat = ISD_additive_noise(feat)
    elif algo == 3:
        feat = SSI_additive_noise(feat, sr)
    elif algo == 4:  # 1+2+3
        feat = LnL_convolutive_noise(feat, sr)
        feat = ISD_additive_noise(feat)
        feat = SSI_additive_noise(feat, sr)
    elif algo == 5:  # 1+2
        feat = LnL_convolutive_noise(feat, sr)
        feat = ISD_additive_noise(feat)
    elif algo == 6:  # 1+3
        feat = LnL_convolutive_noise(feat, sr)
        feat = SSI_additive_noise(feat, sr)
    elif algo == 7:  # 2+3
        feat = ISD_additive_noise(feat)
        feat = SSI_additive_noise(feat, sr)
    elif algo == 8:  # 1||2
        feat1 = LnL_convolutive_noise(feat, sr)
        feat2 = ISD_additive_noise(feat1)
        feat = feat1 + feat2
        feat = normWav(feat, 0)
    else:
        feat = feat
    return feat


def randRange(x1, x2, integer):
    y = np.random.uniform(low=x1, high=x2, size=(1,))
    if integer:
        y = int(y)
    return y


def normWav(x, always):
    if always:
        x = x / np.amax(abs(x))
    elif np.amax(abs(x)) > 1:
        x = x / np.amax(abs(x))
    return x


def genNotchCoeffs(nBands=5, minF=20, maxF=8000, minBW=100, maxBW=1000,
                   minCoeff=10, maxCoeff=100, minG=0, maxG=0, fs=16000):
    b = 1
    for i in range(0, nBands):
        fc = randRange(minF, maxF, 0)
        bw = randRange(minBW, maxBW, 0)
        c = randRange(minCoeff, maxCoeff, 1)

        if c / 2 == int(c / 2):
            c = c + 1
        f1 = fc - bw / 2
        f2 = fc + bw / 2
        if f1 <= 0:
            f1 = 1 / 1000
        if f2 >= fs / 2:
            f2 = fs / 2 - 1 / 1000
        b = np.convolve(signal.firwin(c, [float(f1), float(f2)],
                                      window='hamming', fs=fs), b)

    G = randRange(minG, maxG, 0)
    _, h = signal.freqz(b, 1, fs=fs)
    b = pow(10, G / 20) * b / np.amax(abs(h))
    return b


def filterFIR(x, b):
    N = b.shape[0] + 1
    xpad = np.pad(x, (0, N), 'constant')
    y = signal.lfilter(b, 1, xpad)
    y = y[int(N / 2):int(y.shape[0] - N / 2)]
    return y


# Linear and non-linear convolutive noise
def LnL_convolutive_noise(x, fs, N_f=5, nBands=5, minF=20, maxF=8000, minBW=100,
                          maxBW=1000,
                          minCoeff=10, maxCoeff=100, minG=0, maxG=0,
                          minBiasLinNonLin=5,
                          maxBiasLinNonLin=20):
    y = [0] * x.shape[0]
    for i in range(0, N_f):
        if i == 1:
            minG = minG - minBiasLinNonLin
            maxG = maxG - maxBiasLinNonLin
        b = genNotchCoeffs(nBands, minF, maxF, minBW, maxBW, minCoeff, maxCoeff,
                           minG, maxG, fs)
        y = y + filterFIR(np.power(x, (i + 1)), b)
    y = y - np.mean(y)
    y = normWav(y, 0)
    return y


# Impulsive signal dependent noise
def ISD_additive_noise(x, P=10, g_sd=2):
    beta = randRange(0, P, 0)

    y = copy.deepcopy(x)
    x_len = x.shape[0]
    n = int(x_len * (beta / 100))
    p = np.random.permutation(x_len)[:n]
    f_r = np.multiply(((2 * np.random.rand(p.shape[0])) - 1),
                      ((2 * np.random.rand(p.shape[0])) - 1))
    r = g_sd * x[p] * f_r
    y[p] = x[p] + r
    y = normWav(y, 0)
    return y


# Stationary signal independent noise
def SSI_additive_noise(x, fs, SNRmin=10, SNRmax=40, nBands=5, minF=20,
                       maxF=8000, minBW=100, maxBW=1000,
                       minCoeff=10, maxCoeff=100, minG=0, maxG=0):
    noise = np.random.normal(0, 1, x.shape[0])
    b = genNotchCoeffs(nBands, minF, maxF, minBW, maxBW, minCoeff, maxCoeff,
                       minG, maxG, fs)
    noise = filterFIR(noise, b)
    noise = normWav(noise, 1)
    SNR = randRange(SNRmin, SNRmax, 0)
    noise = noise / np.linalg.norm(noise, 2) * np.linalg.norm(x, 2) \
            / 10.0 ** (0.05 * SNR)
    x = x + noise
    return x
