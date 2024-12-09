import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# import matplotlib
# matplotlib.use('QtAgg')
# plt.ion()


def specAndAutocorr(
    inputSeries,
    fs,
    NFFT=8192,
    specFreqPortion=[0, 100],
    autoFreqPortion=[0, 100],
    specWindowLength=None,
    windowStep=None,
    dynRangePortion=[0, 100],
    autoWindowLength=None,
    meanTimeBounds=None,
    tLim=None,
):

    y = inputSeries.copy()

    if specWindowLength is None:
        specWindowLength = int(y.shape[0] ** (1 / 1.4))

    if autoWindowLength is None:
        autoWindowLength = int(y.shape[0] ** (1 / 1.4))

    if windowStep is None:
        windowStep = int(y.shape[0] ** (1 / 2.5))

    f = fs / 2 * np.linspace(0, 1, int(NFFT / 2 + 1))

    n = np.arange(0, specWindowLength)

    # window=np.exp(-.5*((n-(windowLength-1)/2)/(.4*(windowLength-1)/2))**2)
    # plt.figure()
    # plt.plot(window)
    # plt.show()
    window = sp.signal.windows.gaussian(specWindowLength, std=specWindowLength / 5)
    # window=sp.signal.windows.hamming(specWindowLength)
    # window=sp.signal.windows.hann(specWindowLength)
    # plt.figure()
    # plt.plot(window)
    # plt.show()

    y = [
        y[i : i + specWindowLength]
        for i in range(0, len(y) - specWindowLength, windowStep)
    ]

    y[-1] = np.pad(
        y[-1], (0, len(y[-2]) - len(y[-1])), mode="constant", constant_values=0
    )

    # print(windowLength)
    # print(windowStep)

    # print(y[-4].shape)
    # print(y[-3].shape)
    # print(y[-2].shape)
    # print(y[-1].shape)

    y = np.stack(y).T

    y = sp.fft.fft(y * window[:, None], n=NFFT, axis=0, overwrite_x=True)

    print(f"Full calculated spectrogram matrix is shape {y.shape}")

    i1 = int((specFreqPortion[0] / 200) * y.shape[0])
    i2 = int((specFreqPortion[1] / 200) * y.shape[0] - 1)

    y = 20 * np.log10(abs(y[i1:i2, :]))

    f1 = int((specFreqPortion[0] / 100) * len(f))
    f2 = int((specFreqPortion[1] / 100) * len(f)) - 1

    fPortion = np.linspace(f[f1], f[f2], y.shape[0])

    if tLim is None:
        tSpec = np.linspace(0, len(inputSeries) / fs, y.shape[1])
    else:
        tSpec = np.linspace(tLim[0], tLim[1], y.shape[1])

    if meanTimeBounds is None:
        meanTimeBoundsSpec = [0, len(tSpec)]
    else:
        meanTimeBoundsSpec = [
            np.abs(meanTimeBounds[0] - tSpec).argmin(),
            np.abs(meanTimeBounds[1] - tSpec).argmin(),
        ]

    fullRangeMin = y.min()
    fullRangeMax = y.max()
    dynRange = fullRangeMax - fullRangeMin
    viewingRangeMin = dynRangePortion[0] / 100 * dynRange + fullRangeMin
    viewingRangeMax = dynRangePortion[1] / 100 * dynRange + fullRangeMin

    # plt.figure(figsize=[13,8])
    # plt.pcolormesh(tSpec,fPortion,y-viewingRangeMin,cmap='gist_yarg')
    # plt.xlabel('Time (sec)')
    # plt.ylabel('Frequency (Hz)')
    # plt.colorbar(label='Amplitude (dB)')
    # plt.clim([0,viewingRangeMax-viewingRangeMin])

    yy = inputSeries.copy()

    Tstep = int(np.ceil((len(yy) + autoWindowLength) / windowStep))

    x = np.zeros(len(yy) + 2 * autoWindowLength)

    x[autoWindowLength : len(yy) + autoWindowLength] = yy

    Sautocorr = np.zeros([autoWindowLength, Tstep])

    for count, i in enumerate(np.arange(0, len(yy) + autoWindowLength, windowStep)):

        portion = x[i : i + autoWindowLength]

        # Because numpy.correlate will not give normalized correlations (from -1 to 1), normalize the inputs first, as per
        #     https://stackoverflow.com/questions/5639280/why-numpy-correlate-and-corrcoef-return-different-values-and-how-to-normalize
        firstNormalizedInput = (portion - np.mean(portion)) / (
            np.std(portion) * len(portion)
        )
        secondNormalizedInput = (portion - np.mean(portion)) / np.std(portion)

        Y = np.correlate(
            firstNormalizedInput, secondNormalizedInput, mode="full"
        )  # Calculate autocorrelation for each window of the zero-padded vector

        Sautocorr[:, count] = Y[autoWindowLength - 1 :]  # and only take the last half

    print(f"Full calculated autocorrelogram matrix is shape {Sautocorr.shape}")

    lags = np.arange(0, autoWindowLength) * 1000 / fs

    i1 = int(autoFreqPortion[0] / 100 * Sautocorr.shape[0])
    i2 = int(autoFreqPortion[1] / 100 * Sautocorr.shape[0] - 1)

    if tLim is None:
        tAuto = np.linspace(0, len(inputSeries) / fs, Sautocorr.shape[1])
    else:
        tAuto = np.linspace(tLim[0], tLim[1], Sautocorr.shape[1])

    if meanTimeBounds is None:
        meanTimeBoundsAuto = [0, len(tAuto)]
    else:
        meanTimeBoundsAuto = [
            np.abs(meanTimeBounds[0] - tAuto).argmin(),
            np.abs(meanTimeBounds[1] - tAuto).argmin(),
        ]

    figsize = [15, 12]
    # plt.figure(figsize=figsize)
    # ax1=plt.subplot(211)
    fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        nrows=2, ncols=2, figsize=figsize, width_ratios=[1, 5]
    )
    specMat = y - viewingRangeMin
    plot1 = ax1.plot(
        -specMat[:, meanTimeBoundsSpec[0] : meanTimeBoundsSpec[1]].mean(axis=1),
        fPortion,
    )
    ax1.autoscale(tight=True)
    plot2 = ax2.pcolormesh(tSpec, fPortion, specMat, cmap="gist_yarg")
    # ax1.xlabel('Time (sec)')
    ax2.set_ylabel("Frequency (Hz)")
    cbar1 = plt.colorbar(plot2, label="Amplitude (dB)")
    # cbar1.ax.tick_params(labelsize=20)
    plot2.set_clim([0, viewingRangeMax - viewingRangeMin])

    # plt.figure(figsize=[13,8])
    # ax2=plt.subplot(212, sharex=ax1)
    lagsAxis = lags[i1:i2]
    autoMat = Sautocorr[i1:i2, :]
    plot3 = ax3.plot(
        -np.nanmean(autoMat[:, meanTimeBoundsAuto[0] : meanTimeBoundsAuto[1]], axis=1),
        lagsAxis,
    )
    ax3.autoscale(tight=True)
    plot4 = ax4.pcolormesh(tAuto, lagsAxis, autoMat, cmap="twilight")
    ax4.set_xlabel("Time (sec)")
    ax4.set_ylabel("Lags (ms)")
    cbar2 = plt.colorbar(plot4, label="Correlation")
    # cbar2.ax.tick_params(labelsize=20)
    plot4.set_clim([-1, 1])

    plt.show()


def createPredictorTimeseries(times, fs, lenSignal, values=None):

    if values is None:
        values = np.ones(len(times)) / len(times)
    elif np.isscalar(values):
        values = np.ones(len(times)) * values

    predTimeseries = np.zeros(lenSignal)
    times = np.round(times * fs).astype(int)

    predTimeseries[times] = values

    return predTimeseries
