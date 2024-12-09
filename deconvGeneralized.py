import pyfftw

# from pyfftw.interfaces.scipy_fftpack import fft as fftw, ifft as ifftw
fftw = pyfftw.builders.fft
ifftw = pyfftw.builders.ifft
import numpy as np
import scipy as sp
import time

pyfftw.interfaces.cache.enable()
pyfftw.interfaces.cache.set_keepalive_time(600)
planner_effort = "FFTW_ESTIMATE"

def deconvMain(
    regressor, response, eps, windowStart=0, windowEnd=None, avgRegSpec=None
):
    # fft_x = scipy.fft.fft(regressor, n=len(regressor)*2)
    # fft_y = scipy.fft.fft(response, n=len(regressor)*2)
    # print(regressor.shape)

    print(f"Regressor sum is {np.sum(regressor)}")
    # regressor=regressor/(np.sum(regressor)/regressorCoeff)
    # regressor=regressor/np.sum(regressor)
    # regressor*=response.std(axis=0).mean()/regressor.std()

    regressor = sp.stats.zscore(regressor)
    response = sp.stats.zscore(response)
    print(f"Regressor and response have now both been z-scored")

    print(f"Regressor sum is now {np.sum(regressor)}")
    print(f"Response max is now {np.max(response)}")
    print(f"Response min is now {np.min(response)}")
    print(f"Regressor max is now {np.max(regressor)}")
    print(f"Regressor min is now {np.min(regressor)}")
    print("\n")
    t0 = time.time()
    print("Starting FFT deconvolution")
    # print(windowStart)
    # print(windowEnd)

    regressorPadded = pyfftw.empty_aligned(
        int(regressor.shape[0] - windowStart), dtype="complex128"
    )
    responsePadded = pyfftw.empty_aligned(
        (int(response.shape[0] - windowStart), int(response.shape[1])),
        dtype="complex128",
    )
    fraction = pyfftw.empty_aligned(
        (int(2 * (response.shape[0] - windowStart)), int(response.shape[1])),
        dtype="complex128",
    )

    regressorPadded[:] = np.pad(
        regressor, (0, int(-windowStart)), mode="constant", constant_values=0
    )
    responsePadded[:] = np.pad(
        response, ((int(-windowStart), 0), (0, 0)), mode="constant", constant_values=0
    )
    t1 = time.time()
    print(f"Finished padding regressor and response; time taken was {t1-t0} seconds")
    # print(regressorPadded.shape)
    # print(responsePadded.shape)

    NFFT = len(regressorPadded) * 2
    # print(NFFT)
    calc_fft_x = fftw(
        regressorPadded, n=NFFT, planner_effort=planner_effort, auto_contiguous=True
    )
    fft_x = calc_fft_x()
    t2 = time.time()
    print(f"Finished regressor FFT; time taken was {t2-t1} seconds")
    # print(fft_x.shape)
    # print(response.shape)
    calc_fft_y = fftw(
        responsePadded,
        n=NFFT,
        axis=0,
        planner_effort=planner_effort,
        auto_contiguous=True,
    )
    fft_y = calc_fft_y()
    t3 = time.time()
    print(f"Finished response FFT; time taken was {t3-t2} seconds")
    # print(fft_y.shape)

    # scale recommended by verhulst for abr amplitude scaling
    # dOct = 1. / 6  # Cfs in sixth octaves 1./6
    # cf = 2 ** np.arange(np.log2(125), np.log2(16e3 + 1), dOct)
    # scale = 401 / len(cf)

    # impulse response from frequency domain division
    # add epsilon to denominator to avoid division by 0
    # w = ifftw((np.conj(fft_x) * fft_y) / ((np.conj(fft_x) * fft_x)+eps),
    #        n=len(regressor)*2).real * scale

    xConj = np.conj(fft_x)
    t4 = time.time()
    print(f"xConj is {xConj.shape}")
    print(f"Finished conjugating regressor; time taken was {t4-t3} seconds")
    numerator = fft_y * xConj[:, None]
    t5 = time.time()
    print(f"numerator is {numerator.shape}")
    print(
        f"Finished calculating regressor and response cross correlation; time taken was {t5-t4} seconds"
    )
    if avgRegSpec is not None:
        denominator = avgRegSpec + eps
    else:
        denominator = (xConj * fft_x) + eps
    t6 = time.time()
    print(f"denominator is {denominator.shape}")
    print(
        f"Finished calculating regularized regressor autocovariance; time taken was {t6-t5} seconds"
    )
    fraction[:] = numerator / denominator[:, None]
    t7 = time.time()
    print(f"fraction is {fraction.shape}")
    print(
        f"Finished calculating least squares solution via frequency domain division; time taken was {t7-t6} seconds"
    )

    calc_w = ifftw(
        fraction, n=NFFT, axis=0, planner_effort=planner_effort, auto_contiguous=True
    )
    # calc_w=ifftw(numerator, n=NFFT, axis=0, planner_effort=planner_effort, auto_contiguous=True)

    w = calc_w()
    t8 = time.time()
    print(f"w is {w.shape}")
    print(
        f"Finished converting back to time domain via iFFT; time taken was {t8-t7} seconds"
    )
    w = w.real
    # w=w * scale
    print(f"Finished FFT deconvolution; total time taken was {time.time()-t0} seconds")
    print("\n")
    if windowEnd is not None:
        windowLength = int(windowEnd - windowStart)

        # windowStart=-100
        # windowEnd=500

        # paddedResponse=np.pad(response, ((int(-windowStart), int(windowEnd)), (0, 0)), mode='constant', constant_values=0)
        # print(paddedResponse.shape)

        # numElec=int(response.shape[1])

        # print(numElec)

        # window=np.zeros([windowLength, numElec])

        # print(window.shape)
        # t0=time.time()
        # print('Starting overlap and add method')
        # for i, coeff in enumerate(regressor):
        #    newWindow=paddedResponse[i:i+windowLength,:]
        # print(newWindow.shape)
        # print(coeff)
        # print(window.shape)
        #    newWindow=newWindow*coeff+window
        #    window=newWindow
        #    if i%20000==0:
        #        print(f'Done with {i} regressor samples')
        # print(f'Finished overlap and add method; time taken was {time.time()-t0} seconds')
        # print('\n')

        t0 = time.time()
        print("Starting regularized regression TRF estimation with numpy least squares")

        lam = eps  # eps is input to this function, but for ridge regression, lambda is the same, so copy it here

        I = np.identity(windowLength + 1)
        # I[0,0]=0  # mTRF code on GitHub has this, not sure why, it only seems to cause a problem at the first sample

        # Set up regressor lag matrix
        S = np.zeros([regressor.shape[0], windowLength + 1])
        for i, samp in enumerate(
            np.linspace(int(windowStart), int(windowEnd), windowLength + 1)
        ):
            samp = int(samp)
            if samp < 0:
                temp = regressor[-samp:]
                S[:samp, i] = temp
            elif samp == 0:
                temp = regressor
                S[:, i] = temp
            elif samp > 0:
                temp = regressor[:-samp]
                S[samp:, i] = temp

        # print('Starting regularized TRF estimation with numpy least squares')
        t1 = time.time()
        SScov = S.T @ S
        print(
            f"Regressor autocovariance matrix is shape {SScov.shape} and took {time.time()-t1} seconds to compute, max is {SScov.max()}"
        )
        t1 = time.time()
        SScov = SScov + lam * I
        print(
            f"Regularized autocovariance matrix is shape {SScov.shape} and took {time.time()-t1} seconds to compute"
        )
        t1 = time.time()
        SRcov = S.T @ response
        print(
            f"Regressor - response covariance matrix is shape {SRcov.shape} and took {time.time()-t1} seconds to compute"
        )
        solution = np.linalg.lstsq(SScov, SRcov)
        window = SRcov
        TRF = solution[0]
        print(f"TRF matrix is shape {TRF.shape}")
        print(
            f"Finished least squares TRF estimation; total time taken was {time.time()-t0} seconds"
        )
        print("\n")

    # w = ifftw((fft_y * np.conj(fft_x)[:,None]) / ((np.conj(fft_x) * fft_x)+eps)[:,None],
    #          n=len(regressor)*2,axis=0).real * scale
    # print(w.shape)
    if windowEnd is not None:
        return w, TRF
    else:
        return w
