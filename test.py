#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Sources
# ____________
# http://stackoverflow.com/questions/19122157/fft-bandpass-filter-in-python
# http://stackoverflow.com/questions/2063284/what-is-the-easiest-way-to-read-wav-files-using-python-summary
# http://www.programcreek.com/python/example/58006/scipy.signal.resample
# http://www.astroml.org/book_figures/chapter10/fig_wiener_filter.html
# http://www.astroml.org/modules/generated/astroML.filters.wiener_filter.html#astroML.filters.wiener_filter
# http://stackoverflow.com/questions/2459295/invertible-stft-and-istft-in-python


import numpy as np
import wave
import scipy, scipy.signal
import pylab as plt

def stft(x, framesamp):
    hopsamp = framesamp/2
    w = scipy.hanning(framesamp)
    X = np.array([scipy.fft(w*x[i:i+framesamp])
                     for i in range(0, len(x)-framesamp, hopsamp)])
    return X

def istft(X, size):
    x = np.zeros(size)
    framesamp = X.shape[1]
    hopsamp = framesamp/2
    for n,i in enumerate(range(0, len(x)-framesamp, hopsamp)):
        x[i:i+framesamp] += scipy.real(scipy.ifft(X[n]))
    return x

class SoundFile:
    def  __init__(self):
        self.initialized = False
        pass

    def  load_signal(self, signal):
        self.initialized = True

        self.sampwidth = 2
        self.samplerate = 44100
        self.nsamples = len(signal)
        self.signal = signal

        return self

    def load_wave(self, fname):
        self.initialized = True

        wav = wave.open(fname, "r")
        (nchannels, self.sampwidth, self.samplerate, self.nsamples, comptype, compname) = wav.getparams()
        size = self.nsamples * nchannels
        frames = wav.readframes(size)
        out = wave.struct.unpack_from("%dh" % size, frames)

        # Convert 2 channles to numpy arrays
        if nchannels == 2:
            self.signal = np.array(list(out[0::2]))
        else:
            self.signal = np.array(out)
        return self

    def get_frame(self, winsize, no):
        shift = winsize/2
        start = no*shift
        end = start+winsize
        return self.signal[start:end]

    def add_signal(self, frame, winsize, no ):
        shift = winsize/2
        start = no*shift
        end = start+winsize
        self.signal[start:end] = self.signal[start:end] + frame

    def write(self, file_name):
        if not self.initialized:
            raise SoundFileNotInitialized()

        file = wave.open(file_name, 'wb')
        file.setparams((1, self.sampwidth, self.samplerate, self.nsamples, 'NONE', 'noncompressed'))

        ssignal = ''
        for i in range(self.nsamples):
            ssignal += wave.struct.pack('h', self.signal[i]) # transform to binary

        file.writeframes(ssignal)
        file.close()

    def awgn(self, snr_dB):
        """ Add White Gaussian Noise (AWGN).

        Parameters
        _________
        snr_dB : float
            Output SNR required in dB.
        """

        if not self.initialized:
            raise SoundFileNotInitialized()

        avg_energy = sum(self.signal * self.signal)/len(self.signal)
        snr_linear = 10**(snr_dB/10.0)
        noise_variance = avg_energy/(2.0*snr_linear)
        noise = np.sqrt(2*noise_variance) * np.random.randn(len(self.signal))

        self.signal = self.signal + noise

        return self

    def add_real_background_noise(self, type, snr_dB):
        """ Add background noise.

        Parameters
        _________
        type : string, one of "airport", "babble", "car", "exhibition", "restaurant", "street", "subway", "train"
        snr_dB : float
            Output SNR required in dB.
        """

        if not self.initialized:
            raise SoundFileNotInitialized()

        snr_linear = 10**(snr_dB/10.0)
        noise = SoundFile().load_wave("noise/babble.wav")
        ratio = self.avg_energy() / snr_linear / noise.avg_energy()
        new_noise = noise.signal * ratio
        self.signal = self.signal + np.repeat(new_noise, int(self.nsamples/new_noise.size) + 1)[self.nsamples]

        return self

    def __filter(self, im, mysize=None, noise=None):
        """
        Perform a noise filter on 1-dimensional array `im`.
        Parameters
        ----------
        im : ndarray
            An 1-dimensional array.
        mysize : int, optional
            A scalar giving the size of the Wiener filter window. It should be
            odd.
        noise : noise
        """
        im = np.asarray(im)
        if mysize is None:
            mysize = 3

        substraction_coeff = 5.0
        w = scipy.hanning(mysize)

        n_spec = scipy.fft(noise.signal[:mysize]*w)
        n_pow = scipy.absolute(n_spec)**2.0

        hopsamp = mysize/2
        frames = np.array([scipy.fft(w*self.signal[i:i+mysize])
                         for i in range(0, self.nsamples - mysize, hopsamp)])

        # plt.figure()
        # plt.plot([sum(f) for f in frames])
        # plt.show()
        # new = np.zeros_like(frames)
        # for i, f in enumerate(frames):
            # # Estimate the local mean
            # lMean = scipy.signal.correlate(f, w, 'same') / mysize
            #
            # # Estimate the local variance
            # lVar = (scipy.signal.correlate(f ** 2, w ** 2, 'same') / mysize
            #         - lMean ** 2)
            #
            # res = lMean + (f - lMean)*(1 - noise / lVar)
            # new[i] = np.where(lVar < noise, lMean, res)

        n = sum(n_pow)/len(n_pow)
        y = scipy.absolute(frames)**2
        new = scipy.sqrt(scipy.maximum(y - substraction_coeff*n, 0.0))
        new = new/(1 - float(len(y.flatten())*n)/sum(y.flatten()))
        new = new*scipy.exp(scipy.angle(frames)*1j)

        x = np.zeros(self.nsamples)
        framesamp = new.shape[1]
        hopsamp = framesamp/2
        for n,i in enumerate(range(0, self.nsamples - framesamp, hopsamp)):
            x[i:i+framesamp] += scipy.real(scipy.ifft(new[n]))
        return x

    def filter(self):
        if not self.initialized:
            raise SoundFileNotInitialized()

        framesz = 0.050  # with a frame size of 50 milliseconds
        framesamp = int(framesz*self.samplerate)
        noise = SoundFile().load_signal(self.signal[:self.nsamples])
        self.signal = self.__filter(self.signal, framesamp, noise)
        return self

    def avg_energy(self):
        if not self.initialized:
            raise SoundFileNotInitialized()

        return float(sum(self.signal ** 2)/len(self.signal))

def test_save_sample():
    # let's prepare signal
    duration = 4 # seconds
    samplerate = 44100 # Hz
    samples = duration*samplerate

    frequency = 440 # Hz
    period = samplerate / float(frequency) # in sample points
    omega = np.pi * 2 / period

    xaxis = np.arange(int(period), dtype = np.float) * omega
    ydata = 16384 * np.sin(xaxis)

    signal = np.resize(ydata, (samples,))

    f = SoundFile().load_signal(signal)
    f.write('sine.wav')
    print 'file written'
    # Manual check if file is correctly saved

def test_read_and_write():
    SoundFile().load_wave("hello.wav").write("new_hello.wav")
    # Manual check if file is correctly saved

def test_add_noise():
    SoundFile().load_wave("hello.wav").awgn(15).write("noisy_hello.wav")
    # Manual check if file is correctly saved

def test_filtering():
    SoundFile().load_wave("noisy_hello.wav").filter().write("filtered_hello.wav")
    # Manual check if file is correctly saved

def test_filtering_babble():
    babble = SoundFile().load_wave("hello.wav").add_real_background_noise("babble", 15)
    babble.write("babble_hello.wav")
    babble.filter().write("filtered_babble_hello.wav")
    # Manual check if file is correctly saved

def test_stft_istft():
    f0 = 440         # Compute the STFT of a 440 Hz sinusoid
    fs = 8000        # sampled at 8 kHz
    T = 5            # lasting 5 seconds
    framesz = 0.050  # with a frame size of 50 milliseconds

    # Create test signal and STFT.
    t = scipy.linspace(0, T, T*fs, endpoint=False)
    x = scipy.sin(2*scipy.pi*f0*t)
    X = stft(x, int(fs*framesz))

    # Plot the magnitude spectrogram.
    plt.figure()
    plt.imshow(scipy.absolute(X.T), origin='lower', aspect='auto',
                 interpolation='nearest')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.show()

    # Compute the ISTFT.
    xhat = istft(X, fs*T)

    # Plot the input and output signals over 0.1 seconds.
    T1 = int(0.1*fs)

    plt.figure()
    plt.plot(t[:T1], x[:T1], t[:T1], xhat[:T1])
    plt.xlabel('Time (seconds)')
    plt.show()

    plt.figure()
    plt.plot(t[-T1:], x[-T1:], t[-T1:], xhat[-T1:])
    plt.xlabel('Time (seconds)')
    plt.show()

if __name__=="__main__":
    test_filtering_babble()
