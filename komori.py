import scipy, pylab
from scipy.io import wavfile
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os
import webbrowser

"""Komori visualization subpackage"""

class recording:
    """ Recording class """
    def __init__(self, path):
        self.path = path
        self.samplerate, self.data = wavfile.read(self.path)
        self.duration = float(self.data.size)/self.samplerate

    def resample(self, et=10):
        """
        Resampling on numpy arrays
        Domagoj

        data: numpy array of data recorded with sr samplerate
        sr: recording samplerate
        et: given time expansion factor

        saves resampled.wav into current directory
        """
        rep_sr = 44100
        sr = self.samplerate/et
        secs = self.data.size/sr
        samps = secs * rep_sr
        resampled = signal.resample(self.data, samps)
        # print resampled, 'resampled'
        # TODO: path must be better specified
        wavfile.write('tmp/'+'resampled.wav', 44100, resampled)

    def play(self, et=10):
        self.resample(et)
        webbrowser.open(os.path.abspath("tmp/resampled.wav"))


class spectrogram:
    """Spectrogram class"""
    def __init__(self, recording, n_fft=512, hop=64):
        """
        Spectrogram init variables
        """
        self.data, self.frequencyUnit, self.timeUnit = self.STFT(recording, n_fft=n_fft, hop=hop)

    def STFT(self, recording, n_fft=512, hop=64):
        """
        Short time fourier transformation


        """
        data = np.float64(recording.data)
        w = scipy.hanning(n_fft)
        X = scipy.array([scipy.fft(w*data[i:i+n_fft])
                         for i in range(0, len(data)-n_fft, hop)])
        X = X.T
        X = X[:X.shape[0]/2]

        frequencyUnit = float(recording.samplerate)/n_fft
        timeUnit = ((len(recording.data)/float(recording.samplerate))/X.shape[1])*1000
        return np.sqrt(X), frequencyUnit, timeUnit

    def iSTFT(self, sr, yT, hop=64):
        """
        Inverse short time fourier transformation

        stft: numpy array gained with STFT function
        sr: samplerate
        yT: number of samples or size of data on output
        hop: hop size
        """
        X = self.data
        X2 = np.flipud(X)

        X = np.vstack((X, X2))**2

        x = scipy.zeros(yT)
        framesamp = X.shape[1]
        for n,i in enumerate(range(0, len(x)-framesamp, hop)):
            x[i:i+framesamp] += scipy.real(scipy.ifft(X[n]))
        # print x
        return x

    def RMSE(self):
        """
        Calculates root-mean-square energy.

        y: numpy array opened with scipy.io.wavfile.read
        sr: Samplerate
        wfft: fft window size
        hop: hop_length
        power: rmse power (default=1)
        """
        return np.sqrt(np.mean(np.abs(self.data), axis=0))

    def FFT(self):
        """Calculates root-mean-square energy. in other axis

        y: numpy array opened with scipy.io.wavfile.read
        sr: Samplerate
        wfft: fft window size
        hop: hop_length
        power: rmse power (default=1)
        """
        return np.sqrt(np.mean(np.abs(self.data), axis=1))

    def RMSEandFFT(self):
        """
        Returns RMSE and FFT representaion of STFT data
        """
        return self.RMSE(), self.FFT()

    def subsample(self, step=128):
        """
        Extracting samples using 256x256 sliding window
        Window is being slided between sampling by step ammount of data points in time units.
        Default step is 128 which make samples that overlaps by 50 percent


        """
        subsamples = []
        fl = self.data.shape[1]

        # print fl, fl/64, (fl/64)*64

        start = 0
        stop = 256

        for i in range(fl/step):
            if stop <= (fl/step)*step:
                subsamples.append(self.data[:,start:stop])
                start+=step
                stop+=step

        return subsamples

    def clean_noise(self, events_t):
        """ Noise cleanup """
        # print self.timeUnit
        noises = []
        events= []
        for i in events_t:
            if i[1]-i[0]>int(20/self.timeUnit):
                events.append(i)

        for i in events:
            noises.append(np.mean(np.abs(self.data[:, i[0]:i[1]+1])))

        # print np.where(noises == np.min(noises))[0]
        # print events
        events = events[np.where(noises == np.min(noises))[0][0]]


        noise = self.data[: ,events[0]:events[1]+1]
        noise = np.mean(np.abs(noise), axis=1)
        noise = noise[:, None]
        self.data = (self.data / noise**1)**1

    def findNoise(self):
        rmse = self.RMSE()
        rmse_lm = np.where(rmse.T <= np.mean(rmse.T)+np.std(rmse.T))
        rmse_lm = rmse_lm[0].tolist()
        events_t = self.findstartstop(rmse_lm)

        return events_t

    def findEvents(self):
        """
        This part made system ignore quieter bat calls
        for SURF feature extraction and convolutional neural networks
        sliding window system is implemented
        """
        rmse = self.RMSE()
        rmse_lm = np.where(rmse.T >= np.mean(rmse.T)+np.std(rmse.T))
        rmse_lm = rmse_lm[0].tolist()
        events_t = self.findstartstop(rmse_lm)

        events = []
        for i in events_t:
            if i[1]-i[0]>5:
                events.append(i)


        return events

    def findstartstop(self, ind, skip=1):
        # TODO: make this docstring more understandable
        # TODO: rework variable names
        # TODO: find alternative algorithm
        """
        For data in iterable finds pairs of data in the following manner
        if iterable is [1,2,3,4,5] it will return [[0,4]]
        if iterable is [1,2,3,5,6,7,10,12,15] it will return [[0,2],[3,5]]

        ind: list of integer sets of data
        skip: skip treshold
        """
        sets = []
        start = None
        for i in range(len(ind)):
            if start == None:
                start = i
            try:
                if ind[i+1] - ind[i] <= skip:
                    pass
                else:
                    stop = i
                    sets.append([ind[start], ind[stop]])
                    start = i+1
            except IndexError:
                    stop = i
                    sets.append([ind[start], ind[stop]])
        return sets

    def plot(self):
        """
        Plots the spectrogram using matplotlib

        """
        def time_t(x, pos):
            return '%1.1f ms' %((x*self.timeUnit)/1000)

        def freq_f(x, pos):
            return '%1.1f kHz' %((x*self.frequencyUnit))

        # TODO: add units and spectrogram info
        formatter_t = FuncFormatter(time_t)
        formatter_f = FuncFormatter(freq_f)

        ax=plt.subplot()
        plt.imshow(np.abs(self.data))
        # plt.title('Spektrogram, FFT 512, pomak 64, Hanning',{'fontsize': 10}, loc='left')
        ax.xaxis.set_major_formatter(formatter_t)
        ax.yaxis.set_major_formatter(formatter_f)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(10)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(10)



        plt.ylim(0, len(self.data))
        plt.show()
