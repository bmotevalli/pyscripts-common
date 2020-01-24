from scipy.fftpack import fft
import numpy as np


def get_fft(t,x):

    """
    This function returns fast-fourier transform. fftpack in scipy is used for this purpose, however, 
    numpy.fft.fft can also be used in similar manner.

    t:       time
    x:       signal

    fr:      frequency axis
    X_fr:    amplitude complex number
    amp:     amplitude
    angle:   phase angle
    """

    # Sampling Frequency
    Fs = 1 / (t[1] - t[0])
    
    # Generate Frequency Axis
    n = np.size(t)
    n_half = int(n / 2)
    fr = (Fs / 2) * np.linspace(0, 1, n_half)
    
    # Compute FFT
    X     = fft(x)
    X_fr  = (1 / n_half) * X[0:n_half]
    amp   = np.absolute(X_fr)
    angle = np.angle(X_fr)
    
    return fr, X_fr, amp, angle

def get_ifft(X):

    """
    X:            amplitude complex number
    """
    return np.fft.ifft(X).real * np.size(X) / 2



def fft_comps_cutoff(t, x, cutoff = 0.01):
    
    fr, X_fr, amp, angle = get_fft(t, x)
    
    # filter
    amp_norm = amp / amp.max()
    mask = amp_norm > cutoff

    return fr[mask], X_fr[mask], amp[mask], angle[mask]

def construct_time_signal_from_comps(t_vec, freq, amplitude, angle):

    y_vec = np.zeros_like(t_vec)
    for fr, amp, a in zip(freq, amplitude, angle):
        # y_vec += amp * np.sin(2 * np.pi * fr * t_vec + a)
        y_vec += amp * np.sin(2 * np.pi * fr * t_vec) # + a)

    return y_vec
