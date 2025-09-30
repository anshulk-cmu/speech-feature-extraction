#! /usr/bin/env/python

# Calculates acoustic features for ASR Applications

import numpy as np 
from scipy.fftpack import dct
import os 
from scipy.io import wavfile


def hz2mel(hz: float):
    """Converts a value in Hertz to Mels.

    :param hz: a value in Hz. This can also be a np array, conversion proceeds element-wise.
    :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
    """
    return 2595 * np.log10(1+hz/700.)


def mel2hz(mel: float):
    """Converts a value in Mels to Hertz.

    :param mel: a value in Mels. This can also be a np array, conversion proceeds element-wise.
    :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
    """
    return 700*(10**(mel/2595.0)-1)


def preemphasis(signal: np.ndarray, k=0.95):
    """Performs preemphasis on the input signal.
    
    The pre-emphasis filter is represented by the difference equation
    p[n] = x[n] - k * x[n-1]

    :signal: The signal to filter.
    :param k: The preemphasis coefficient. 0 is no filter, default is 0.95.
    :returns: The filtered signal.
    """
    return np.append(signal[0], signal[1:] - k * signal[:-1])


def get_log_feats(fbank_feats: np.ndarray):
    """Computes the logarithm of the fbank feats.
    
    :param fbank_feats: Computed Filterbank features. 
    Make sure that these features do not have zero anywhere as you cannot then take the log.
    :returns log_fbank_feats
    """
    return np.log(fbank_feats)


def get_mel_bank(num_filters=80, lowfreq=0, highfreq=8000, nfft=512, sampling_rate=16000):
    """Computes the mel scaling triangular filterbank function.
    
    :param num_filters: Number of mel coefficients
    :param lowfreq: Low Band Edge where Mel Filterbanks start (0 Hz)
    :param highfreq: Highest Band Edge that the Mel Filterbanks go to (usually Sampling Rate // 2)
    :returns fbank: Mel filberbanks of shape (num_filters, NFFT//2+1)
    """

    ## Get center points evenly spaced on the Mel Frequency scale
    lower_mel_frequency = hz2mel(lowfreq)
    high_mel_frequency = hz2mel(highfreq)
    mel_centers = np.linspace(lower_mel_frequency, high_mel_frequency, num_filters + 2)
    
    ## Now convert the centers back to Frequency scale to get the filters using normal frequency scale
    freq_bins = np.floor(((nfft+1)*mel2hz(mel_centers)/sampling_rate))
    fbank = np.zeros([num_filters, nfft//2 + 1])
    for j in range(0, num_filters):
        for i in range(int(freq_bins[j]), int(freq_bins[j+1])):
            fbank[j,i] = (i - freq_bins[j]) / (freq_bins[j+1]-freq_bins[j])
        for i in range(int(freq_bins[j+1]), int(freq_bins[j+2])):
            fbank[j,i] = (freq_bins[j+2]-i) / (freq_bins[j+2]-freq_bins[j+1])
    return fbank


def lifter(cepstra: np.ndarray, L=22):
    """Applys a cepstral lifter the the matrix of cepstra. This has the effect of increasing the
    magnitude of the high frequency DCT coeffs.

    :param cepstra: the matrix of mel-cepstra, will be numframes * numcep in size.
    :param L: the liftering coefficient to use. Default is 22. L <= 0 disables lifter.
    """
    if L > 0:
        _,ncoeff = np.shape(cepstra)
        n = np.arange(ncoeff)
        lift = 1 + (L/2.)*np.sin(np.pi*n/L)
        return lift * cepstra
    else:
        # values of L <= 0, do nothing
        return cepstra


## Below are two wrapper functions to compute LMF and MFCC

def compute_lmf_feats(
    raw_signal: np.ndarray,
    window_length,
    overlap_length,
    sampling_rate,
    preemph=True,
    mel_low_freq=0,
    mel_high_freq=8000,
    num_mel_filters=80
):
    """Computes log mel filterbank (LMF) features for a given signal.

    :param raw_signal: Input audio signal
    :param window_length: Window length in s
    :param overlap_length: Overlap length in s
    :param sampling_rate: Sampling Rate in Hz
    :param preemph: Do Pre-emphasis on audio signal
    :param mel_low_freq: Low Band Edge where Mel Filterbanks start (0 Hz)
    :param mel_high_freq: Highest Band Edge that the Mel Filterbanks go to (usually Sampling Rate // 2)
    :param num_mel_filters: Number of filters in the mel filterbank, which is usually 23/40/80
    :returns log_mel_fbanks: the log mel filterbank (LMF) features     
    """

    ## calculate the window length and overlap length in samples
    window_sample_length = int(window_length * sampling_rate)
    overlap_sample_length = int(overlap_length * sampling_rate)
    
    ## Step 0: Perform Pre-emphasis in the Time domain using the provided pre-emphasis function
    pre_emph_signal = preemphasis(raw_signal)
    
    ## Step 1: Get the windowed signal (frames)
    framed_signal = frame_with_overlap(
        pre_emph_signal, window_sample_length, overlap_sample_length
    )
    
    ## Step 2: Compute the powerspectrum of the Framed Signal
    power_spectrum, nfft = compute_powerspec(framed_signal, window_sample_length)  
    
    ## Step 3: obtain the Mel Triangular Weighting Filterbanks with the given specifications
    melfilterbanks = get_mel_bank(
        num_mel_filters, mel_low_freq, mel_high_freq, nfft, sampling_rate
    )
    
    ## Step 4: Get Mel Fbank features 
    mel_fbank_feats = get_mel_fbank_feat(power_spectrum, melfilterbanks)
    
    ## Step 5: Get Log Fbank Features
    lmf_feats = get_log_feats(mel_fbank_feats)

    return lmf_feats


def compute_mfcc_feats(
    raw_signal: np.ndarray,
    window_length,
    overlap_length,
    sampling_rate,
    preemph=True,
    mel_low_freq=0,
    mel_high_freq=8000,
    num_mel_filters=80,
    num_ceps=23,
    ceplifter=22
):
    """Computes MFCC features for a given signal.

    :param raw_signal: Input audio signal
    :param window_length: Window length in s
    :param overlap_length: Overlap length in s
    :param sampling_rate: Sampling Rate in Hz
    :param preemph: Do Pre-emphasis on audio signal
    :param mel_low_freq: Low Band Edge where Mel Filterbanks start (0 Hz)
    :param mel_high_freq: Highest Band Edge that the Mel Filterbanks go to (usually Sampling Rate // 2)
    :param num_mel_filters: Number of filters in the mel filterbank
    :param num_ceps: Number of cepstral coefficients considered from the DCT
    :param L: Last "L" coefficients to apply filtering in the cepstral domain on (Liftering)
    :returns mfcc_coeffs: Returns the mfcc_coefficients     
    """
    ## Step 6 : MFCC feats 
    
    ## Get the Log Mel Filterbank features
    lmel_feats = compute_lmf_feats(
        raw_signal, window_length, overlap_length, sampling_rate,
        preemph, mel_low_freq, mel_high_freq, num_mel_filters
    )
    
    ## Compute the DCT of the Log Mel Features
    mfcc_feats = dct(lmel_feats, type=2, axis=1, norm='ortho')[:,:num_ceps]
    
    ## Filtering in the Cepstral domain (called Liftering)
    liftered_mfcc = lifter(mfcc_feats, ceplifter)

    return liftered_mfcc

def visualize_features(audio: np.ndarray, lmf_features: np.ndarray, mfcc_features: np.ndarray, 
                      sampling_rate=16000, window_length=0.025, overlap_length=0.01, 
                      output_filename="feature_visualization.png"):
    """Visualizes power spectrum, LMF features, and MFCC features for comparison.
    
    :param audio: Audio
    :param lmf_features: Log Mel Filterbank features of shape (num_frames, num_mel_filters)
    :param mfcc_features: MFCC features of shape (num_frames, num_ceps)
    :param sampling_rate: Sampling rate in Hz
    :param window_length: Window length in seconds
    :param overlap_length: Overlap length in seconds  
    :param output_filename: Output filename for the saved plot
    """
    import matplotlib.pyplot as plt
    ###########################################################################
    ### Calculate power spectrum #####
    window_sample_length = int(0.025 * sampling_rate)
    overlap_sample_length = int(0.01 * sampling_rate)
    pre_emph_signal = preemphasis(audio)
    framed_signal = frame_with_overlap(pre_emph_signal, window_sample_length, overlap_sample_length)
    power_spectrum, nfft = compute_powerspec(framed_signal, window_sample_length)
    ###########################################################################

    
    # Calculate time axis for frames
    hop_length = window_length - overlap_length
    num_frames = lmf_features.shape[0]
    time_frames = np.arange(num_frames) * hop_length
    
    # Calculate frequency axis for power spectrum
    nfft = (power_spectrum.shape[1] - 1) * 2
    freq_bins = np.linspace(0, sampling_rate//2, power_spectrum.shape[1])
    
    # Create figure with three subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot 1: Power Spectrum
    im1 = axes[0].imshow(10 * np.log10(power_spectrum.T + 1e-10), 
                        aspect='auto', origin='lower', 
                        extent=[time_frames[0], time_frames[-1], freq_bins[0], freq_bins[-1]],
                        cmap='viridis')
    axes[0].set_title('Power Spectrogram', fontsize=14)
    axes[0].set_ylabel('Frequency (Hz)', fontsize=12)
    axes[0].set_xlabel('Time (s)', fontsize=12)
    plt.colorbar(im1, ax=axes[0], label='Power (dB)')
    
    # Plot 2: LMF Features  
    im2 = axes[1].imshow(lmf_features.T, 
                        aspect='auto', origin='lower',
                        extent=[time_frames[0], time_frames[-1], 0, lmf_features.shape[1]-1],
                        cmap='viridis')
    axes[1].set_title('Log Mel Filterbank (LMF) Features', fontsize=14)
    axes[1].set_ylabel('Mel Filter Index', fontsize=12)
    axes[1].set_xlabel('Time (s)', fontsize=12)
    plt.colorbar(im2, ax=axes[1], label='Log Magnitude')
    
    # Plot 3: MFCC Features
    im3 = axes[2].imshow(mfcc_features.T, 
                        aspect='auto', origin='lower',
                        extent=[time_frames[0], time_frames[-1], 0, mfcc_features.shape[1]-1],
                        cmap='viridis')
    axes[2].set_title('Mel-Frequency Cepstral Coefficients (MFCC)', fontsize=14)
    axes[2].set_ylabel('Cepstral Coefficient Index', fontsize=12)
    axes[2].set_xlabel('Time (s)', fontsize=12)
    plt.colorbar(im3, ax=axes[2], label='Coefficient Value')
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Visualization saved as {output_filename}")


def frame_with_overlap(signal: np.ndarray, window_length: int, overlap_length: int):
    """Creates overlapping windows (frames) of the input signal.
    
    :param signal: 1-D audio signal of shape (num_samples,)
    :param window_length: window length (in samples)
    :param overlap_length: overlapping length (in samples)
    :returns: 2-D array of shape (num_frames, window_length) 
    """
    
    hop_length = window_length - overlap_length
    num_samples = len(signal)
    frames = []
    start = 0
    
    # The loop continues as long as the start of a frame is inside the signal
    while start < num_samples:
        end = start + window_length
        frame = signal[start:end]
        
        # If the frame is shorter than the window, it MUST be the last frame.
        # Pad it, add it, and then STOP.
        if len(frame) < window_length:
            padding_needed = window_length - len(frame)
            padding = np.zeros(padding_needed)
            frame = np.concatenate((frame, padding))
            frames.append(frame)
            break
        
        # If the frame is full, just add it and continue
        frames.append(frame)
        start += hop_length
        
    return np.array(frames)


def compute_powerspec(framed_signal: np.ndarray, window_sample_length: int):
    """Computes the power spectrum from the framed signal.
    
    :param framed_signal: framed signal of shape (num_frames, window_sample_length)
    :param window_sample_length: Length of the window in samples 
    :returns power_spectrum: the Power Spectrum of the Short Time Fourier Transform of the signal.
    :returns nfft: the Fourier Transform dimension for the Fast Fourier Transform (always a power of two >= window length)
    If framed signal is a [T,w] matrix, output power_spec will be [T,(NFFT/2+1)]
    """

    # Step 1: Compute the number of FFT points you need: nfft 
    # It must be the smallest power of 2 greater than or equal to the window length. 
    nfft = int(2**np.ceil(np.log2(window_sample_length)))

    # Step 2: Get the STFT of the windowed signal using the real FFT (rfft). 
    # We apply this to each frame (along the last axis).
    stft = np.fft.rfft(framed_signal, n=nfft, axis=-1)
    
    # Step 3: Get the magnitude of the complex STFT. 
    magnitude = np.abs(stft)
    
    # Step 4: Get the Power Spectrum by squaring the magnitude and scaling it. 
    power_spectrum = (magnitude**2) / nfft
    
    return power_spectrum, nfft


def get_mel_fbank_feat(power_spec: np.ndarray, mel_filterbanks: np.ndarray, eps=1e-08):
    """Computes the Mel Filterbank features as the dot product of the Power-spectrum and the Mel Filter banks.

    :param power_spec: power Spectrum of the STFT Magnitude, shape: (T, (NFFT/2+1))
    :param mel_filterbanks: Mel Scale filterbank function, shape: (num_filters, NFFT//2+1)
    :param eps: Small value used where feature value is zero to make sure that the log is valid
    :returns mel_fbank: Mel Filterbank features of signal
    """

    # Take the dot product between the power spectrum of each frame and the filterbank.
    # We use the transpose (.T) of the filterbanks to align the matrix dimensions for multiplication.
    # (num_frames, nfft/2+1) @ (nfft/2+1, num_filters) -> (num_frames, num_filters)
    mel_fbank = np.dot(power_spec, mel_filterbanks.T)
    
    # Replace any features that are exactly zero with a small
    # epsilon value to prevent errors when taking the logarithm in the next step. 
    mel_fbank[mel_fbank == 0] = eps
    
    return mel_fbank


if __name__ == "__main__":
    sampling_rate, audio = wavfile.read(os.path.join("example_data", "example_audio.wav"))
    feat_arrays = np.load(os.path.join("example_data", "example_feats.npz"))
    lmf_feats = feat_arrays["lmel"]
    mfcc_feats = feat_arrays["mfcc"]

    my_lmf_feats = compute_lmf_feats(
        raw_signal=audio,
        window_length=0.025,
        overlap_length=0.01,
        sampling_rate=sampling_rate,
        preemph=True,
        mel_low_freq=0,
        mel_high_freq=8000,
        num_mel_filters=80
    )
    my_mfcc_feats = compute_mfcc_feats(
        raw_signal=audio,
        window_length=0.025,
        overlap_length=0.01,
        sampling_rate=sampling_rate,
        preemph=True,
        mel_low_freq=0,
        mel_high_freq=8000,
        num_mel_filters=80
    )

    assert np.allclose(my_lmf_feats, lmf_feats), "LMF failed on the example audio."
    assert np.allclose(my_mfcc_feats, mfcc_feats), "MFCC failed on the example audio."

    print("---------- Success! ----------")


    # Generate visualization
    visualize_features(audio, my_lmf_feats, my_mfcc_feats, 
                    sampling_rate=sampling_rate, 
                    window_length=0.025, 
                    overlap_length=0.01)

