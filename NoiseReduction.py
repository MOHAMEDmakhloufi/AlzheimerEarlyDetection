import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
from scipy.signal import wiener


# calcule the snr metric to know the quality of the audio
def calculate_snr(signal):
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean((signal - np.mean(signal)) ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr


# convert mp3 file to wav file
def convert_mp3_to_wav(input_mp3, temp_wav):
    # Convert MP3 to WAV using pydub
    audio = AudioSegment.from_mp3(input_mp3)
    audio.export(temp_wav, format="wav")
    print(f"Converted {input_mp3} to {temp_wav}")


def spectral_subtraction_with_wiener(signal, fs, IS=0.5):
    W = int(0.025 * fs)  # length of the frame  (25 ms)
    SP = int(W * 0.4)  # Step length
    # Segment the audio signal into overlapping frames
    y_segments = segment(signal, W, SP)
    # Apply FFT to each segment
    Y = np.fft.rfft(y_segments, axis=1)
    Y_mag = np.abs(Y)
    Y_phase = np.angle(Y)
    # Calculate initial noise profile
    initial_noise_frames = int(IS * fs / W)
    noise_spectrum = np.mean(Y_mag[:initial_noise_frames, :], axis=0)

    # Adaptive spectral subtraction
    alpha = 1.0
    X_mag = np.maximum(Y_mag - alpha * noise_spectrum, 0)

    # Wiener filtering to smooth out residual noise
    X_mag_smoothed = wiener(X_mag)

    # Reconstruct signal with original phase
    X = X_mag_smoothed * np.exp(1j * Y_phase)
    reconstructed_segments = np.fft.irfft(X, axis=1)

    # Reconstruct the time-domain signal from overlapping segments
    reconstructed_signal = reconstruct(reconstructed_segments, W, SP)

    return reconstructed_signal


# segment the audio signal into small frames
def segment(signal, W, SP):
    L = len(signal)
    segments = []
    for start in range(0, L - W + 1, SP):
        segments.append(signal[start:start + W])
    return np.array(segments)


# reconstruct the audio signal from the segments
def reconstruct(segments, W, SP):
    L = (segments.shape[0] - 1) * SP + W
    reconstructed_signal = np.zeros(L)
    window = np.hanning(W)

    for i, segment in enumerate(segments):
        start = i * SP
        reconstructed_signal[start:start + W] += segment * window

    return reconstructed_signal


# calculate the psnr metric to know the quality of the audio
def calculate_psnr(original, processed):
    # Trim or pad the processed signal to match the original signal's length
    min_len = min(len(original), len(processed))
    original = original[:min_len]
    processed = processed[:min_len]

    mse = np.mean((original - processed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def noise_reduction(input_mp3, output_wav='output.wav'):
    temp_wav = 'temp.wav'  # Temporary file for WAV format
    # 1-  Convert MP3 to WAV
    convert_mp3_to_wav(input_mp3, temp_wav)
    # 2- Load the temporary WAV file
    signal, fs = librosa.load(temp_wav, sr=None)
    # 3- Print PSNR before processing
    original_psnr = calculate_psnr(signal, signal)
    print(f"PSNR before processing (no noise reduction applied): {original_psnr:.2f} dB")
    # 4- Apply adaptive spectral subtraction with Wiener filtering
    processed_audio = spectral_subtraction_with_wiener(signal, fs)
    # 5- Normalize and format the processed audio
    processed_audio = np.clip(processed_audio, -1.0, 1.0).astype(np.float32)
    # 6- Calculate and print PSNR after processing
    processed_psnr = calculate_psnr(signal, processed_audio)
    print(f"PSNR after processing (with noise reduction): {processed_psnr:.2f} dB")
    # 7- Save the processed audio as the final WAV output
    sf.write(output_wav, processed_audio, fs)
    print(f"Processed audio saved as {output_wav}")