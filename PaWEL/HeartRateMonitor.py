import numpy as np
import time
from scipy.signal import find_peaks, savgol_filter

class HeartRateMonitor:
    def __init__(self):
        """
        Initialize the HeartRateMonitor object
        """
        self.forehead_buffer = []  # Buffer to store forehead signal values and timestamps
        self.window_duration = 15  # Duration of the time window for heart rate computation (in seconds)
        self.fps = 60  # Frame rate in frames per second (FPS) for better frequency resolution
        self.buffer_size = int(self.window_duration * self.fps)  # Size of the forehead buffer in number of frames
        self.min_bpm = 40  # Minimum allowed heart rate in beats per minute (BPM)
        self.max_bpm = 180  # Maximum allowed heart rate in BPM
        self.peak_threshold = 0.6  # Threshold for peak detection in the frequency domain
        self.stable_heart_rate = None  # Averaged and stabilized heart rate value
        self.stable_heart_rate_count = 0  # Count of frames used for averaging the stable heart rate

    def compute_heart_rate(self, frame):
        """
        Compute the heart rate from the given frame
        """
        frame_time = time.time()
        green_average = np.average(frame, axis=(0, 1))[1]  # Compute the average green channel value

        if len(self.forehead_buffer) < self.buffer_size:
            self.forehead_buffer.append((frame_time, green_average))
        else:
            self.forehead_buffer.pop(0)

        if len(self.forehead_buffer) >= 2:
            fft = self.get_fft([val[1] for val in self.forehead_buffer])  # Compute FFT of the forehead signal

            # Find peaks in the FFT using the find_peaks function
            peaks, _ = find_peaks([val[1] for val in fft], height=self.peak_threshold * np.max([val[1] for val in fft]))

            # Get the peak with the highest magnitude within the specified BPM range
            best_peak = self.find_best_peak(fft, peaks)

            if best_peak is not None:  # Check if a valid peak was found
                best_bin = best_peak[1]
                heartrate = self.bin_to_bpm(best_bin)

                # Apply simple averaging to stabilize heart rate
                if self.stable_heart_rate is None:
                    self.stable_heart_rate = heartrate
                    self.stable_heart_rate_count = 1
                else:
                    self.stable_heart_rate = (self.stable_heart_rate * self.stable_heart_rate_count + heartrate) / (
                            self.stable_heart_rate_count + 1)
                    self.stable_heart_rate_count += 1

                # Set a minimum threshold for stable heart rate values
                if self.stable_heart_rate_count >= 5 and self.min_bpm <= self.stable_heart_rate <= self.max_bpm:
                    return int(self.stable_heart_rate)
            else:
                # Reset stable heart rate if no valid peak found
                self.stable_heart_rate = None
                self.stable_heart_rate_count = 0

        return None  # No valid stable heart rate found

    def smooth_signal(self, signal):
        """
        Smooth the signal using Savitzky-Golay filter (only use in very noisy environments - untested)
        """
        if len(signal) >= 5:  # Check if there are enough data points for smoothing
            window_length = min(len(signal), 5)  # Limit the window_length to the size of the signal
            smoothed_signal = savgol_filter(signal, window_length=window_length, polyorder=3)
        elif len(signal) > 1:
            # If there are 2 to 4 data points, simply replicate the last value to maintain signal continuity
            smoothed_signal = np.repeat(signal[-1], len(signal))
        else:
            # If there's only one data point, return it as the smoothed signal
            smoothed_signal = signal

        return smoothed_signal

    def get_fft(self, values):
        """
        Compute the FFT of the interpolated forehead signal
        """
        even_times = np.linspace(self.forehead_buffer[0][0], self.forehead_buffer[-1][0], len(values))
        interpolated = np.interp(even_times, *zip(*self.forehead_buffer))
        fft = np.fft.rfft(interpolated)
        return list(zip(np.abs(fft), np.angle(fft)))

    def bin_to_bpm(self, bin):
        """
        Convert a frequency bin to heart rate (BPM)
        """
        return (60.0 * bin * self.fps) / float(len(self.forehead_buffer))

    def find_best_peak(self, fft, peaks):
        """
        Find the peak with the highest magnitude within the specified BPM range
        """
        best_peak = None
        for peak_idx in peaks:
            bin_value = peak_idx + 1
            heartrate = self.bin_to_bpm(bin_value)
            if self.min_bpm <= heartrate <= self.max_bpm:
                if best_peak is None or fft[peak_idx][0] > best_peak[0]:
                    best_peak = (fft[peak_idx][0], peak_idx)
        return best_peak
