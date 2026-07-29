import numpy as np


def _blackmanharris(num_samples: int) -> np.ndarray:
  """4-term Blackman-Harris (scipy.signal.windows.blackmanharris compatible)."""
  n = np.arange(num_samples, dtype=np.float64)
  if num_samples == 1:
    return np.ones(1, dtype=np.float64)
  a0, a1, a2, a3 = 0.35875, 0.48829, 0.14128, 0.01168
  fac = n * (2.0 * np.pi / (num_samples - 1))
  return a0 - a1 * np.cos(fac) + a2 * np.cos(2.0 * fac) - a3 * np.cos(3.0 * fac)


class DopplerAlgo:
  """Compute range-doppler map for one FMCW frame."""

  def __init__(self, config: dict, num_ant: int, mti_alpha: float = 0.8):
    self.num_chirps_per_frame = config["num_chirps_per_frame"]
    num_samples_per_chirp = config["num_samples_per_chirp"]
    self.range_window = _blackmanharris(num_samples_per_chirp).reshape(1, num_samples_per_chirp)
    self.doppler_window = _blackmanharris(self.num_chirps_per_frame).reshape(1, self.num_chirps_per_frame)
    self.mti_alpha = mti_alpha
    self.mti_history = np.zeros((self.num_chirps_per_frame, num_samples_per_chirp, num_ant))

  def compute_doppler_map(self, data: np.ndarray, i_ant: int):
    data = data - np.average(data)
    data_mti = data - self.mti_history[:, :, i_ant]
    self.mti_history[:, :, i_ant] = data * self.mti_alpha + self.mti_history[:, :, i_ant] * (1 - self.mti_alpha)

    fft1d = self.fft_spectrum(data_mti, self.range_window)
    fft1d = np.transpose(fft1d)
    fft1d = np.multiply(fft1d, self.doppler_window)
    zp2 = np.pad(fft1d, ((0, 0), (0, self.num_chirps_per_frame)), "constant")
    fft2d = np.fft.fft(zp2) / self.num_chirps_per_frame
    return np.fft.fftshift(fft2d, (1,))

  def fft_spectrum(self, mat, range_window):
    numchirps, chirpsamples = np.shape(mat)
    avgs = np.average(mat, 1).reshape(numchirps, 1)
    mat = mat - avgs
    mat = np.multiply(mat, range_window)
    zp1 = np.pad(mat, ((0, 0), (0, chirpsamples)), "constant")
    range_fft = np.fft.fft(zp1) / chirpsamples
    range_fft = 2 * range_fft[:, range(int(chirpsamples))]
    return range_fft
