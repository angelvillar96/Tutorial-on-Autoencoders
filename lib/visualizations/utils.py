
##########################################
# method that ensures that the coded message has the correct energy
def energy_normalization(x):
    energy = K.sum(K.square(x)) / K.cast(K.shape(x)[0], float)
    norm_x = x / K.sqrt(energy)
    return norm_x


##########################################
# method that computes the Noise stddev from the SNR_dB
def compute_noise_power( SNR_dB=10 ):
    SNR = 10 ** (SNR_dB / 10)
    noise_power = 1 / (2 * SNR)
    stddev_noise = np.sqrt(noise_power)
    return stddev_noise
