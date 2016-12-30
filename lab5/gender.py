from __future__ import division

from os import listdir
from os.path import isfile, join
from numbers import Number
from pylab import *
from scipy import *
from scipy.io import wavfile


def extract_single_channel(wav):
    while not isinstance(wav[0], Number):
        wav = [s[0] for s in wav]
    return wav


def multiply_arrays(left, right):
    for (idx, value) in enumerate(right):
        left[idx] *= value
    return left


def hps(samples, signal, result_array):
    for i in range(1, samples):
        analyzed = signal[::i]
        multiply_arrays(result_array, analyzed)


def extract_highest_value(freqs, result, endpoint, left_limit=50, right_limit=350, iteration_limit=100):
    highest_value = freqs[array(result[:endpoint]).argmax()]
    iterations = 0
    while (highest_value < left_limit or highest_value > right_limit) and iterations < iteration_limit:
        endpoint = int(len(freqs) / 2)
        freqs = delete(freqs, [array(result[:endpoint]).argmax()])
        result = delete(result, [array(result[:endpoint]).argmax()])
        highest_value = freqs[array(result[:endpoint]).argmax()]
        iterations += 1
    return highest_value


def get_gender(filename):
    w, wave = wavfile.read(filename)
    wave = extract_single_channel(wave)
    signal = wave[:51000]
    signal = signal * hanning(len(signal))
    signal1 = fft(signal)
    signal1 = abs(signal1) / w
    res = copy(signal1)
    hps(5, signal1, res)
    freqs = fftfreq(len(signal), 1/w)
    endpoint = int(len(freqs)/2)
    highest = extract_highest_value(freqs, res, endpoint)
    # print('Best freq: %f' % highest)

    gender = 'K' if highest > 200 else 'M'
    print(gender)
    return gender


# onlyfiles = [f for f in listdir('train') if isfile(join('train', f))]
# all = 0
# failed = 0
# hit = 0
# for file in onlyfiles:
#     values = get_gender("train/" + file)
#     all += 1
#     if values == file[4]:
#         hit += 1
#     else:
#         failed += 1
# print("All: %d, Failed: %d, hit: %d, accuracy: %f" % (all, failed, hit, hit/all))

get_gender(sys.argv[1])

