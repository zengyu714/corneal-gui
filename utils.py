import os
import imageio
import numpy as np

from operator import itemgetter
from itertools import groupby
from scipy.misc import imsave
from scipy.cluster import vq
from scipy.signal import find_peaks_cwt
from collections import Counter


def convert_to_frames(video_name):
    """Convert video to frames."""
    video_name = video_name.rstrip('.avi')
    video_path = 'repo/{}.avi'.format(video_name)
    base_name = video_path.split('/')[-1][:-4]
    frame_dir = 'static/cache/frame/{}'.format(base_name)
    if os.path.exists(frame_dir):
        return sorted([os.path.join(frame_dir, p) for p in os.listdir(frame_dir)])
    else:
        os.makedirs(frame_dir)
        video_reader = imageio.get_reader(video_path, 'ffmpeg')
        frame_names = []
        for i, im in enumerate(video_reader, start=1):
            frame_name = '{}/{}_{:03d}.jpg'.format(frame_dir, base_name, i)
            imsave(frame_name, im, format='jpeg')
            frame_names.append(frame_name)
        return frame_names


def is_already_executed():
    """If all videos have been already executed, the inferred directory should be exist."""
    status = [os.path.exists('static/cache/infer/{}'.format(video_name[:-4]))
              for video_name in os.listdir('repo')]
    return all(status)


def isolate_bound(im):
    """Separate the upper and lower bound in a curve mask.
    Argument:
        im: (array) black-and-white image
    Returns:
        xs: (list) the sequential column index of input image
        y_up: (list) the row index which is relatively smaller
        y_lw: (list) the row index which is relatively larger
    """
    coords = sorted(list(zip(*np.where(im > 0)[::-1])))
    d = {}
    for key, value in coords:
        d[key] = d.get(key, []) + [value]

    if len(coords) & 1:
        # something wrong with the mask, where y is not a unique mapping of x
        # say the two pixels in x-axis is identical
        bad = {key: d[key] for key in d if len(d[key]) > 2}
        for k, v in bad.items():
            d[k] = vq.kmeans(np.array(v, dtype=float), 2)[0].astype(int).tolist()

    xs, ys = [*d.keys()], [v for dv in d.values() for v in dv]
    y_up, y_lw = [ys[i::2] for i in [0, 1]]
    return xs, y_up, y_lw


def compute_curvature(primary_item, roi_deviation=100, im_width=576):
    """
    Argument:
        primary_item: (dict) E.g., {'index': ..., 'thickness': ..., 'xs': ..., 'y_up', 'y_lw'}
        roi_deviation: (int) deviation from the middle of image that truly counted
    """
    middle = im_width // 2

    x, y_up, y_lw = [primary_item.get(key) for key in ['xs', 'y_up', 'y_lw']]
    bulk = [z for z in zip(*[x, y_up, y_lw]) if middle - roi_deviation <= z[0] <= middle + roi_deviation]

    x, y_up, _ = list(zip(*bulk))
    f_up = np.poly1d(np.polyfit(x, y_up, 2))
    # Derivative and compute the curvature in interested range, only care about the front surface
    return abs(np.polyval(np.polyder(f_up), x)).mean()


def get_applanation_time(curvatures, value):
    """Consider the curvatures line is mirror symmetry in horizontal."""

    ar = np.array(curvatures)
    mid = len(ar) // 2
    lhs_index = (np.abs(ar[:mid] - value)).argmin() + 1
    rhs_index = (np.abs(ar[mid:] - value)).argmin() + 1 + mid
    max_index = ar[lhs_index: rhs_index].argmax() + 1 + lhs_index
    return lhs_index, rhs_index, max_index


def get_applanation_length(primary_item):
    x, y = [np.array(primary_item.get(key)) for key in ['xs', 'y_up']]  # upper
    y_most = list(map(itemgetter(0), Counter(y).most_common(2)))
    y_flat = np.mean(y_most)

    x_most = x[np.where((y >= min(y_most)) & (y <= max(y_most)))]
    x_start = np.mean(sorted(x_most)[:5])
    x_end = np.mean(sorted(x_most)[-5:])
    return abs(x_end - x_start), y_flat


def get_applanation_velocity(primary_dicts,
                             first_flat_index, second_flat_index,
                             first_y_flat, second_y_flat, interval_frames=20):
    first = first_flat_index - interval_frames
    second = second_flat_index + interval_frames
    first_y_peak, second_y_peak = [np.sort(primary_dicts[index - 1].get('y_up'))[:5].mean()
                                   for index in [first, second]]
    sec = interval_frames / 15.0  # frame rate
    v_in = abs(first_y_flat - first_y_peak) / sec
    v_out = abs(second_y_flat - second_y_peak) / sec
    return v_in, v_out


def get_deformation_amplitude(primary_dicts, hc_index):
    y_init = np.sort(primary_dicts[0].get('y_up'))[:5].mean()
    y_concave = np.sort(primary_dicts[hc_index - 1].get('y_up'))[-5:].mean()
    return abs(y_init - y_concave)


def get_peak_distance(primary_dicts, hc_index):
    x, y = [np.array(primary_dicts[hc_index - 1].get(key)) for key in ['xs', 'y_up']]  # upper
    peak_index = find_peaks_cwt(y, np.arange(1, 10))
    peak_pair = sorted(list(zip(y[peak_index], peak_index + x[0])))  # (y, x)
    peak_y_1, peak_x_1 = peak_pair[0]
    for p in peak_pair[1:]:
        x_distance = abs(p[1] - peak_x_1)
        y_distance = abs(p[0] - peak_y_1)
        if x_distance > 200 and y_distance < 5:
            return x_distance
    return -1
