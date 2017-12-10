import os
import imageio
import numpy as np

from operator import itemgetter
from itertools import groupby
from scipy.misc import imsave
from scipy.cluster import vq
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
        xs: (array) the sequential column index of input image
        y_up: (array) the row index which is relatively smaller
        y_lw: (array) the row index which is relatively larger
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
    return [np.array(item) for item in [xs, y_up, y_lw]]


def compute_curvature(primary_item, roi_deviation=50, im_width=576):
    """
    Argument:
        primary_item: (dict) E.g., {'index': ..., 'thickness': ..., 'curve_mask': ...}
        roi_deviation: (int) deviation from the middle of image that truly counted
    """
    middle = im_width // 2
    xn = range(middle - roi_deviation, middle + roi_deviation)

    # Permute to (x, y) and sort by x-axis
    xs, y_up, y_lw = [primary_item.get(key)[xn] for key in ['xs', 'y_up', 'y_lw']]
    f_up = np.poly1d(np.polyfit(xs, y_up, 2))
    # Derivative and compute the curvature in interested range
    up = np.polyval(np.polyder(f_up), xn).mean()
    # Care about the front surface
    return up


def get_applanation_time(curvatures, value):
    """Consider the curvatures line is mirror symmetry in horizontal."""

    ar = np.array(curvatures)
    mid = len(ar) // 2
    lhs_index = (np.abs(ar[:mid] - value)).argmin() + 1
    rhs_index = (np.abs(ar[mid:] - value)).argmin() + 1 + mid
    min_index = ar.argmin()
    return lhs_index, rhs_index, min_index


def get_applanation_length(primary_item):
    x, y = [primary_item.get(key) for key in ['xs', 'y_up']]  # upper
    y_most = list(map(itemgetter(0), Counter(y).most_common(2)))
    x_most = x[np.where((y >= min(y_most)) & (y <= max(y_most)))]

    # find the longest continuous x indexes
    candidate = [list(map(itemgetter(1), g))
                 for _, g in groupby(enumerate(sorted(x_most)), lambda i: i[0] - i[1])]  # index[0], value[1]
    return len(sorted(candidate, key=len)[-1])
