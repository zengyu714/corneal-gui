import os
import imageio
import numpy as np

from operator import itemgetter
from itertools import groupby
from scipy.misc import imsave
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
        upper: array([x, ...], [y, ...]), the row index is relatively smaller
        lower: array([x, ...], [y, ...]), the row index is relatively bigger
    """
    x_sort = sorted(list(zip(*np.where(im > 0)[::-1])))
    upper, lower = [np.array(list(zip(*x_sort[i::2]))) for i in [0, 1]]
    return upper, lower


def compute_curvature(primary_item, roi_deviation=50, im_width=576):
    """
    Argument:
        primary_item: (dict) E.g., {'index': ..., 'thickness': ..., 'curve_mask': ...}
        roi_deviation: (int) deviation from the middle of image that truly counted
    """
    middle = im_width // 2
    xn = range(middle - roi_deviation, middle + roi_deviation)
    roi = primary_item['curve_mask'][:, xn]

    # upper, lower = isolate_bound(roi)
    # f_up, f_lw = [np.poly1d(np.polyfit(x, y, 2)) for x, y in [upper, lower]]
    # up, lw = [np.polyval(np.polyder(item), xn).mean() for item in [f_up, f_lw]]
    # return up + lw

    # Permute to (x, y) and sort by x-axis
    upper, _ = isolate_bound(roi)
    f_up = np.poly1d(np.polyfit(*upper, 2))
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
    (x, y), _ = isolate_bound(primary_item['curve_mask'])
    y_common = Counter(y).most_common(2)[0][0]
    x_common = x[np.where(y == y_common)]

    # find the longest continuous x indexes
    candidate = [list(map(itemgetter(1), g))
               for _, g in groupby(enumerate(sorted(x_common)), lambda i: i[0] - i[1])]  # index[0], value[1]
    return len(sorted(candidate, key=len)[-1])