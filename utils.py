import os
import csv
import imageio
import numpy as np

from operator import itemgetter
from itertools import groupby
from collections import Counter
from scipy.misc import imsave
from scipy.signal import find_peaks_cwt
from scipy.cluster import vq
from scipy.optimize import fsolve

# Read air puff and time
csv_reader = csv.reader(open('air_puff.csv'))
csv_file = np.array(list(zip(*list(csv_reader))))
air_puff_force, time = np.array(csv_file[[1, 2], 1:], dtype=float)


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


# TODO Optimize to class
def get_applanation_time(curvatures, value):
    """Consider the curvatures line is mirror symmetry in horizontal."""

    ar = np.array(curvatures)
    mid = len(ar) // 2
    lhs_index = (np.abs(ar[:mid] - value)).argmin() + 1
    rhs_index = (np.abs(ar[mid:] - value)).argmin() + 1 + mid
    max_index = ar[lhs_index: rhs_index].argmax() + 1 + lhs_index
    return lhs_index, rhs_index, max_index


def get_applanation_length(primary_item):
    x, y = [np.array(primary_item[key]) for key in ['xs', 'y_up']]  # upper
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
    sec = interval_frames * 0.231  # frame rate
    v_in = abs(first_y_flat - first_y_peak) / sec
    v_out = abs(second_y_flat - second_y_peak) / sec
    return v_in, v_out


def get_deformation_amplitude(primary_dicts, hc_index):
    y_init = np.sort(primary_dicts[0].get('y_up'))[:5].mean()
    y_concave = np.sort(primary_dicts[hc_index - 1].get('y_up'))[-5:].mean()
    return abs(y_init - y_concave)


def get_peak_distance_and_hc_radius(primary_dicts, hc_index):
    x, y = [np.array(primary_dicts[hc_index - 1][key]) for key in ['xs', 'y_up']]  # upper
    peak_index = find_peaks_cwt(y, np.arange(1, 10))
    peak_pair = sorted(list(zip(y[peak_index], peak_index + x[0])))  # (y, x)
    peak_y_1, peak_x_1 = peak_pair[0]
    peak_y_m, peak_x_m = peak_pair[-1]  # the peak middle lowest point with max y-index

    # Compute the peak distance
    peak_distance, peak_y_2, peak_x_2 = -1, -1, -1
    for p in peak_pair[1:]:
        peak_y_2, peak_x_2 = p
        x_distance = abs(peak_x_2 - peak_x_1)
        y_distance = abs(peak_y_2 - peak_y_1)
        if x_distance > 200 and y_distance < 5:
            peak_distance = x_distance
            break

    x_scale = 0.015625
    y_scale = 0.0165

    # Compute the highest concavity radius
    def circle_equation(c, points):
        D, E, F = c[0], c[1], c[2]
        return [x ** 2 + y ** 2 + D * x + E * y + F for x, y in points]

    radius = 0
    if peak_x_2 + peak_y_2 > 0:  # success to find another peak
        points = [[peak_x_1 * x_scale, peak_y_1 * y_scale],
                  [peak_x_2 * x_scale, peak_y_2 * y_scale],
                  [peak_x_m * x_scale, peak_y_m * y_scale]]
        D, E, F = fsolve(circle_equation, [0, 0, 0], args=(points))
        radius = np.sqrt(D ** 2 + E ** 2 - 4 * F) / 2
        # centroid = [-D / 2, -E / 2]
    p_x_1 = min(peak_x_1, peak_x_2)
    p_x_2 = max(peak_x_1, peak_x_2)
    return peak_distance, radius, p_x_1, p_x_2


def generate_scatter3d_curve(primary_dicts, curve_type='y_up'):
    """Show the curve in 3d scatter"""

    scatter3d_curve = []
    for i, pd in enumerate(primary_dicts):
        for j, pd_x in enumerate(pd['xs']):
            scatter3d_curve.append([i, pd_x, -pd[curve_type][j]])
    return scatter3d_curve


def get_peak_stat(primary_dicts, hc_index, roi_width=3):
    """Return deviation between peaks and
    the specific velocity between (33, 42) in the inward stage.
    """

    x, y = [np.array(primary_dicts[hc_index - 1][key]) for key in ['xs', 'y_up']]  # upper
    peak_index = find_peaks_cwt(y, np.arange(1, 10))
    peak_pair = sorted(list(zip(y[peak_index], peak_index + x[0])))  # (y, x)
    peak_y_m, peak_x_m = peak_pair[-1]

    peak_y = np.array([np.mean(primary_dicts[i - 1]['y_up'][peak_x_m - roi_width: peak_x_m + roi_width])
                       for i in range(len(primary_dicts))])
    v_inward = abs(peak_y[42 - 1] - peak_y[33 - 1]) / ((42 - 33) * 0.231)  # frame rate
    peak_deviation = abs(peak_y - peak_y[0])
    return peak_deviation, v_inward


def get_corneal_contour_deformation(primary_dicts, hc_index):
    y_init = np.array(primary_dicts[0]['y_up'])[:20].mean()
    y_peak = np.array(primary_dicts[hc_index - 1]['y_up'])[:20].mean()
    return abs(y_init - y_peak)


def get_max_deformation_area(primary_dicts, hc_index, peak_x_1, peak_x_2):
    x_peak, y_peak = [np.array(primary_dicts[hc_index - 1][key]) for key in ['xs', 'y_up']]
    y_init = np.array(primary_dicts[0]['y_up'])
    index = np.where((x_peak > peak_x_1) & (x_peak < peak_x_2))
    sum_area = abs(y_init[index] - y_peak[index]).sum()
    return sum_area


def get_max_deformation_time(curvatures, hc_index):
    trend = [b - a for a, b in zip(curvatures[::1], curvatures[1::1])]
    index = 1
    for neg_k, tr in groupby(trend, lambda k: k < -0.0001):
        tr_list = list(tr)
        index += len(tr_list)
        if neg_k and np.mean(tr_list) < -0.003:
            break
    return (hc_index - index) * 0.231


def get_energy_absorbed_area_and_k(peak_deviation):
    """
    air_puff_force <--> peak_displacement
    Compute k by displacement between [0.2, 0.5]
    """
    y_scale = 0.0165
    epsilon = 0.03

    rel = np.array(list((zip(peak_deviation * y_scale, air_puff_force))))
    start_i, end_i = [np.where(abs(rel[:, 0] - lim) < epsilon)[0][0] for lim in [0.2, 0.5]]

    x_displacement, y_puff = rel[end_i] - rel[start_i]
    k = y_puff / x_displacement

    max_displacement_index = rel.argmax(axis=0)[0]
    onload_area = rel[:max_displacement_index, 1].sum()
    unload_area = rel[max_displacement_index:, 1].sum()
    energy_absorbed_area = onload_area - unload_area
    return energy_absorbed_area, k


def corneal_creep_rate(peak_deviation):
    trend = [b - a for a, b in zip(peak_deviation[::1], peak_deviation[1::1])]
    index = 1
    for neg_k, tr in groupby(trend, lambda k: k < - 0.0001):
        tr_list = list(tr)
        index += len(tr_list)
        if neg_k and np.mean(tr_list) < - 1:
            break

    xs = range(index, len(peak_deviation))
    ys = peak_deviation[xs]

    linear_func = np.polyfit(xs, ys, deg=1)
    return linear_func[0]
