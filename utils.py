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

from configuration import conf, air_puff_force


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


def compute_curvature(primary_dicts, roi_deviation):
    """ Compute the curvature according to the specific area
    Argument:
        roi_deviation: (int) deviation from the middle of image that truly counted
    """

    def frame_curvature(primary_item, deviation):
        """
        Argument:
            primary_item: (dict) E.g., {'index': ..., 'thickness': ..., 'xs': ..., 'y_up': ..., 'y_lw': ...}
            deviation: (int) deviation from the middle of image that truly counted
        """
        middle = conf.frame_width // 2

        x, y_up, y_lw = [primary_item.get(key) for key in ['xs', 'y_up', 'y_lw']]
        bulk = [z for z in zip(*[x, y_up, y_lw]) if middle - deviation <= z[0] <= middle + deviation]

        x, y_up, _ = list(zip(*bulk))
        f_up = np.poly1d(np.polyfit(x, y_up, 2))
        # Derivative and compute the curvature in interested range, only care about the front surface
        return abs(np.polyval(np.polyder(f_up), x)).mean()

    return [round(frame_curvature(pd, roi_deviation), 3) for pd in primary_dicts]


def generate_scatter3d_curve(primary_dicts, curve_type='y_up'):
    """Show the curve in 3d scatter"""

    scatter3d_curve = []
    for i, pd in enumerate(primary_dicts):
        for j, pd_x in enumerate(pd['xs']):
            scatter3d_curve.append([i, pd_x, -pd[curve_type][j]])
    return scatter3d_curve


def get_applanation_time(curvatures, value=0):
    """Given the curvatures line is mirror symmetry in horizontal."""

    ar = np.array(curvatures)
    mid = len(ar) // 2
    lhs_index = (np.abs(ar[:mid] - value)).argmin()
    rhs_index = (np.abs(ar[mid:] - value)).argmin() + mid
    max_index = ar[lhs_index: rhs_index].argmax() + lhs_index
    return lhs_index, rhs_index, max_index


def get_applanation_length(primary_item):
    x, y = [np.array(primary_item[key]) for key in ['xs', 'y_up']]  # upper
    y_most = list(map(itemgetter(0), Counter(y).most_common(2)))
    y_flat = np.mean(y_most)

    x_most = x[np.where((y >= min(y_most)) & (y <= max(y_most)))]
    x_start = np.mean(sorted(x_most)[:5])
    x_end = np.mean(sorted(x_most)[-5:])
    return abs(x_end - x_start), y_flat


def get_peak_vertex(primary_dicts, peak_index):
    x, y = [np.array(primary_dicts[peak_index][key]) for key in ['xs', 'y_up']]
    peak_index = find_peaks_cwt(y, np.arange(1, 10))
    peak_pair = sorted(list(zip(y[peak_index], peak_index + x[0])))  # (y, x)

    # Middle lowest vertex with max y-index
    peak_y_1, peak_x_1 = peak_pair[0]
    peak_y_m, peak_x_m = peak_pair[-1]

    # Compute the peak distance
    peak_y_2, peak_x_2 = 0, 0
    for p in peak_pair[1:]:
        peak_y_2, peak_x_2 = p
        x_distance = abs(peak_x_2 - peak_x_1)
        y_distance = abs(peak_y_2 - peak_y_1)
        if x_distance > 200 and y_distance < 5:
            break
    # Make sure the left-hand-side is the one with smaller x-index
    peak_mid_vertex = peak_x_m, peak_y_m
    peak_lhs_vertex, peak_rhs_vertex = sorted([[peak_x_1, peak_y_1], [peak_x_2, peak_y_2]])
    return peak_lhs_vertex, peak_mid_vertex, peak_rhs_vertex


def get_peak_deviation(primary_dicts, peak_x, roi_width=3):
    """roi_width: interested width around the peak is 3"""
    peak_ys = np.array([np.mean(primary_dicts[i]['y_up'][peak_x - roi_width: peak_x + roi_width])
                        for i in range(len(primary_dicts))])
    # deviation
    return peak_ys - peak_ys[0]


class BioParams:
    """Compute the common biological parameters about corneal limbus"""

    def __init__(self, checked_video_name):
        self.primary_dicts = np.load('static/cache/infer/primary_results_{}.npy'.format(checked_video_name))
        self.video_length = len(self.primary_dicts)
        self.curvatures = compute_curvature(self.primary_dicts, roi_deviation=100)
        self.flat_index_1, self.flat_index_2, self.peak_index = get_applanation_time(self.curvatures)
        self.flat_length_1, self.flat_y_1 = get_applanation_length(self.primary_dicts[self.flat_index_1])
        self.flat_length_2, self.flat_y_2 = get_applanation_length(self.primary_dicts[self.flat_index_2])
        self.peak_lhs_vertex, self.peak_mid_vertex, self.peak_rhs_vertex = get_peak_vertex(self.primary_dicts,
                                                                                           self.peak_index)
        self.peak_deviation = get_peak_deviation(self.primary_dicts, self.peak_mid_vertex[0])

    def get_applanation_velocity(self, interval_frames=10):
        """Compute the velocity from initial peak to flat status by the interval
        Argument:
            interval_frames: the frame interval to get consumed time, timed by frame-time-ratio (0.231)
        """
        first = self.flat_index_1 - interval_frames
        second = self.flat_index_2 + interval_frames
        first_y_peak = np.sort(self.primary_dicts[first].get('y_up'))[:5].mean()
        second_y_peak = np.sort(self.primary_dicts[second].get('y_up'))[:5].mean()

        sec = interval_frames * conf.frame_time_ratio
        v_in = abs(self.flat_y_1 - first_y_peak) * conf.y_scale / sec
        v_out = abs(self.flat_y_2 - second_y_peak) * conf.y_scale / sec
        return v_in, v_out

    def get_deformation_amplitude(self):
        y_init = np.sort(self.primary_dicts[0].get('y_up'))[:5].mean()
        y_concave = np.sort(self.primary_dicts[self.peak_index].get('y_up'))[-5:].mean()
        return abs(y_init - y_concave)

    def get_curvature_radius(self):
        """Compute the highest concavity radius"""

        def circle_equation(c, points):
            D, E, F = c[0], c[1], c[2]
            return [x ** 2 + y ** 2 + D * x + E * y + F for x, y in points]

        radius = 0
        if any(self.peak_rhs_vertex):  # success to find another peak
            vertexs = np.array([self.peak_lhs_vertex, self.peak_rhs_vertex, self.peak_mid_vertex]).reshape([3, 2])
            points = vertexs * [conf.x_scale, conf.y_scale]
            # points = [[peak_x_1 * conf.x_scale, peak_y_1 * conf.y_scale],
            #           [peak_x_2 * conf.x_scale, peak_y_2 * conf.y_scale],
            #           [peak_x_m * conf.x_scale, peak_y_m * conf.y_scale]]
            D, E, F = fsolve(circle_equation, [0, 0, 0], args=(points))
            radius = np.sqrt(D ** 2 + E ** 2 - 4 * F) / 2
            # centroid = [-D / 2, -E / 2]
        return radius

    def get_inward_velocity(self):
        """Return the inward velocity between the 33th and the 42th frame."""
        peak_y_33 = self.peak_deviation[33 - 1]
        peak_y_42 = self.peak_deviation[42 - 1]
        v_inward = abs(peak_y_42 - peak_y_33) * conf.y_scale / ((42 - 33) * 0.231)  # frame rate
        return v_inward

    def get_corneal_creep_rate(self):
        trend = [b - a for a, b in zip(self.peak_deviation[::1], self.peak_deviation[1::1])]
        index = 1
        for neg_k, tr in groupby(trend, lambda k: k < - 0.0001):
            tr_list = list(tr)
            index += len(tr_list)
            if neg_k and np.mean(tr_list) < - 1:
                break

        xs = range(index, self.video_length)
        ys = self.peak_deviation[xs]

        linear_func = np.polyfit(xs, ys, deg=1)
        return linear_func[0]

    def get_corneal_contour_deformation(self):
        y_init = np.array(self.primary_dicts[0]['y_up'])[:20].mean()
        y_peak = np.array(self.primary_dicts[self.peak_index]['y_up'])[:20].mean()
        return abs(y_init - y_peak)

    def get_max_deformation_area(self):
        x_peak, y_peak = [np.array(self.primary_dicts[self.peak_index][key]) for key in ['xs', 'y_up']]
        y_init = np.array(self.primary_dicts[0]['y_up'])
        index = np.where((x_peak > self.peak_lhs_vertex[0]) & (x_peak < self.peak_rhs_vertex[0]))
        sum_area = abs(y_init[index] - y_peak[index]).sum()
        return sum_area

    def get_max_deformation_time(self):
        trend = [b - a for a, b in zip(self.curvatures[::1], self.curvatures[1::1])]
        index = 0
        for neg_k, tr in groupby(trend, lambda k: k < -0.0001):
            tr_list = list(tr)
            index += len(tr_list)
            if neg_k and np.mean(tr_list) < -0.003:
                break
        return self.peak_index - index

    def get_energy_absorbed_area_and_k(self):
        """Relationship: air_puff_force <--> peak_displacement
        Compute k by displacement between [0.2, 0.5]
        """
        epsilon = 0.03

        rel = np.array(list((zip(self.peak_deviation * conf.y_scale, air_puff_force))))
        start_i, end_i = [np.where(abs(rel[:, 0] - lim) < epsilon)[0][0] for lim in [0.2, 0.5]]

        x_displacement, y_puff = rel[end_i] - rel[start_i]
        k = y_puff / x_displacement

        max_displacement_index = rel.argmax(axis=0)[0]
        onload_area = rel[:max_displacement_index, 1].sum()
        unload_area = rel[max_displacement_index:, 1].sum()
        energy_absorbed_area = onload_area - unload_area
        return energy_absorbed_area, k
