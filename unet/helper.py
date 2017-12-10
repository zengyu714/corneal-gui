import numpy as np

from scipy.signal import savgol_filter
from skimage.feature import corner_harris, corner_peaks
from skimage.filters import threshold_isodata
from skimage.morphology import remove_small_objects, convex_hull_image, \
    binary_closing, binary_opening, disk, rectangle
from skimage.morphology import label as morphology_abel
from skimage.segmentation import find_boundaries

from utils import isolate_bound


def blend(image, label, coords=[None] * 3, alpha=0.3):
    """"Simulate colormap `jet`."""
    image, label = [item.astype(float) for item in [image, label]]
    r = label * alpha + 20
    b = (image + image.mean()) * (1 + alpha)
    g = np.minimum(r, b)
    rgb = np.dstack([r, g, b] + image * 0.3)
    if coords[0] is not None:
        # curve_mask = curve_mask[..., None]  # for broadcast
        # rgb += curve_mask * 0.5
        xs, y_up_fit, y_lw_fit = coords
        if len(xs) == len(y_up_fit) == len(y_lw_fit):
            curve_mask = np.zeros_like(label)
            curve_mask[y_up_fit.astype(int), xs] = 255
            curve_mask[y_lw_fit.astype(int), xs] = 255
            rgb[..., 1] += curve_mask * 0.5  # add to blue channel
    # vis.image(rgb.transpose(2, 0, 1))
    return rgb.astype(np.uint8)


def remove_watermark(frame, template_bw, surround_bw):
    """Remove watermark by fill surrounding intensity."""
    # Gray value around template
    surround_intensity = frame[surround_bw].mean()
    # Subtract
    demark = np.select([~template_bw], [frame], default=surround_intensity)
    return demark


def traditional_seg(im):
    tsh = threshold_isodata(im, return_all=False)
    bw = im > tsh
    bw = binary_closing(bw, selem=disk(3))
    bw = binary_opening(bw, selem=disk(3))
    bw = remove_small_objects(bw, min_size=1024)
    return bw.astype(int)


def post_process(mask):
    """Mainly remove the convex areas"""

    bw = morphology_abel(mask == 1)
    # 0) fill gap
    bw = binary_closing(bw, disk(3))
    # 1) detach mislabeled pixels
    bw = binary_opening(bw, rectangle(2, 20))
    # 2) remove small objects
    bw = remove_small_objects(bw, min_size=4096, connectivity=2)
    # 3) solve the defeat, typically the convex outline
    coords = corner_peaks(corner_harris(bw, k=0.2), min_distance=5)
    valid = [c for c in coords if 100 < c[1] < 476]  # only cares about this valid range
    if valid:
        y, x = zip(*valid)
        # corners appear in pair
        if len(y) % 2 == 0:
            # select the lowest pair
            left_x, right_x = [func(x[0], x[1]) for func in (min, max)]
            sep_x = np.arange(left_x, right_x + 1).astype(int)
            sep_y = np.floor(np.linspace(y[0], y[1] + 1, len(sep_x))).astype(int)
            # make the gap manually
            bw[sep_y, sep_x] = 0
            bw = binary_opening(bw, disk(6))
        else:
            mask = np.zeros_like(bw)
            mask[y, x] = 1
            chull = convex_hull_image(mask)
            bw = np.logical_xor(chull, bw)
            bw = binary_opening(bw, disk(6))
    return bw


def fitting_curve(mask, margin=(60, 60)):
    """Compute thickness by fitting the curve
    Argument:
        margin: indicate valid mask region in case overfit

    Return:
        thickness: between upper and lower limbus
        (xs, y_up_fit, y_lw_fit): coordinates of corresponding results
    """
    # 1. Find boundary
    bound = find_boundaries(mask > 127, mode='outer')
    # 2. Crop marginal parts (may be noise)
    lhs, rhs = margin
    bound[:, :lhs] = 0  # left hand side
    bound[:, -rhs:] = 0  # right hand side
    # 3. Process upper and lower boundary respectively
    xs, y_up, y_lw = isolate_bound(bound)
    # 1) fit poly
    f_up, f_lw = [np.poly1d(np.polyfit(xs, ys, 6)) for ys in [y_up, y_lw]]
    # 2) interpolation
    rw, width = 30, mask.shape[1]  # roi width
    y_up_fit, y_lw_fit = [f(xs) for f in [f_up, f_lw]]
    thickness = (y_up_fit - y_lw_fit)[width // 2 - rw: width // 2 + rw]

    return abs(thickness.mean()), (xs, y_up_fit, y_lw_fit)


def interp1d_curve(mask, margin=(60, 60)):
    """Compute thickness by interpolation
    Argument:
        margin: indicate valid mask region in case overfit

    Return:
        thickness: between upper and lower limbus
        (xs, y_up_interp, y_lw_interp): coordinates of corresponding results
    """
    # 1. Find boundary
    bound = find_boundaries(mask > 127, mode='outer')
    # 2. Crop marginal parts (may be noise)
    lhs, rhs = margin
    bound[:, :lhs] = 0  # left hand side
    bound[:, -rhs:] = 0  # right hand side
    # 3. Process upper and lower boundary respectively
    xs, y_up, y_lw = isolate_bound(bound)
    # 1) interp1d
    y_up_interp, y_lw_interp = [savgol_filter(y, 27, 2) for y in [y_up, y_lw]]
    # 2) get thickness
    rw, width = 30, mask.shape[1]  # roi width
    thickness = (y_up_interp - y_lw_interp)[width // 2 - rw: width // 2 + rw]

    return abs(thickness.mean()), (xs, y_up_interp, y_lw_interp)
