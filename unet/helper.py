import numpy as np

from skimage.feature import corner_harris, corner_peaks
from skimage.filters import threshold_isodata
from skimage.morphology import remove_small_objects, convex_hull_image, \
    binary_closing, binary_opening, disk, rectangle, label
from skimage.segmentation import find_boundaries


def blend(image, label, curve_mask=None, alpha=0.3):
    """"Simulate colormap `jet`."""
    image, label = [item.astype(float) for item in [image, label]]
    r = label * alpha + 20
    b = (image + image.mean()) * (1 + alpha)
    g = np.minimum(r, b)
    rgb = np.dstack([r, g, b] + image * 0.3)
    if curve_mask is not None:
        # curve_mask = curve_mask[..., None]  # for broadcast
        # rgb += curve_mask * 0.5
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

    bw = label(mask == 1)
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
        curve_mask: the same shape as mask while labeled by 1
    """
    # 1. Find boundary
    bound = find_boundaries(mask, mode='outer')
    # 2. Crop marginal parts (may be noise)
    lhs, rhs = margin
    bound[:, :lhs] = 0  # left hand side
    bound[:, -rhs:] = 0  # right hand side
    # 3. Process upper and lower boundary respectively
    labeled_bound = label(bound, connectivity=bound.ndim)
    upper, lower = labeled_bound == 1, labeled_bound == 2
    # 1) fit poly
    f_up, f_lw = [np.poly1d(np.polyfit(np.where(limit)[1], np.where(limit)[0], 6)) for limit in [upper, lower]]
    # 2) interpolation
    width = mask.shape[1]
    x_cord = range(width)
    y_up_fit, y_lw_fit = [f(x_cord) for f in [f_up, f_lw]]
    rw = 30  # roi width
    thickness = (y_up_fit - y_lw_fit)[width // 2 - rw: width // 2 + rw]

    curve_mask = np.zeros_like(mask)
    y_up_fit, y_lw_fit = [np.array(y, dtype=int) for y in [y_up_fit, y_lw_fit]]  # int for slice
    curve_mask[y_up_fit[lhs: -rhs], x_cord[lhs: -rhs]] = 255
    curve_mask[y_lw_fit[lhs: -rhs], x_cord[lhs: -rhs]] = 255

    return abs(thickness.mean()), curve_mask



