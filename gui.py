import os
import shutil
import subprocess

import numpy as np

from scipy.signal import savgol_filter

from flask import Flask, flash, redirect, render_template, request, session, abort, url_for, Response
from flask import send_from_directory

from flask_bootstrap import Bootstrap
from flask_dropzone import Dropzone

from utils import convert_to_frames, is_already_executed, \
    compute_curvature, get_applanation_time, get_applanation_length, get_applanation_velocity, \
    get_deformation_amplitude, get_peak_distance_and_hc_radius, generate_scatter3d_curve, get_peak_stat, \
    get_max_deformation_area, get_max_deformation_time, get_corneal_contour_deformation, get_energy_absorbed_area_and_k, \
    corneal_creep_rate

INTERPRETER_PATH = '/usr/local/bin/anaconda2/envs/pytorch/bin/python'

app = Flask(__name__)
Bootstrap(app)
dropzone = Dropzone(app)

app.config.update(
        SECRET_KEY='19960714',
        REPOSITORY_PATH=os.getcwd() + '/repo',
        DROPZONE_ALLOWED_FILE_TYPE='video',
        DROPZONE_MAX_FILE_SIZE=200,
        DROPZONE_MAX_FILES=12,
        DROPZONE_INPUT_NAME='video_dropzone'
        # DROPZONE_UPLOAD_MULTIPLE=True,
        # DROPZONE_PARALLEL_UPLOADS=4
)


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static/images'), 'favicon.ico', mimetype='image/x-icon')


@app.route('/', methods=['GET', 'POST'])
def index():
    # Update the video files in repository
    files = os.listdir(app.config['REPOSITORY_PATH'])

    if request.method == 'POST':
        # 1. Populate the warehouse
        f = request.files.get('video_dropzone')
        if f is not None:
            f.save(os.path.join(app.config['REPOSITORY_PATH'], f.filename))
            return render_template('index.html', **locals())

        # 2. Inspect the checked video
        checked_video_name = request.form['checked_video_name'][:-4]
        if checked_video_name:
            return redirect('/inspect/{}'.format(checked_video_name))

    return render_template('index.html', **locals())


@app.route('/clear_repo')
def clear_repo():
    # clear repo
    shutil.rmtree(app.config['REPOSITORY_PATH'])
    os.mkdir(app.config['REPOSITORY_PATH'])
    # clear infer
    shutil.rmtree('static/cache')
    os.mkdir('static/cache')
    return render_template('index.html', **locals())


def run_in_subprocess():
    proc = subprocess.Popen(
            [INTERPRETER_PATH + ' deploy.py'],
            shell=True,
            stdout=subprocess.PIPE,
            universal_newlines=True)
    return proc


@app.route('/run_all')
def run_all():
    if not is_already_executed():
        proc = run_in_subprocess()
        for line in iter(proc.stdout.readline, ''):
            line = line.strip()
            if line.startswith('===>'):
                flash(line)
            elif line.startswith('Oops'):
                flash('Please take a look at {} frame'.format(line.split()[-2]))
            print(line)
        flash('===> Done')
    elif not os.listdir('./repo'):
        flash('No video in the repository.')
    else:
        flash('Already done, please select and inspect one video directly.')
    return redirect('/')


@app.route('/inspect/<string:checked_video_name>')
def inspect(checked_video_name):
    # Convert video to frames and return the full path
    checked_frame_full_path = convert_to_frames(checked_video_name)
    checked_frame_path = [p[7:] for p in checked_frame_full_path]  # strip 'static/'
    # 'cache/frame/someone_someday/someone_someday_039.jpg'
    # ==> 'cache/infer/someone_someday/blend/039.jpg'
    checked_infer_path = []
    for p in checked_frame_path:
        i = p.split('_')[-1]  # e.g., '039.jpg'
        p = os.path.dirname(p).replace('frame', 'infer')
        checked_infer_path.append(os.path.join(p, 'blend', i))

    # Charts
    primary_dicts = np.load('static/cache/infer/primary_results_{}.npy'.format(checked_video_name))
    video_length = len(primary_dicts)
    thick_data = [round(pd['thick'] * 0.0165, 3) for pd in primary_dicts]
    thick_data_smoothed = [round(i, 3) for i in savgol_filter(thick_data, 7, 2)]
    min_thick, max_thick = [func(thick_data_smoothed) for func in [min, max]]
    curvatures = [round(compute_curvature(pd), 3) for pd in primary_dicts]
    curvatures_smoothed = [round(i, 3) for i in savgol_filter(curvatures, 7, 2)]

    # curve3d_lw = generate_scatter3d_curve(primary_dicts, curve_type='y_lw')
    curve3d_up = generate_scatter3d_curve(primary_dicts, curve_type='y_up')
    min_height, max_height = [int(func([i[2] for i in curve3d_up])) for func in [min, max]]

    # Bio-parameters
    first_AT, second_AT, HC_time = get_applanation_time(curvatures_smoothed, 0)
    (first_AL, first_y_flat), (second_AL, second_y_flat) = [get_applanation_length(primary_dicts[AT - 1])
                                                            for AT in [first_AT, second_AT]]
    v_in, v_out = get_applanation_velocity(primary_dicts, first_AT, second_AT, first_y_flat, second_y_flat)
    da = get_deformation_amplitude(primary_dicts, HC_time)
    pd, hc_radius, peak_x_1, peak_x_2 = get_peak_distance_and_hc_radius(primary_dicts, HC_time)

    # 0.231 ms/frame
    x_scale = 0.015625
    y_scale = 0.0165
    bio_params_1 = {
        'The first applanation time, 1AT'          : '{:.2f} ms'.format(first_AT * 0.231),
        'The first applanation length, 1AL'        : '{:.2f} mm'.format(first_AL * x_scale),
        'The first applanation velocity, V_in'     : '{:.2f} mm/ms'.format(v_in * y_scale),
        'The second applanation time, 2AT'         : '{:.2f} ms'.format(second_AT * 0.231),
        'The second applanation length, 2AL'       : '{:.2f} mm'.format(second_AL * x_scale),
        'The second applanation velocity, V_out'   : '{:.2f} mm/ms'.format(v_out * y_scale),
        'Start ==> highest concavity time, HC_time': '{:.2f} ms'.format(HC_time * 0.231),
        'Deformation amplitude, DA'                : '{:.2f} mm'.format(da * y_scale),
        'Peak distance, PD'                        : '{:.2f} mm'.format(pd * x_scale),
    }

    peak_deviation, v_inward = get_peak_stat(primary_dicts, HC_time)
    ma = get_max_deformation_area(primary_dicts, HC_time, peak_x_1, peak_x_2)
    ma_time = get_max_deformation_time(curvatures, HC_time)
    v_ccr = corneal_creep_rate(peak_deviation)
    ccd = get_corneal_contour_deformation(primary_dicts, HC_time)
    e_absorbed, k = get_energy_absorbed_area_and_k(peak_deviation)

    bio_params_2 = {
        'Central highest curvature radius, HC_r': '{:.2f} mm'.format(hc_radius),
        'Corneal inward velocity, V_inward'     : '{:.2f} mm/ms'.format(v_inward * y_scale),  # consider 33 to 42
        'Corneal creep rate, V_creep'           : '{:.2f} mm/ms'.format(v_ccr * y_scale / x_scale),
        'Corneal contour deformation, CCD'      : '{:.2f} mm'.format(ccd * y_scale),
        'Maximum deformation area, MA'          : '{:.2f} mm^2'.format(ma * x_scale * y_scale),
        'Maximum deformation area time, MA_time': '{:.2f} mm/ms'.format(ma_time * y_scale),
        'Energy absorbed area, A_absorbed'      : '{:.2f} '.format(e_absorbed),
        'Tangent stiffness coefficient,S_TSC'   : '{:.3f} '.format(k),
        'Central corneal thickness, CCT'        : 'see chart below',
    }

    return render_template('inspect.html', **locals())


if __name__ == '__main__':
    app.run(host="172.21.20.65", port=5003)
