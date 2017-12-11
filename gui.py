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
    get_deformation_amplitude, get_peak_distance, generate_bar3d_curve

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
    thick_data = [round(pd['thick'], 3) for pd in primary_dicts]
    curvatures = [round(compute_curvature(pd), 3) for pd in primary_dicts]

    # curve3d_lw = generate_bar3d_curve(primary_dicts, curve_type='y_lw')
    curve3d_up = generate_bar3d_curve(primary_dicts, curve_type='y_up')
    min_height, max_height = [int(func([i[2] for i in curve3d_up])) for func in [min, max]]

    # smooth
    thick_data_smoothed = [round(i, 3) for i in savgol_filter(thick_data, 7, 2)]
    curvatures_smoothed = [round(i, 3) for i in savgol_filter(curvatures, 7, 2)]
    min_thick, max_thick = [int(func(thick_data_smoothed)) for func in [min, max]]
    # Bio-parameters
    first_AT, second_AT, HC_time = get_applanation_time(curvatures_smoothed, 0)
    (first_AL, first_y_flat), (second_AL, second_y_flat) = [get_applanation_length(primary_dicts[AT - 1])
                                                            for AT in [first_AT, second_AT]]
    v_in, v_out = get_applanation_velocity(primary_dicts, first_AT, second_AT, first_y_flat, second_y_flat)
    da = get_deformation_amplitude(primary_dicts, HC_time)
    pd = get_peak_distance(primary_dicts, HC_time)

    # 15 is frame rate
    bio_params_1 = {
        'The first applanation time, 1AT'          : '{:>10.2f} s'.format(first_AT / 15.0),
        'The first applanation length, 1AL'        : '{:>10.0f} pixels'.format(first_AL),
        'The first applanation velocity, V_in'     : '{:>10.2f} pixels/s'.format(v_in),
        'The second applanation time, 2AT'         : '{:>10.2f} s'.format(second_AT / 15.0),
        'The second applanation length, 2AL'       : '{:>10.0f} pixels'.format(second_AL),
        'The second applanation velocity, V_out'   : '{:>10.2f} pixels/s'.format(v_out),
        'Start ==> highest concavity time, HC_time': '{:>10.2f} s'.format(HC_time / 15.0),
        'Deformation amplitude, DA'                : '{:>10.0f} pixels'.format(da),
        'Peak distance, PD'                        : '{:>10.0f} pixels'.format(pd),
        'Central corneal thickness, CCT'           : 'see chart below',
    }

    keys_2 = ['V_in_max', 'V_out_max', 'V_creep', 'CCD', 'HC_radius',
              'MA', 'MA_time', 'A_absorbed', 'S_TSC']
    bio_params_2 = {
        'Central curvature radius at, HC HC_radius'  : 0,
        'Maximum corneal inward velocity, V_in_max'  : 0,
        'Maximum corneal outward velocity, V_out_max': 0,
        'Corneal creep rate, V_creep'                : 0,
        'Corneal contour deformation, CCD'           : 0,
        'Maximum deformation area, MA'               : 0,
        'Maximum deformation area time, MA_time'     : 0,
        'Energy absorbed area, A_absorbed'           : 0,
        'Tangent stiffness coefficient,S_TSC'        : 0,
    }

    bio_params_2 = dict.fromkeys(keys_2, 0.0)

    return render_template('inspect.html', **locals())


if __name__ == '__main__':
    app.run(host="172.21.20.65", port=5003)
