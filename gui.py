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
    compute_curvature, get_applanation_time, get_applanation_length

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
        flash('Take around 8 s/video, click the ✈ above to see console output details.')
        yeild()
    elif not os.listdir('./repo'):
        flash('No video in the repository.')
    else:
        flash('Already done, please select and inspect one video directly.')
    return redirect('/')


@app.route('/yield')
def yeild():
    def inner():
        proc = run_in_subprocess()
        for line in iter(proc.stdout.readline, ''):
            yield line.rstrip() + '<br/>\n'

    # text/html is required for most browsers to show th$
    return Response(inner(), mimetype='text/html')


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
    # line_xs = [*range(1, video_length + 1)]
    thick_data = [round(pd['thick'], 3) for pd in primary_dicts]
    curvatures = [round(compute_curvature(pd), 3) for pd in primary_dicts]
    # smooth
    thick_data_smoothed = [round(i, 3) for i in savgol_filter(thick_data, 13, 2)]
    curvatures_smoothed = [round(i, 3) for i in savgol_filter(curvatures, 13, 2)]
    # Bio-parameters
    first_AT, second_AT, HC_time = get_applanation_time(curvatures_smoothed, 0)
    print(first_AT, second_AT, '-' * 30, 'first_AT', 'second_AT')

    first_AL, second_AL = [get_applanation_length(primary_dicts[AT - 1]) for AT in [first_AT, second_AT]]
    print(first_AL, second_AL,  '-' * 30, 'first_AL', 'second_AL')

    keys_1 = ['The first applanation time, 1AT',
              '1AL', 'V_in', '2AT', '2AL', 'V_out',
              'CCT', 'HC_time', 'DA', 'PD'
              ]
    # 15 is frame rate
    bio_params_1 = {
        'The first applanation time, 1AT'                         : first_AT / 15.0,
        'The first applanation length, 1AL'                       : 0,
        'The first applanation velocity, V_in'                    : 0,
        'The second applanation time, 2AT'                        : second_AT / 15.0,
        'The second applanation length, 2AL'                      : 0,
        'The second applanation velocity, V_out'                  : 0,
        'Central corneal thickness, CCT'                          : 'see chart below',
        'Time from the start until the highest concavity, HC_time': HC_time / 15.0,
        'Deformation amplitude, DA'                               : 0,
        'Peak distance, PD'                                       : 0
    }

    bio_params_1 = dict.fromkeys(keys_1, 0.0)

    keys_2 = ['V_in_max', 'V_out_max', 'V_creep', 'CCD', 'HC_radius',
              'MA', 'MA_time', 'A_absorbed', 'S_TSC']
    bio_params_2 = dict.fromkeys(keys_2, 0.0)

    return render_template('inspect.html', **locals())


if __name__ == '__main__':
    app.run(host="172.21.20.65", port=5003)
