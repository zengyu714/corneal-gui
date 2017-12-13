import os
import shutil
import subprocess

from scipy.signal import savgol_filter

from flask import Flask, flash, redirect, render_template, request
from flask import send_from_directory
from flask_bootstrap import Bootstrap
from flask_dropzone import Dropzone
from flask_script import Manager

from utils import convert_to_frames, is_already_executed, BioParams, generate_scatter3d_curve
from configuration import conf

app = Flask(__name__)
Bootstrap(app)
dropzone = Dropzone(app)
manager = Manager(app)

app.config.update(
        SECRET_KEY='19960714',
        REPOSITORY_PATH=conf.repository_path,
        DROPZONE_ALLOWED_FILE_TYPE='video',
        DROPZONE_MAX_FILE_SIZE=200,
        DROPZONE_MAX_FILES=12,
        DROPZONE_INPUT_NAME='video_dropzone'
        # DROPZONE_UPLOAD_MULTIPLE=True,
        # DROPZONE_PARALLEL_UPLOADS=4
)


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static/images'), 'favicon.png', mimetype='image/x-icon')


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
    # clear repo except corneal_demo.avi
    [os.remove(os.path.join(app.config['REPOSITORY_PATH'], p))
     for p in os.listdir(app.config['REPOSITORY_PATH']) if not p.startswith('corneal_demo')]

    # clear inference
    shutil.rmtree('static/cache')
    os.mkdir('static/cache')
    return render_template('index.html', **locals())


@app.route('/run_all')
def run_all():
    if not is_already_executed():
        proc = subprocess.Popen(
                [conf.interpreter_path + ' deploy.py'],
                shell=True,
                stdout=subprocess.PIPE,
                universal_newlines=True)
        for line in iter(proc.stdout.readline, ''):
            line = line.strip()
            if line.startswith('===>'):
                flash(line)
            elif line.startswith('Oops'):
                flash('Please take a look at {} frame'.format(line.split()[-2]))
            print(line)
        flash('===> Done')
    elif not os.listdir(app.config['REPOSITORY_PATH']):
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
    bio_param = BioParams(checked_video_name)

    thick_data = [round(pd['thick'] * conf.y_scale, 3) for pd in bio_param.primary_dicts]
    thick_data_smoothed = [round(i, 3) for i in savgol_filter(thick_data, 7, 2)]
    min_thick, max_thick = [func(thick_data_smoothed) for func in [min, max]]

    curvatures = bio_param.curvatures
    curvatures_smoothed = [round(i, 3) for i in savgol_filter(curvatures, 7, 2)]

    # curve3d_lw = generate_scatter3d_curve(primary_dicts, curve_type='y_lw')
    curve3d_up = generate_scatter3d_curve(bio_param.primary_dicts, curve_type='y_up')
    min_height, max_height = [int(func([i[2] for i in curve3d_up])) for func in [min, max]]

    # Bio-parameters
    first_AT = (bio_param.flat_index_1 + 1) * conf.frame_time_ratio
    first_AL = bio_param.flat_length_1 * conf.x_scale
    second_AT = (bio_param.flat_index_2 + 1) * conf.frame_time_ratio
    second_AL = bio_param.flat_length_2 * conf.x_scale

    v_in, v_out = bio_param.get_applanation_velocity()
    HC_time = bio_param.peak_index * conf.frame_time_ratio

    da = bio_param.get_deformation_amplitude() * conf.y_scale
    pd = (bio_param.peak_rhs_vertex[0] - bio_param.peak_lhs_vertex[0]) * conf.x_scale

    bio_params_1 = {
        'The first applanation time, 1AT'          : '{:.2f} ms'.format(first_AT),
        'The first applanation length, 1AL'        : '{:.2f} mm'.format(first_AL),
        'The first applanation velocity, V_in'     : '{:.3f} mm/ms'.format(v_in),
        'The second applanation time, 2AT'         : '{:.2f} ms'.format(second_AT),
        'The second applanation length, 2AL'       : '{:.2f} mm'.format(second_AL),
        'The second applanation velocity, V_out'   : '{:.3f} mm/ms'.format(v_out),
        'Start ==> highest concavity time, HC_time': '{:.2f} ms'.format(HC_time),
        'Deformation amplitude, DA'                : '{:.2f} mm'.format(da),
        'Peak distance, PD'                        : '{:.2f} mm'.format(pd),
    }

    hc_radius = bio_param.get_curvature_radius()
    v_inward = bio_param.get_inward_velocity()
    v_ccr = bio_param.get_corneal_creep_rate() * conf.y_scale / conf.x_scale
    ccd = bio_param.get_corneal_contour_deformation() * conf.y_scale
    ma = bio_param.get_max_deformation_area() * conf.x_scale * conf.y_scale

    # finished ================================================================================== <<<<<
    primary_dicts = bio_param.primary_dicts
    video_length = bio_param.video_length
    ma_time = bio_param.get_max_deformation_time() * conf.frame_time_ratio
    e_absorbed, k = bio_param.get_energy_absorbed_area_and_k()

    bio_params_2 = {
        'Central highest curvature radius, HC_r': '{:.2f} mm'.format(hc_radius),
        'Corneal inward velocity, V_inward'     : '{:.3f} mm/ms'.format(v_inward),  # consider 33 to 42
        'Corneal creep rate, V_creep'           : '{:.3f} mm/ms'.format(v_ccr),
        'Corneal contour deformation, CCD'      : '{:.2f} mm'.format(ccd),
        'Maximum deformation area, MA'          : '{:.2f} mm^2'.format(ma),
        'Maximum deformation area time, MA_time': '{:.2f} mm/ms'.format(ma_time),
        'Energy absorbed area, A_absorbed'      : '{:.2f} '.format(e_absorbed),
        'Tangent stiffness coefficient,S_TSC'   : '{:.3f} '.format(k),
        'Central corneal thickness, CCT'        : 'see chart below',
    }

    return render_template('inspect.html', **locals())


if __name__ == '__main__':
    manager.run()
