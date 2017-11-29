import os
import shutil

from flask import Flask, flash, redirect, render_template, request, session, abort, url_for
from flask import send_from_directory

from flask_bootstrap import Bootstrap
from flask_dropzone import Dropzone

from utils import convert_to_frames

app = Flask(__name__)
Bootstrap(app)
dropzone = Dropzone(app)

app.config.update(
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
    shutil.rmtree(app.config['REPOSITORY_PATH'])
    os.mkdir(app.config['REPOSITORY_PATH'])
    return render_template('index.html', **locals())


@app.route('/inspect/<string:checked_video_name>')
def inspect(checked_video_name):
    # Convert video to frames and return the full path
    checked_frame_full_path = convert_to_frames(checked_video_name)
    checked_frame_path = [p[7:] for p in checked_frame_full_path]  # strip 'static/'
    return render_template('inspect.html',  **locals())


if __name__ == '__main__':
    app.run(host="172.21.20.65", port=5003)
