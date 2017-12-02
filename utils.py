import os
import imageio

from scipy.misc import imsave


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
