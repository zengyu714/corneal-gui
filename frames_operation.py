import os
import imageio
from scipy.misc import imsave


def convert_to_frames(video_path):
    """Convert video to frames in the same data"""
    video_reader = imageio.get_reader(video_path, 'ffmpeg')
    for i, im in enumerate(video_reader, start=1):
        base_name = video_path.split('/')[-1][:-4]
        save_dir = 'static/cache/{}'.format(base_name)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        frame_name = '{}/{}_{:03d}.jpg'
        imsave(frame_name.format(save_dir, base_name, i), im, format='jpeg')


# if __name__ == '__main__':
#     convert_to_frames()
