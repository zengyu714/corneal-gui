import os
import sys
import torch
import imageio
import warnings
import numpy as np

from glob import glob
from tqdm import tqdm
from torch.autograd import Variable
from scipy.misc import imread, imsave, imresize

from utils import convert_to_frames
from unet.model import UNetVanilla
from unet.helper import remove_watermark, post_process, \
    traditional_seg, fitting_curve, blend

assert torch.cuda.is_available(), 'Error: CUDA not found!'

# Init
# ===========================================================================
model = UNetVanilla()
weights = torch.load('unet/weights/UNetVanilla_best.pth')
model.load_state_dict(weights)
model.eval().cuda()
print('\n\n', ' Loading model '.center(110, '='))
# template: binary image of watermark
# surround: dilated template for later computation of surrounding intensity
template_bw, surround_bw = [np.load('unet/weights/{}.npy'.format(f))
                            for f in ['template_bw', 'surround_bw']]


# ===========================================================================

def do_deploy():
    for video_name in os.listdir('repo'):
        video_name = video_name[:-4]
        print('\n\n', ' Processing video: 【{}.avi】 '.format(video_name).center(120, '='))
        frame_paths = sorted(glob('static/cache/frame/{}/*.jpg'.format(video_name)))

        primary_results = []
        for i, p in tqdm(enumerate(frame_paths, start=1), file=sys.stdout,
                         total=len(frame_paths), unit=' frames', dynamic_ncols=True):
            im = remove_watermark(imread(p, mode='L'), template_bw, surround_bw)
            x = Variable(torch.from_numpy(im[None, None, ...]), volatile=True).float().cuda()
            # if image is the first frame of a video
            # then get the segmentation by traditional threshold method
            if i == 1:
                bw = traditional_seg(im)
                y_prev = Variable(torch.from_numpy(bw[None, None, ...]), volatile=True).float().cuda()
            output = model(x, y_prev)
            y = output.data.max(1)[1]
            y_prev = Variable(y[None, ...], volatile=True).float()  # as next frame's input

            y = y[0].cpu().numpy()
            # post-process
            with warnings.catch_warnings():
                # ignore the warning about remove_small_objects
                warnings.simplefilter("ignore")
                display = post_process(y) * 255
            try:
                thickness, curve_mask = fitting_curve(display)
            except (IndexError, TypeError):
                thickness, curve_mask = 0, None
                tqdm.write('Oops, fail to detect {}th frame...'.format(i))

            # Append and save results
            primary_results.append(
                    {'index': i - 1, 'thick': thickness, 'curve_mask': curve_mask})

            # Save and Display
            deploy_dir = ['static/cache/infer/{}/{}'.format(video_name, subdir) for subdir in ['bw', 'blend']]
            [os.makedirs(dd) for dd in deploy_dir if not os.path.exists(dd)]

            imsave(deploy_dir[0] + '/{:03d}.jpg'.format(i), display)
            imsave(deploy_dir[1] + '/{:03d}.jpg'.format(i), blend(im, display, curve_mask))

        np.save('static/cache/infer/primary_results_{}.npy'.format(video_name), primary_results)


def generate_video():
    for video_name in os.listdir('repo'):
        video_name = video_name[:-4]
        print('\n\n', ' Generating inferred video: 【{}.mp4】 '.format(video_name).center(120, '='))
        # Generate inferred video and make original frames into video in mp4 format
        with imageio.get_writer('static/cache/infer/blend_{}.mp4'.format(video_name), mode='I') as writer:
            for im_path in sorted(glob('static/cache/infer/{}/blend/*.jpg'.format(video_name))):
                image = imresize(imread(im_path), (208, 576))  # resize for video compatibility
                writer.append_data(image)
        with imageio.get_writer('static/cache/infer/original_{}.mp4'.format(video_name), mode='I') as writer:
            for im_path in sorted(glob('static/cache/frame/{}/*.jpg'.format(video_name))):
                image = imresize(imread(im_path), (208, 576))  # resize for video compatibility
                writer.append_data(image)


if __name__ == '__main__':
    torch.cuda.set_device(0)
    # 1. Generate all frames of the video in the repository
    # [convert_to_frames(video_name) for video_name in os.listdir('repo')]
    # 2. Do the deploy
    do_deploy()
    # 3. Output the video for visualization
    generate_video()
