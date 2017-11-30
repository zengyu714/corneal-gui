import os
import torch
import imageio
import warnings
import numpy as np

from glob import glob
from tqdm import tqdm
from torch.autograd import Variable
from scipy.misc import imread, imsave, imresize

from unet.model import UNetVanilla
from unet.helper import remove_watermark, post_process, traditional_seg, fitting_curve, blend

assert torch.cuda.is_available(), 'Error: CUDA not found!'

# Init
# ===========================================================================
model = UNetVanilla()
weights = torch.load('unet/weights/UNetVanilla_best.pth')
model.load_state_dict(weights)
model.eval().cuda()
print('===> Loading model...')

# template: binary image of watermark
# surround: dilated template for later computation of surrounding intensity
template_bw, surround_bw = [np.load('unet/weights/{}.npy'.format(f))
                            for f in ['template_bw', 'surround_bw']]


# ===========================================================================

def do_deploy():
    for video_name in os.listdir('repo'):
        video_name = video_name[:-4]
        print('===> Processing video: {}...'.format(video_name))
        frame_paths = sorted(glob('static/cache/frame/{}/*.jpg'.format(video_name)))
        thicks = []
        for i, p in tqdm(enumerate(frame_paths, start=1)):
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
                thicks.append(thickness)
            except (IndexError, TypeError):
                curve_mask = None
                thicks.append(0)
                print('Oops, fail to detect {}th frame...'.format(i))

            # Save and Display
            deploy_dir = ['static/cache/infer/{}/{}'.format(video_name, subdir) for subdir in ['bw', 'blend']]
            [os.makedirs(dd) for dd in deploy_dir if not os.path.exists(dd)]

            imsave(deploy_dir[0] + '/{:03d}.jpg'.format(i), display)
            imsave(deploy_dir[1] + '/{:03d}.jpg'.format(i), blend(im, display, curve_mask))

        np.save('static/cache/infer/thickness_{}.npy'.format(video_name), thicks)


def generate_video():
    for video_name in os.listdir('repo'):
        video_name = video_name[:-4]
        print('===> Generating inferred video: {}...'.format(video_name))
        with imageio.get_writer('static/cache/infer/blend_{}.mp4'.format(video_name), mode='I') as writer:
            for im_path in sorted(glob('static/cache/infer/{}/blend/*.jpg'.format(video_name))):
                image = imresize(imread(im_path), (208, 576))  # resize for video compatibility
                writer.append_data(image)


if __name__ == '__main__':
    torch.cuda.set_device(0)
    do_deploy()
    generate_video()
