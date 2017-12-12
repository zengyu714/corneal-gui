import os
import csv
import numpy as np


def read_air_puff_file(file_path):
    """Read air puff data (the 1st column)"""

    csv_reader = csv.reader(open(file_path))
    csv_file = np.array(list(zip(*list(csv_reader))))
    return np.array(csv_file[1, 1:], dtype=float)


class Configuration:
    def __init__(self):
        # Path dependencies
        # ========================================================================
        # path to the python interpreter
        self.interpreter_path = '/usr/local/bin/anaconda2/envs/pytorch/bin/python'
        # path to the repository, suggest not editing it
        self.repository_path = os.getcwd() + '/repo'
        # path to air puff force data
        self.air_puff_path = os.getcwd() + '/air_puff.csv'
        # ========================================================================

        # Calibration
        # ========================================================================
        # mm/pixel in x-axis
        self.x_scale = 0.015625
        # mm/pixel in y-axis
        self.y_scale = 0.0165
        # ms/frame
        self.frame_time_ratio = 0.231
        # frame width
        self.frame_width = 576
        # frame height
        self.frame_height = 200
        # ========================================================================


conf = Configuration()
air_puff_force = read_air_puff_file(conf.air_puff_path)
