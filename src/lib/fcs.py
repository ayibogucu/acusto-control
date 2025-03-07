from itertools import count

import numpy as np
from pipython import GCSDevice, pitools
from pypylon import pylon

from lib.cmr import return_single_image
from lib.cnf import Config


class FocusMaximizer:
    def __init__(self, patience=3):
        self.best_score = None
        self.best_index = None
        self.current_index = 0
        self.patience = patience
        self.drop_counter = 0

    def process_score(self, score):
        if self.best_score is None:
            self.best_score = score
            self.best_index = self.current_index
            self.current_index += 1
            return (False, None, None)

        if score > self.best_score:
            self.best_score = score
            self.best_index = self.current_index
            self.drop_counter = 0
        else:
            if score < self.best_score:
                self.drop_counter += 1

        self.current_index += 1

        if self.drop_counter >= self.patience:
            return (True, self.best_index, self.best_score)
        else:
            return (False, None, None)


def move_to_focus(
    pidevice: GCSDevice,
    camera: pylon.InstantCamera,
    config: Config,
) -> float:
    focus_maximizer = FocusMaximizer(patience=5)
    for idx in count(0):
        target_z = config.movement.min_z + config.movement.dz * idx
        pidevice.MOV(config.axes.z, target_z)
        pitools.waitontarget(pidevice, config.axes.z)
        img = return_single_image(camera)
        score = _fft_focus_measure(img)
        print(f"At index {idx}, FFT Focus Score: {score:.4f}")

        max_found, max_index, max_score = focus_maximizer.process_score(score)
        if max_found:
            print(f"Maximum detected at index {max_index} with score {max_score}")
            break

    best_target_z = config.movement.min_z + config.movement.dz * max_index
    pidevice.MOV(config.axes.z, best_target_z)
    pitools.waitontarget(pidevice, config.axes.z)
    return best_target_z


def _fft_focus_measure(gray, low_freq_radius_ratio=0.1):
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)

    total_energy = np.sum(magnitude_spectrum)

    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2

    radius = int(low_freq_radius_ratio * min(rows, cols))
    Y, X = np.ogrid[:rows, :cols]
    distance = np.sqrt((Y - crow) ** 2 + (X - ccol) ** 2)
    low_freq_mask = distance <= radius

    high_freq_energy = np.sum(magnitude_spectrum[~low_freq_mask])

    ratio = high_freq_energy / total_energy
    return ratio
