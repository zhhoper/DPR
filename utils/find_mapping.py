import cv2
import numpy as np


def create_lighting_filter(img_x: np.ndarray, img_y: np.ndarray, epsilon: float=1e-5):
    lab_x = cv2.cvtColor(img_x, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_y = cv2.cvtColor(img_y, cv2.COLOR_BGR2LAB).astype(np.float32)

    lx = lab_x[:, :, 0]
    ly = lab_y[:, :, 0]

    gain_map = ly / (lx + epsilon)

    gain_map = np.clip(gain_map, 0.1, 10.0)
    radius = 100
    eps = 0.1 * (255 ** 2)
    smooth_gain = cv2.ximgproc.guidedFilter(guide=lx.astype(np.uint8),
                                            src=gain_map,
                                            radius=radius,
                                            eps=eps)
    return smooth_gain, lab_x
    # return gain_map, lab_x

def apply_darkening_from_gain(smooth_gain: np.ndarray,
                              shadow_strength: float = 0.4,
                              curve_power: float = 1.5):
    """
    Darkens the lighting filter itself, without using the image (L-channel).
    Only modifies `smooth_gain` directly.
    """

    # Normalize gain to [0, 1]
    g = smooth_gain.astype(np.float32)
    g_norm = g / g.max()

    # Create a darkening curve entirely from the gain itself
    # High gain → brighter areas → get reduced less
    # Low gain → naturally darker → get reduced more
    darkening_curve = (1.0 - shadow_strength * (1.0 - g_norm) ** curve_power)

    # Final filter
    final_gain = g * darkening_curve

    return final_gain

def apply_shadow_enhancement(lab_x: np.ndarray, smooth_gain: np.ndarray, shadow_strength: float = 0.4):
    
    lx = lab_x[:, :, 0].astype(np.float32)

    dark_weight = 1.0 - (lx / 255.0)

    dark_weight = dark_weight ** 2.5

    darkening_filter = 1.0 - shadow_strength * dark_weight

    final_gain = smooth_gain * darkening_filter

    new_l = lx * final_gain
    new_l = np.clip(new_l, 0, 255)

    result_lab = lab_x.copy()
    result_lab[:, :, 0] = new_l

    result_bgr = cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return result_bgr, final_gain

def apply_lighting_filter(target_img: np.ndarray, gain_filter: np.ndarray):
    
    lab = cv2.cvtColor(target_img, cv2.COLOR_BGR2LAB).astype(np.float32)

    lab[:, :, 0] = lab[:, :, 0] * gain_filter
    
    lab[:, :, 0] = np.clip(lab[:, :, 0], 0, 255)

    result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return result
