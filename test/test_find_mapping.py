import cv2
import numpy as np
from utils.find_mapping import create_lighting_filter, apply_lighting_filter, apply_shadow_enhancement, apply_darkening_from_gain
import unittest


class MyTestCase(unittest.TestCase):
    def test_create_lighting_filter(self):
        refrence_image: np.ndarray = cv2.imread("/home/abolfazl/Documents/DPR/obama.jpg")
        target_image: np.ndarray = cv2.imread("candidiate_config/candidate1/result/rotate_light_06.jpg")
        smooth_gain, b = create_lighting_filter(img_x=refrence_image, img_y=target_image)
        cv2.imshow('Lighting Filter Map', smooth_gain)
        cv2.waitKey(0)
        cv2.destroyWindow("Lighting Filter Map")
        print(50 * '=','Filter info', 50* '=')
        print("reference image shape: ", refrence_image.shape)
        print("target image shape: ", target_image.shape)
        print("filter shape: ", smooth_gain.shape)
        print("min value: ", smooth_gain.min())
        print("max value: ", smooth_gain.max())

    def test_apply_smooth_gain(self):
        refrence_image: np.ndarray = cv2.imread("/home/abolfazl/Documents/DPR/obama.jpg")
        target_image: np.ndarray = cv2.imread("candidiate_config/candidate1/result/rotate_light_06.jpg")
        smooth_gain, _ = create_lighting_filter(img_x=refrence_image, img_y=target_image)
        cv2.imshow('Lighting Filter Map', smooth_gain)
        cv2.waitKey(0)
        cv2.destroyWindow("Lighting Filter Map")
        result = apply_lighting_filter(refrence_image, smooth_gain)
        cv2.imshow('Result', result)
        cv2.waitKey(0)
        cv2.destroyWindow("Result")
        cv2.imwrite(f"apply-gain.jpg", result)
        
    def test_shadow_enhancement(self):
        shadow_strength=0.9
        refrence_image: np.ndarray = cv2.imread("/home/abolfazl/Documents/DPR/obama.jpg")
        target_image: np.ndarray = cv2.imread(
            "candidiate_config/candidate1/result/rotate_light_06.jpg")
        smooth_gain, lab_x = create_lighting_filter(img_x=refrence_image, img_y=target_image)
        _, final_gain = apply_shadow_enhancement(lab_x, smooth_gain, shadow_strength=shadow_strength)

        cv2.imshow('Lighting Filter Map', final_gain)
        cv2.waitKey(0)
        cv2.destroyWindow("Lighting Filter Map")
        print(50 * '=', 'Filter info', 50 * '=')
        print("reference image shape: ", refrence_image.shape)
        print("target image shape: ", target_image.shape)
        print("filter shape: ", final_gain.shape)
        print("min value: ", final_gain.min())
        print("max value: ", final_gain.max())

    def test_apply_lighting_filter(self):
        shadow_strength=0.9
        refrence_image: np.ndarray = cv2.imread("/home/abolfazl/Documents/DPR/obama.jpg")
        target_image: np.ndarray = cv2.imread("candidiate_config/candidate1/result/rotate_light_06.jpg")
        smooth_gain, lab_x = create_lighting_filter(img_x=refrence_image, img_y=target_image)
        _, final_gain = apply_shadow_enhancement(lab_x, smooth_gain, shadow_strength=shadow_strength)


        result = apply_lighting_filter(target_image, final_gain)
        filter_vis = cv2.normalize(final_gain, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imshow('Lighting Filter Map', filter_vis)
        cv2.waitKey(0)

        cv2.imwrite(f"/home/abolfazl/Documents/DPR/res_darker_{shadow_strength}.jpg", result)
        cv2.imshow("result_img", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def test_darkening_from_gain(self):
        refrence_image: np.ndarray = cv2.imread("/home/abolfazl/Documents/DPR/obama.jpg")
        target_image: np.ndarray = cv2.imread("candidiate_config/candidate1/result/rotate_light_06.jpg")
        smooth_gain, lab_x = create_lighting_filter(img_x=refrence_image, img_y=target_image)
        cv2.imshow("filter", smooth_gain)
        cv2.waitKey(0)

        final_gain = apply_darkening_from_gain(smooth_gain=smooth_gain, shadow_strength=1.5, curve_power=1.5)
        result = apply_lighting_filter(target_img=refrence_image, gain_filter=final_gain)
        cv2.imshow("dark filter", final_gain)
        cv2.imshow("result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    unittest.main()
