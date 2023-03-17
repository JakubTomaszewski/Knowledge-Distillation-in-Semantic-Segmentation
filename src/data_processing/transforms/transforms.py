# Ref: https://mmdetection.readthedocs.io/en/v2.10.0/_modules/mmdet/datasets/pipelines/transforms.html
import torchvision.transforms.functional as F

class RandomResize:
    pass

class PhotoMetricDistortion:
    pass


class FixedHeightResize(object):
    def __init__(self, height):
        self.height = height

    def __call__(self, img):
        size = (self.height, self._calc_new_width(img))
        return F.resize(img, size)

    def _calc_new_width(self, img):
        old_width, old_height = img.size
        aspect_ratio = old_width / old_height
        return round(self.height * aspect_ratio)
