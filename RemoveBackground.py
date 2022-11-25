import copy

import numpy as np
from mmdet.datasets import PIPELINES


@PIPELINES.register_module()
class RemoveBackground:
    """Remove backgrounds of bboxes.

    Args:
        p (float, optional): Probability of shifts. Default 0.5.

    Note:
        - The keys for bboxes, labels and masks must be aligned. That is,
          `gt_bboxes` corresponds to `gt_labels` and `gt_masks`, and
          `gt_bboxes_ignore` corresponds to `gt_labels_ignore` and
          `gt_masks_ignore`.
        - If the crop does not contain any gt-bbox region, skip this image.
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def _crop_boxes(self, results: dict):
        """Function to randomly crop bounding boxes from images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            results (dict): Result dict having randomly cropped bounding box images.
        """
        results = copy.deepcopy(results)
        bboxes = results["gt_bboxes"]
        img = results["img"]

        bboxes_int = bboxes.astype("int")  # casting (float -> int) for indexing
        overlap_cnt = np.zeros_like(img)  # save number of overlap by pixel

        for bbox in bboxes_int:
            x_min, y_min, x_max, y_max = bbox
            cnt = np.ones_like(img)
            overlap_cnt[y_min:y_max, x_min:x_max, :] += cnt[y_min:y_max, x_min:x_max, :]
        overlap = np.where(overlap_cnt > 0.0, 1.0, 0.0)
        final_img = img * overlap  # bbox cropping
        results["img"] = final_img
        return results

    def __call__(self, results: dict):
        """Call function to randomly remove background of bounding boxes.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            results (dict): Result dict having randomly cropped bounding box images.
        """
        rand_p = np.random.random(1)[0]
        if rand_p <= self.p:
            results = self._crop_boxes(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(random probability={self.p})"
        return repr_str


# Example
# test = RemoveBackground(p=0)
# random_remove_results = test(results)
# plt.imshow(random_remove_results['img'], interpolation='bicubic')
# plt.show()

# For using mmdetection pipeline
# 1. Go to 'mmdetection/mmdet/datasets/pipelines/transforms.py' and paste above code blocks.
# 2. Go to 'mmdetection/mmdet/datasets/pipelines/__init__.py'
# 2-1. Import RemoveBackground from transforms .
# 2-2. Add RemoveBackground into __all__
# 3. Add 'dict(type="RemoveBackground", p=0.5)' into your train_pipeline.

# Test Pipeline
# 'python tools/misc/browse_dataset.py ../my_test/ssd.py --output-dir custom'
