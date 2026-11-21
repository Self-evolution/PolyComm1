"""Task heads for different experiment types."""

from src.task_heads.classification import ClassificationHead
from src.task_heads.segmentation import SegmentationHead
from src.task_heads.video_classification import VideoClassificationHead

__all__ = ["ClassificationHead", "SegmentationHead", "VideoClassificationHead"]
