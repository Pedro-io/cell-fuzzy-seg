# Dataset loading utilities

This folder contains the dataset-loading components used by the segmentation pipeline. These scripts provide a common interface for reading images and masks, validating dataset configuration, and optionally integrating Cellpose-based segmentation outputs.

## Files

### base_dataset.py
This module defines the abstract BaseDataset class. It loads dataset settings from the YAML configuration file, validates the required fields, and provides helper methods for accessing image and mask directories. All concrete dataset implementations in this project inherit from this class.

### monuseg_dataset.py
This module implements MonusegDataset, a concrete dataset loader for the MoNuSeg data format. It discovers matching image and mask files, loads them from disk, and supports several mask formats, including XML, NumPy, and standard image-based masks. Each sample returned by the dataset includes the image, the ground-truth mask, and metadata such as the file paths.

### monuseg_cellpose_dataset.py
This module implements MonusegCellposeDataset, which extends the base MoNuSeg dataset with Cellpose inference. For each sample, it generates a segmentation mask using Cellpose and adds it to the sample as an additional field. The file also includes helper functions for preparing images and creating collated batches that combine the original image, the Cellpose segmentation, and distance-map targets.

## Notes
- Dataset paths and file extensions are configured through the YAML file in the project configuration folder.
- The dataset classes assume that each image has a corresponding mask with the same base filename.
- The Cellpose-based wrapper is mainly used when the pipeline needs both the ground-truth mask and a model-generated segmentation mask during training or evaluation.
