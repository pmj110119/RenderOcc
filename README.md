# RenderOcc

### [paper](https://arxiv.org/abs/2309.09502) | [video](https://www.youtube.com/watch?v=UcdXM3FNLAc)

![demo](assets/demo.gif)

## Note: Codebase under active development

We are in the process of organizing and documenting the complete codebase. Currently, the **core model components** are available in this repository, while the rest of the code is being prepared for release <u>in the coming days</u>.

# Overview

This project is a comprehensive computer vision library that provides a wide range of functionalities for building and training various types of models. The library includes a variety of modules and classes for tasks such as rendering scenes, computing loss, forward propagation, and performing operations on tensors. It also includes implementations of several popular algorithms and network architectures, such as the Adam optimization algorithm, the Feature Pyramid Network (FPN), and the CenterPoint detector. The library supports both Python and C++, and includes CUDA implementations for efficient computation on GPUs.

# Technologies and Frameworks

- Python
- C++
- CUDA
- PyTorch
- OpenMMLab
- ResNet
- ResNeXt
- SwinTransformer
- SSDVGG
- HRNet


# Installation

This guide will walk you through the process of setting up the project. Please follow the steps below:

## Step 1: Clone the Repository

First, clone the repository to your local machine using git. Open your terminal and run the following command:

```bash
git clone https://github.com/pmj110119/RenderOcc.git
```

## Step 2: Install Python Packages

This project requires several Python packages. You can install them using pip. Run the following command in your terminal:

```bash
pip install torch numpy torch_scatter torch_efficient_distloss mmdet mmcv cv2 pybind11
```

## Step 3: Install CUDA

The project requires CUDA for GPU acceleration. You can download CUDA from the official NVIDIA website. Please follow the instructions provided on the website to install CUDA.s
