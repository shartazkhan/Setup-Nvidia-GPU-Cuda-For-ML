# Setup NVIDIA GPU Cuda for Machine Learning

This guide will help you set up your NVIDIA GPU for deep learning by installing the necessary drivers, libraries, and frameworks.

## Prerequisites

Ensure you have an NVIDIA GPU that supports CUDA.

---

## Step 1: Install NVIDIA GPU Driver

Download and install the latest NVIDIA driver for your GPU from the official website:

- [NVIDIA GPU Driver Download](https://www.nvidia.com/Download/index.aspx)

---

## Step 2: Install Visual Studio with C++

CUDA requires Visual Studio with C++ support. Download and install Visual Studio Community Edition and make sure to select C++ components during installation.

- [Visual Studio Community Edition](https://visualstudio.microsoft.com/vs/community/)

---

## Step 3: Install Anaconda or Miniconda

Anaconda is a package manager for Python that simplifies the installation of deep learning frameworks.

- [Download Anaconda](https://www.anaconda.com/download/success)

---

## Step 4: Install CUDA Toolkit

Download and install the appropriate CUDA Toolkit version that is compatible with your deep learning framework.

- [Download CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive)

---

## Step 5: Install cuDNN

Download and install cuDNN, which is required for deep learning frameworks to use CUDA efficiently.

- [Download cuDNN](https://developer.nvidia.com/rdp/cudnn-archive)

---

## Step 6.1: Install PyTorch

Install PyTorch with CUDA support by following the instructions on the official website.

- [Install PyTorch](https://pytorch.org/get-started/locally/)

---

## Step 6.2: Install TensorFlow

Install TensorFlow with GPU support by following the instructions on the official website.

- [Install TensorFlow](https://www.tensorflow.org/install)

---

## Step 7: Verify Cuda is working
Run the following Python script to check if your GPU is properly set up for deep learning:

For PyTorch
```python
import torch

print("Number of GPUs:", torch.cuda.device_count())
print("GPU Name:", torch.cuda.get_device_name(0))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
```
For TensorFlow
```python
import tensorflow as tf

print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    print("GPU Name:", gpus[0])
    tf.config.experimental.set_memory_growth(gpus[0], True)
```
---

## Kudos! 

If everything is set up correctly, you should see details about your GPU in the output. Now, you're ready to start training deep learning models with GPU acceleration!



