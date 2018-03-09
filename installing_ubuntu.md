High-level instructions:

https://askubuntu.com/questions/767269/how-can-i-install-cudnn-on-ubuntu-16-04

### Install CUDA

Source: http://developer.download.nvidia.com/compute/cuda/7.5/Prod/docs/sidebar/CUDA_Installation_Guide_Linux.pdf

```
sudo apt-get install gcc
```

Download on Ubuntu using distribution-specific package.

Source: https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1604&target_type=debnetwork

**Be sure to install version 8 -- Tensorflow does not support version 9!.**

```
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb

sudo dpkg -i cuda-repo-ubuntu1604_8.0.61-1_amd64.deb

sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub

sudo apt-get update
```

### Install cuDNN

Check installed CUDA version to ensure version 8 is installed.

```
> cat /usr/local/cuda/version.txt
CUDA Version 9.0.176
```

Download the appropriate cuDNN package from [this page](https://developer.nvidia.com/rdp/cudnn-download). Be sure to download **both the runtime and developer packages**.

https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v7.0.3/prod/9.0_20170926/Ubuntu16_04-x64/libcudnn7-dev_7.0.3.11-1+cuda9.0_amd64-deb

`scp` both the runtime and developer `.deb` packages to your remote GPU instance. Once they're copied over, run:

```
sudo dpkg -i libcudnn7.0.3.11-1+cuda9.0_amd64.deb
sudo dpkg -i libcudnn7-dev_7.0.3.11-1+cuda9.0_amd64.deb
```


### Install Tensorflow

Ensure Python3 and pip are installed:

```
sudo apt install python3 python3-pip
pip3 install --upgrade pip
```

Simply run:

```
# Tensorflow needs this
sudo apt-get install libcupti-dev

# Be sure to use the "-gpu" version to enable GPU support.
pip3 install --upgrade tensorflow-gpu
```

### Running Tensorflow Scripts

Tensorflow uses CUDA libraries when running. This means you need to add those libraries to `LD_LIBRARY_PATH` whenever you run Tensorflow code:

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/
```





FOR WINDOWS:

https://www.tensorflow.org/install/install_windows
