## Deep Learning with Recurrent Neural Networks

General structure:

1. TODO
2. TODO
3. TODO

### General Tensorflow Learning Materials

[Tensorflow -- Getting Started](https://www.tensorflow.org/get_started/get_started)

[Tensorboard -- Getting Started](https://www.tensorflow.org/get_started/summaries_and_tensorboard)

### Install Required Libraries

[Tensorflow's Official Installation Guide](https://www.tensorflow.org/install/)

#### Windows

TODO

#### Ubuntu

TODO

#### Mac OS X

```
# Install Homebrew if you haven't already installed it.
# This is a Linux-like package manager for Macs.
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

# Install Python 3 using homebrew.
# We'll be using Python 3 for these exercises.
brew install python3

# Install Tensorflow.
pip3 install --upgrade tensorflow

# Install Tensorboard -- a web UI for visualising your Tensorflow models.
pip3 install --upgrade tensorboard
```

### Running 

Run with `TF_CPP_MIN_LOG_LEVEL=2` to disable verbose warnings:

```
TF_CPP_MIN_LOG_LEVEL=2 python3 exercises/1.py
```

Run Tensorboard in the same directory you're running the training scripts like so:

```
tensorboard --logdir log
```
