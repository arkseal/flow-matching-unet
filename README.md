# Flow Matching Unet Implementation

This repository contains a very simple PyTorch implementation of a Flow Matching Unet model for generative tasks, based of the paper [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747).

# TODO
- [x] Add gif generation script for results visualization
    - [x] Add evaluate mode
- [x] Add resume training functionality
- [ ] Implement main.py CLI
    - [x] Add argument parsing
    - [x] Add arguments for hyperparameters
    - [ ] Add logging functionality
- [ ] Add more image dataset support
    - [ ] CIFAR-10
    - [ ] ImageNet (not important for now)
- [ ] Implement Audio implementation
- [ ] Add evaluation metrics
- [ ] Add pre-trained model checkpoints

## Getting Started

Clone the repository and set up the python environment (Python 3.14.2 was used).

```bash
git clone https://github.com/arkseal/flow-matching-unet.git
cd flow-matching-unet
```

Create a virtual environment and install the required packages:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Datasets

This implementation used MINIST dataset for training and evaluation. The code will be modified to support other datasets in the future.

## Training
To train the model, run the following command:

```bash
python main.py --train
```

## Generation
To generate samples from the model, run the following command:

```bash
python main.py --generate
```

## References

- [1] Lipman, Yaron, et al. "Flow Matching for Generative Modeling." [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)
- [2] Ronneberger, Olaf et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation." [arXiv:1505.04597](https://arxiv.org/pdf/1505.04597)
- [3] Janupalli, Pranay. "Understanding Sinusoidal Positional Encoding in Transformers." [Medium](https://medium.com/@pranay.janupalli/understanding-sinusoidal-positional-encoding-in-transformers-26c4c161b7cc)
- [4] [keishihara/flow-matching](https://github.com/keishihara/flow-matching)