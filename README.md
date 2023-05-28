# Dual Temporal Transformers for Fine-Grained Dangerous Action Recognition

DTT is a toolbox focusing on action recognition based on **SK**e**L**eton data with **PY**Torch. Various algorithms will be supported for skeleton-based action recognition. We build this project based on the OpenSource Project [MMAction2](https://github.com/open-mmlab/mmaction2).

## Installation
```shell
git clone 
cd dtt
# Please first install pytorch according to instructions on the official website: https://pytorch.org/get-started/locally/. Please use pytorch with version smaller than 1.11.0 and larger (or equal) than 1.5.0
# The following command will install mmcv-full 1.5.0 from source, which might be very slow (take ~10 minutes). You can also follow the instruction at https://github.com/open-mmlab/mmcv to install mmcv-full from pre-built wheels, which will be much faster.
pip install -r requirements.txt
pip install -e .
```

## Demo

Check [demo.md](/demo/demo.md).

## Data Preparation
see

## Training & Testing

You can use following commands for training and testing. Basically, we support distributed training on a single server with multiple GPUs.
```shell
# Training
bash tools/dist_train.sh {config_name} {num_gpus} {other_options}
# Testing
bash tools/dist_test.sh {config_name} {checkpoint} {num_gpus} --out {output_file} --eval top_k_accuracy mean_class_accuracy
```
For specific examples, please go to the README for each specific algorithm we supported.


## Contributing

DTT

## Contact

For any questions, feel free to contact: jiebaoxd@gmail.com
