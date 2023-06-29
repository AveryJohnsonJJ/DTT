# Dual Temporal Transformers for Fine-Grained Dangerous Action Recognition

DTT is a toolbox focusing on dangerous action recognition based on SKeLeton data with PYTorch. We build this project based on the OpenSource Project [MMAction2](https://github.com/open-mmlab/mmaction2) and [PYSKL](https://github.com/kennymckormick/pyskl.git).

## Installation
```shell
git clone 
cd dtt
# Please first install pytorch according to instructions on the official website: https://pytorch.org/get-started/locally/. Please use pytorch with version smaller than 1.11.0 and larger (or equal) than 1.5.0
# The following command will install mmcv-full 1.5.0 from source, which might be very slow (take ~10 minutes). You can also follow the instruction at https://github.com/open-mmlab/mmcv to install mmcv-full from pre-built wheels, which will be much faster.
pip install -r requirements.txt
pip install -e .
```


## Datasets
We provide processed data in pkl format [Download Dataset](https://drive.google.com/drive/folders/1KExekOP4OPZLkJykNRXV0pek0M6Jx_f7?usp=sharing).

If you want to view specific information about the data format and processing methods, please Check [datasets.md](/tools/data/README.md).

### Supported Dangerous Action Datasets
- [x] NTU-15
- [x] Anomal action-18
- [x] Open environment-12



## Training & Testing

You can use following commands for training and testing. Basically, we support distributed training on a single server with multiple GPUs.
```shell
# Training
bash tools/dist_train.sh {config_name} {num_gpus} {other_options}
# Training on NTU-15 with one gpu
bash tools/dist_train.sh configs/dtt/ntu15_joint.py 1
# Training on Open environment-12 with four gpus
bash tools/dist_train.sh configs/dtt/o12_joint.py 4
# Testing
bash tools/dist_test.sh {config_name} {checkpoint} {num_gpus} --out {output_file} --eval top_k_accuracy mean_class_accuracy
```


## Contributing

We present DTT, a new visual transformer that generates a hierarchical local-to-global feature extraction technique for human behaviors. 
Our DTT retrieves the action-invariant characteristics tailored for open-scene actions successfully. On the three widely used benchmarks, NTU-15, Anomaly Action-18, and Open Environment-12, our DDT achieves state-of-the-art performance. 

## Acknowledgement

We have used codes from other great research work, including [MMAction2](https://github.com/open-mmlab/mmaction2) and [PYSKL](https://github.com/kennymckormick/pyskl.git).We sincerely thank these authors for their awesome work.

