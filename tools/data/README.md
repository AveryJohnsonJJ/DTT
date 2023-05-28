# Things you need to know about DTT data format

## The format of the pickle files

Each pickle file corresponds to an action recognition dataset. The content of a pickle file is a dictionary with two fields: `split` and `annotations`

1. Split: The value of the `split` field is a dictionary: the keys are the split names, while the values are lists of video identifiers that belong to the specific clip.
2. Annotations: The value of the `annotations` field is a list of skeleton annotations, each skeleton annotation is a dictionary, containing the following fields:
   1. `frame_dir` (str): The identifier of the corresponding video.
   2. `total_frames` (int): The number of frames in this video.
   3. `img_shape` (tuple[int]): The shape of a video frame, a tuple with two elements, in the format of (height, width). Only required for 2D skeletons.
   4. `original_shape` (tuple[int]): Same as `img_shape`.
   5. `label` (int): The action label.
   6. `keypoint` (np.ndarray, with shape [M x T x V x C]): The keypoint annotation. M: number of persons; T: number of frames (same as `total_frames`); V: number of keypoints (25 for NTURGB+D 3D skeleton, 17 for CoCo, 18 for OpenPose, etc. ); C: number of dimensions for keypoint coordinates (C=2 for 2D keypoint, C=3 for 3D keypoint).
   7. `keypoint_score` (np.ndarray, with shape [M x T x V]): The confidence score of keypoints. Only required for 2D skeletons.

You can download an annotation file and browse it to get familiar with our annotation formats.

## Download the pre-processed skeletons

We provide links to the pre-processed skeleton annotations, you can directly download them and use them for training & testing.

- NTU-15

For Kinetics400, since the skeleton annotations are large, we do not provide the direct download links on aliyun. Please use the following link to download the `kpfiles` and extract it under `$PYSKL/data/k400` for Kinetics-400 training & testing: https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EeyDCVskqLtClMVVwqD53acBF2FEwkctp3vtRbkLfnKSTw?e=B3SZlM



### BibTex items for each provided dataset

```BibTex
% NTURGB+D
@inproceedings{shahroudy2016ntu,
  title={Ntu rgb+ d: A large scale dataset for 3d human activity analysis},
  author={Shahroudy, Amir and Liu, Jun and Ng, Tian-Tsong and Wang, Gang},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={1010--1019},
  year={2016}
}
% NTURGB+D 120
@article{liu2019ntu,
  title={Ntu rgb+ d 120: A large-scale benchmark for 3d human activity understanding},
  author={Liu, Jun and Shahroudy, Amir and Perez, Mauricio and Wang, Gang and Duan, Ling-Yu and Kot, Alex C},
  journal={IEEE transactions on pattern analysis and machine intelligence},
  volume={42},
  number={10},
  pages={2684--2701},
  year={2019},
  publisher={IEEE}
}
% Kinetics-400
@inproceedings{carreira2017quo,
  title={Quo vadis, action recognition? a new model and the kinetics dataset},
  author={Carreira, Joao and Zisserman, Andrew},
  booktitle={proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={6299--6308},
  year={2017}
}
% UCF101
@article{soomro2012ucf101,
  title={UCF101: A dataset of 101 human actions classes from videos in the wild},
  author={Soomro, Khurram and Zamir, Amir Roshan and Shah, Mubarak},
  journal={arXiv preprint arXiv:1212.0402},
  year={2012}
}
% HMDB51
@inproceedings{kuehne2011hmdb,
  title={HMDB: a large video database for human motion recognition},
  author={Kuehne, Hildegard and Jhuang, Hueihan and Garrote, Est{\'\i}baliz and Poggio, Tomaso and Serre, Thomas},
  booktitle={2011 International conference on computer vision},
  pages={2556--2563},
  year={2011},
  organization={IEEE}
}

```
