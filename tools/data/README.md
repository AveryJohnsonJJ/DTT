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
   6. `keypoint` (np.ndarray, with shape [M x T x V x C]): The keypoint annotation. M: number of persons; T: number of frames (same as `total_frames`); V: number of keypoints (25 for NTURGB+D 3D skeleton); C: number of dimensions for keypoint coordinates.
   7. `keypoint_score` (np.ndarray, with shape [M x T x V]): The confidence score of keypoints.

You can download an annotation file and browse it to get familiar with our annotation formats.

## Introduction to the Dataset

We provide links to the pre-processed skeleton annotations, you can directly download them and use them for training & testing.[Download](https://drive.google.com/drive/folders/1KExekOP4OPZLkJykNRXV0pek0M6Jx_f7?usp=sharing)

### NTU-15
**NTU-15** is a dataset composed of 15 behaviors selected from **NTU-RGB+D 120** that conform to dangerous campus behaviors.Among the 120 types of behaviors, we identify 15 that meet the criteria for dangerous behaviors and divide them according to the cross-subject mode. A total of 8,621 sequences are selected for the training set, while 5,632 sequences are reserved for the test set.
| label | 42 | 43                     | 44                              | 45            | 46            | 47                    | 49                       | 50                 | 51                 | 56                         | 105                        | 106                                   | 107                                       | 108                   | 109                              |
|-------|----|------------------------|---------------------------------|---------------|---------------|-----------------------|--------------------------|--------------------|--------------------|----------------------------|----------------------------|---------------------------------------|-------------------------------------------|-----------------------|----------------------------------|
| name  | falling | touch head (headache) | touch chest (stomachache/heart pain) | touch back (backache) | touch neck (neckache) | nausea or vomiting condition | punching/slapping other person | kicking other person | pushing other person | touch other person's pocket | hit other person with something | wield knife towards other person | knock over other person (hit with body) | grab other person's stuff | shoot at other person with a gun |

For the processing of NTU-15, first you need to download the processed **NTU-RGB+D 120** 3D dataset from [PYSKL](https://github.com/kennymckormick/pyskl/tree/main/tools/data).Then use the following command to extract dangerous behaviors from the dataset.
```
python split_dangerourslist.py
```

### Anomaly action-18
**Anomaly action-18** is a dataset composed of **NTU-15** and other RGB datasets, such as **Kinetics**, **UCF101** and **HMDB51**. Compared with **NTU-15**, these RGB datasets mostly from YouTube have poorer video quality, but the number of videos and types of behaviors are richer. Similarly, we select videos that conform to the dangerous behaviors and add them to the dataset.

The table below shows the selected types of behaviors in the RGB dataset.
#### HMDB51

| label | name          |
|-------|---------------|
| 12    | fall floor    |
| 16    | handstand     |
| 17    | hit           |
| 27    | punch         |
| 35    | shoot bow     |
| 36    | shoot gun     |

#### UCF101

| label | name                 |
|-------|----------------------|
| 37    | handstandWalking     |
| 70    | punch                |
| 73    | RockClimbingIndoor   |
| 74    | RopeClimbing         |

#### Kinetics400

| label | name            |
|-------|-----------------|
| 105   | drop kicking    |
| 278   | rock climbing   |
| 314   | slapping        |
| 396   | wrestling       |

For the processing of Anomaly action-18, first you need download **Kinetics**, **UCF101** and **HMDB51** dataset from [MMaction2](https://github.com/open-mmlab/mmaction2/tree/main/tools/data).Then use scripts to extract data from videos of dangerous behaviors in the dataset. Finally, save the extracted dataset with **NTU-15**.


scripts:comming soon

#### Anomaly action-18
| label | 0             | 1             | 2            | 3             | 4             | 5          | 6           | 7       | 8          | 9             | 10           | 11          | 12          | 13         | 14    | 15          | 16    | 17         |
|-------|---------------|---------------|--------------|---------------|---------------|------------|-------------|---------|------------|---------------|-------------|-------------|-------------|------------|-------|-------------|-------|------------|
| name  | falling       | touch head    | touch chest  | touch back    | touch neck    | vomiting   | punch/slap  | kicking | push       | touch pocket  | handstand   | climbing   | wrestling  | shoot bow  | hit   | knock over  | grab  | shoot gun  |


### Open Environment-12
**Open Environment-12** is a dataset consisting of over 100 videos in the wild without constraint. Different from the existing dataset, these actions focus on dangerous actions outdoors with multiple persons. The videos in this dataset feature at least two individuals primarily performing in an open environment.

| label | 42                                      | 43                     | 44                               | 45                        | 46                        | 47                    | 49                             | 50                      | 51                      | 56                          | 105                                 | 107                                       |
|-------|-----------------------------------------|------------------------|----------------------------------|---------------------------|---------------------------|-----------------------|--------------------------------|-------------------------|-------------------------|-----------------------------|------------------------------------|-------------------------------------------|
| name  | falling                                 | touch head (headache)  | touch chest (stomachache/heart pain) | touch back (backache)    | touch neck (neckache)    | nausea or vomiting condition | punching/slapping other person | kicking other person     | pushing other person     | touch other person's pocket | hit other person with something    | knock over other person (hit with body) |

We provide the original video data, processing scripts, as well as the division method for dataset training and testing.

```shell
bash tools/dist_run.sh tools/data/custom_2d_skeleton.py 1 --video-list tools/data/video_list.txt --out ourvideo.pkl
```

### BibTex items for dataset

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
