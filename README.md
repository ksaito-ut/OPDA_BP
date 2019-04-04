# A Pytorch Implementation of [Open Set Domain Adaptation by Back-propagation](https://arxiv.org/pdf/1804.10427.pdf) (ECCV 2018)


## Introduction
Official Implementation of Open Set Domain Adaptation by Back-propagation.
We publicize code for VisDA experiment.
### Data Preparation
Follow https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification to download visda dataset.
Input path to utils/list_visda.py and run list_visda.py to generate path list file.

## Train
```
 CUDA_VISIBLE_DEVICES=$GPU_ID python trainer_osda.py \
                    --net alex
```

## Note
We reported the performance using pytorch 0.3.
Results below are obtained using pytorch 0.4 for Alexnet and VGG.
We are investigating the issues caused by the chanage in versions.

Network|bicycle  |bus  |car  |motorcycle  |train  |truck |unknown |
---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
AlexNet |54.3  |76.0  |38.6  |77.8  |71.7  |1.0  |70.6
VGG |58.4  |69.6  |50.0  |81.3  |81.2  |28.3  |91.7

## Citation
Please cite the following reference if you utilize this repository for your project.

```
@inproceedings{saito2018open,
  title={Open set domain adaptation by backpropagation},
  author={Saito, Kuniaki and Yamamoto, Shohei and Ushiku, Yoshitaka and Harada, Tatsuya},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={153--168},
  year={2018}
}
```
