
This project hosts the official implementation for the paper: 

**Deep Dive into Gradients: Better Optimization for 3D Object Detection with Gradient-Corrected IoU Supervision** 
<!-- [[PDF](https://ming71.github.io/Files/papers/TIOE.pdf)][[BibTex](https://ming71.github.io/Files/BibTeX/TIOEDet.html)]

( accepted by **CVPR 2023**).  -->




### Setup

```
pip install spconv-cu111
pip install Cmake
pip install -r requirement.txt
pip install mayavi
python setup.py develop

cd pcdet/ops/iou3d/cuda_op
python setup.py install
```


### Training
* Data Prepare
Download KITTI and organize it into the following form:
├── data
│   ├── kitti
│   │   │── ImageSets
│   │   │── training
│   │   │   ├──calib & velodyne & label_2 & image_2 & (optional: planes)
│   │   │── testing
│   │   │   ├──calib & velodyne & image_2

* Generatedata infos:
`python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml`
* Creat `.yaml` file
* Run `sh train.sh`



## Visualizations

![demo](./docs/demo.png) 




Feel free to contact [me](chaser.ming@gmail.com)  if you have any questions.

