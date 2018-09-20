Real-time 'Actor-Critic' tracking
=========================================
Code for [Real-time 'Actor-Critic' tracking](https://drive.google.com/file/d/18N0G1vX148SQWBuvG5sdAXjlSJH3yUua/view) accepted by ECCV 2018

Introduction
--------------------------------
We propose a novel tracking algorithm with real-time performance based on the ‘Actor-Critic’ framework.</br>
![](https://github.com/bychen515/ACT/blob/master/ACT.png)  

Requirements
--------------------------
1. Tensorflow 1.4.0 (Train) and Pytorch 0.3.0 (Test)
2. CUDA 8.0 and cuDNN 6.0
3. Python 2.7

Usage
--------------------------
### Train
  1. Please download the `ILSVRC VID dataset`, and put the `VID` folder into `$(ACT_root)/train/` </br>
  (We adopt the same videos as [meta_trackers](https://github.com/silverbottlep/meta_trackers). You can find more details in `ilsvrc_train.json`.)
  2. Run the `$(ACT_root)/train/DDPG_train.py` for train the 'Actor and Critic' network.
### Test
  Please run `$(ACT_root)/tracking/run_tracker.py` for demo.
 
Citation
--------------------
If you find ACT useful in your research, please kindly cite our paper:</br>

    @InProceedings{Chen_2018_ECCV,
    author = {Chen, Boyu and Wang, Dong and Li, Peixia and Wang, Shuang and Lu, Huchuan},
    title = {Real-time 'Actor-Critic' Tracking},
    booktitle = {The European Conference on Computer Vision (ECCV)},
    month = {September},
    year = {2018}
    }

Contact
--------------------
If you have any questions, please feel free to contact bychen@mail.dlut.edu.cn

Acknowledgments
------------------------------
Many parts of this code are adopted from other related works ([py-MDNet](https://github.com/HyeonseobNam/py-MDNet) and [meta_trackers](https://github.com/silverbottlep/meta_trackers))

