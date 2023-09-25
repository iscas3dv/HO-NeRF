# Novel-view Synthesis and Pose Estimation for Hand-Object Interaction from SparseViews
## Official PyTorch implementation of the ICCV '23 paper ##

[Project page](https://iscas3dv.github.io/HO-NeRF) | [Paper](https://arxiv.org/pdf/2308.11198)

## Table of contents
-----
  * [Installation](#installation)
  * [Preparing](#preparing)
  * [Training and Testing](#usage)
  * [Evaluation](#eval)
  * [Citation](#citation)
  * [License](#license)
------

## Installation ##
PyTorch with CUDA support are required. Our code is tested on python 3.9, torch 1.13.0, CUDA 11.6, and RTX 3090.
Clone the repository and install the dependencies:
```
git clone https://github.com/Qwentian/honerf.git
cd honerf
conda create -n honerf python=3.9 -y
conda activate honerf
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```
We also need to install pytorch3d from [here](https://github.com/facebookresearch/pytorch3d).

## Preparing ##
### Preparing Datasets ###

Download the [HandObject](https://pan.baidu.com/s/1nJE1Jfmo7wwqS7SNqthNrQ?pwd=8pek) dataset and name it 'data' and put it in the root directory.

### Download Pre-trained Models (Optional) ###

Download the [Pre-trained Models](https://pan.baidu.com/s/1nJE1Jfmo7wwqS7SNqthNrQ?pwd=8pek) which is named as 'exp' and put it in the root directory.

## Training and Testing ##

### Offline stage training and testing ###

Run the following command to train the offline stage object and hand models:
```bash
# Object model
python exp_runner.py --mode train --conf ./confs/wmask_realobj_bean.conf --case bean --gpu 0
# Hand model
python exp_runner.py --mode train --conf ./confs/wmask_realhand_hand1.conf --case hand1 --gpu 0
```
The results will be saved in ./exp/(CASE_NAME)/wmask_realobj(hand)/checkpoints

Run the following command to test the offline stage models to get rendering results or mesh models:
```bash
# Object model
python exp_runner.py --mode test --conf ./confs/wmask_realobj_bean.conf --case bean --gpu 0 --is_continue
# Hand model
python exp_runner.py --mode mesh --conf ./confs/wmask_realhand_hand1.conf --case hand1 --gpu 0 --is_continue
```
The results will be saved in ./exp/(CASE_NAME)/wmask_realobj(hand)/test_render(meshes)

### Online stage fitting ###

### Fitting on single frame ###

First run the following command to perform optimization based on render loss and pose regularizer loss:
```bash
python fitting_single.py --conf ./fit_confs/fit_1_8views.conf --case 1_8view --gpu 0
```
The results will be saved in ./fit_res/view_8/1/obj_name/frame_name/pose_1

Then run the following command to perform optimization based on render loss, pose regularizer loss and interaction loss :
```bash
python fitting_single.py --conf ./fit_confs/fit_12_8views.conf --case 12_8view --gpu 0
```
The results will be saved in ./fit_res/view_8/12/obj_name/frame_name/pose_12

### Fitting on video ###

Based on the above command, we obtain the results based on single frame optimization and save them in ./fit_res/view_8/12/.

(Optional 1)Then we run the following code to implement video-based optimization using stable loss:
```bash
python fitting_video.py --conf ./fit_confs/fit_123_8views_0.conf --case 123_8view_id0 --gpu 0
```
The results will be saved in ./fit_res/view_8/123/123/obj_name/frame_name/pose_4

(Optional 2)We can also run the following code to implement video-based optimization using stable loss and stable contact loss:
```bash
python fitting_video.py --conf ./fit_confs/fit_1234_8views_0.conf --case 1234_8view_id0 --gpu 0
```
The results will be saved in ./fit_res/view_8/1234/obj_name/frame_name/pose_4

## Evaluation ##

### Our results ###

You can also download the [Our results](https://pan.baidu.com/s/1nJE1Jfmo7wwqS7SNqthNrQ?pwd=8pek) named 'fit_res.zip'.

### Get more results ###
We need to obtain the hand-object interaction reconstruction results, penetration points, and rendering results to evaluate our method.
Run the following command to get the online stage results:
```bash
# Get reconstruction results for fit_type: '1'
python get_res.py --conf ./fit_confs/get_res_1.conf --case get_res_1 --gpu 0
# Get reconstruction results and penetration points for fit_type: '12'
python get_res.py --conf ./fit_confs/get_res_12.conf --case get_res_12 --gpu 0
# Get penetration points for fit_type: '123'
python get_res.py --conf ./fit_confs/get_res_123.conf --case get_res_123 --gpu 0
# Get penetration points for fit_type: '1234'
python get_res.py --conf ./fit_confs/get_res_1234.conf --case get_res_1234 --gpu 0
# Get rendering results for fit_type: '12'
python get_res.py --conf ./fit_confs/get_render_type12.conf --case render_res_type12 --gpu 0 --render True
```
The reconstruction results will be saved in ./fit_res/view_num/fit_type/obj_name/frame_name/mesh_*.

The penetration points will be saved in ./fit_res/view_num/fit_type/obj_name/frame_name/inner_*.

The rendering results will be saved in ./fit_res/view_num/fit_type/obj_name/frame_name/render_*.

### Evaluation ###

Then run the following command to get the evaluation metrics.
```bash
#Get the pose error.
python ./analys_results/analys_hand_obj_pose.py
#Get the interaction error.
python ./analys_results/analys_interaction.py
#Get the acceleration error.
python ./analys_results/analys_acc_err.py
#Get the PCI error.
python ./analys_results/analys_pci.py
# Get the rendering quality error.
python ./analys_results/analys_psnr_ssim_lpips.py
```

## Citation ##

```bash
@article{qu2023novel,
  title={Novel-view Synthesis and Pose Estimation for Hand-Object Interaction from Sparse Views},
  author={Qu, Wentian and Cui, Zhaopeng and Zhang, Yinda and Meng, Chenyu and Ma, Cuixia and Deng, Xiaoming and Wang, Hongan},
  journal={arXiv preprint arXiv:2308.11198},
  year={2023}
}
```

This codebase adapts modules from [NeuS](https://github.com/Totoro97/NeuS), [HALO](https://github.com/korrawe/halo). Please consider citing them as well.


## License ##
