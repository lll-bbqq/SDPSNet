# SDPSNet
SDPSNet is LiDAR 3D object detection method. This repository is based off [OpenPCDet].
Sparse Dynamic Parallel Attention and Star Interaction Network for 3D Object Detection
Bingqian Lv, Changming Sun, Lei Lu, Qing Li, Lijuan Zhang
"This code is directly related to the manuscript submitted to *The Visual Computer*. If you use this code, please cite the corresponding paper."

## Model Zoo
### KITTI 3D Object Detection Baselines

Selected supported methods are shown in the below table. The results are the 3D detection performance of moderate difficulty on the val set of KITTI dataset.

1. All models are trained with 1 NVIDIA RTX A6000 GPU. and are available for download.
2. The training time is measured with 1 NVIDIA RTX A6000 GPU and PyTorch 1.10.

<table>
  <tr>
    <th rowspan="2">Method</th>
    <th rowspan="2">FPS</th>
    <th colspan="3">Car</th>
    <th colspan="3">Ped</th>
    <th colspan="3">Cyc</th>
  </tr>
  <tr>
    <th>easy</th><th>mod</th><th>hard</th>
    <th>easy</th><th>mod</th><th>hard</th>
    <th>easy</th><th>mod</th><th>hard</th>
  </tr>
  <tr>
    <td>Pvrcnn</td><td>16</td><td>89.35</td><td>79.33</td><td>78.59</td><td>59.99</td><td>52.28</td><td>48.00</td><td>84.50</td><td>71.69</td><td>65.45</td>
  </tr>
  <tr>
    <td>SDPSNet</td><td>25</td><td>89.08</td><td>79.08</td><td>78.42</td><td>61.72</td><td>53.53</td><td>48.48</td><td>85.64</td><td>72.78</td><td>69.95</td>
  </tr>
</table>

## 📥 Pretrained Model Weights

| Item | Description |
|------|-------------|
| **File path** | `output/ckpt/checkpoint_epoch_80.pth` |
| **Download** | [⬇️ Click to download](output/ckpt) |

### Verify the file exists
```bash
ls -lh output/ckpt/checkpoint_epoch_80.pth
```

### DAIR-2X-V 3D Object Detection Baselines

将DAIR-V2X-V划分为 **11163帧** 训练集和 **4464帧** 测试集

在测试集上的结果如下：

<table>
  <tr>
    <th rowspan="2">Method</th>
    <th rowspan="2">FPS</th>
    <th colspan="3">Car</th>
    <th colspan="3">Ped</th>
    <th colspan="3">Cyc</th>
  </tr>
  <tr>
    <th>easy</th><th>mod</th><th>hard</th>
    <th>easy</th><th>mod</th><th>hard</th>
    <th>easy</th><th>mod</th><th>hard</th>
  </tr>
  <tr>
    <td>Pvrcnn</td><td>19</td><td>69.57</td><td>60.20</td><td>57.87</td><td>43.13</td><td>41.52</td><td>41.16</td><td>41.12</td><td>39.90</td><td>37.53</td>
  </tr>
  <tr>
    <td>SDPSNet</td><td>32</td><td>69.66</td><td>60.24</td><td>58.26</td><td>44.64</td><td>41.43</td><td>40.95</td><td>43.25</td><td>40.85</td><td>38.24</td>
  </tr>
</table>

## 📥 Pretrained Model Weights

| Item          | Description                           |
| ------------- | ------------------------------------- |
| **File path** | `output/ckpt/checkpoint_epoch_40.pth` |
| **Download**  | [⬇️ Click to download](output/ckpt)    |

## Training Commands & Parameters

### Environment Setup

All required packages are listed in `requirements.txt`.

```bash
# Clone repository
git clone https://github.com/lll-bbqq/SDPSNet.git
cd SDPSNet

# Create Conda Environment
conda create -n sdpsnet python=3.8 -y
conda activate sdpsnet

# Install dependencies
pip install -r requirements.txt

# Training
python train.py --cfg_file cfgs/kitti_models/SDPSNet.yaml --extra_tag sdpsnet

# Test(kitti)
python test.py --cfg_file cfgs/kitti_models/SDPSNet.yaml --ckpt ../output/ckpt/checkpoint_epoch_80.pth  --save_to_file --extra_tag sdpsnet
# Test(kitti)
python test.py --cfg_file cfgs/kitti_models/SDPSNet.yaml --ckpt ../output/ckpt/checkpoint_epoch_40.pth  --save_to_file --extra_tag sdpsnet
```


## Acknowledgement
We would like to thank the authors of OpenPCDet for their open source release of their codebase.

