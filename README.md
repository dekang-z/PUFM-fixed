# PUFM-fixed
PUFM 项目的修复版，解决原代码运行 bug，对原作者中给出的复现步骤有所优化，可以调通。
**Efficient Point Cloud Upsampling via Flow Matching**

[arXiv](https://arxiv.org/abs/2501.15286)

This is the official PyTorch implementation of our paper **"Efficient Point Cloud Upsampling via Flow Matching"**.

> **Note:** This repository has been fixed and verified to run out-of-the-box on **Python 3.9 + PyTorch 1.13 + CUDA 11.6**. See [Known Fixes](#known-fixes) for details on bugs patched in the original release.


## Abstract

Diffusion models are a powerful framework for tackling ill-posed problems, with recent advancements extending their use to point cloud upsampling. Despite their potential, existing diffusion models struggle with inefficiencies as they map Gaussian noise to real point clouds, overlooking the geometric information inherent in sparse point clouds. To address these inefficiencies, we propose PUFM, a flow matching approach to directly map sparse point clouds to their high-fidelity dense counterparts. Our method first employs midpoint interpolation to sparse point clouds, resolving the density mismatch between sparse and dense point clouds. Since point clouds are unordered representations, we introduce a pre-alignment method based on Earth Mover's Distance (EMD) optimization to ensure coherent interpolation between sparse and dense point clouds, which enables a more stable learning path in flow matching. Experiments on synthetic datasets demonstrate that our method delivers superior upsampling quality but with fewer sampling steps. Further experiments on ScanNet and KITTI also show that our approach generalizes well on RGB-D point clouds and LiDAR point clouds, making it more practical for real-world applications.


---

## Installation

### Environment

**Recommended:** Python 3.9 + PyTorch 1.13 (CUDA 11.6)

```bash
# Create conda environment
conda create -n pufm python=3.9
conda activate pufm

# Install PyTorch with CUDA 11.6
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

# Install dependencies
pip install numpy==1.25.2 open3d==0.17.0 einops==0.3.2 scikit-learn==1.3.1 tqdm==4.62.3 h5py==3.6.0 plyfile
```

### CUDA Extension Setup

```bash
cd models/pointops
python setup.py install
cd ../../Chamfer3D
python setup.py install
cd ../emd_assignment
python setup.py install
cd ../..
```

> **Tip:** If `import pointops_cuda` fails with `libc10.so not found`, set the library path:
> ```bash
> export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/torch/lib:$LD_LIBRARY_PATH
> ```
> To make it permanent, add the above line to your `~/.bashrc` or conda activate script.

---

## Data Preparation

Please download [PU1K](https://github.com/guochengqian/PU-GCN) and [PUGAN](https://github.com/liruihui/PU-GAN).

Generate test data using `dataset/prepare_data.py`:

```bash
cd dataset

# Generate PUGAN 4x test set (2048 -> 8192)
python prepare_data.py --input_pts_num 2048 --R 4 --mesh_dir ../PUGAN/test --save_dir ../PUGAN/input_2048_4X/test

cd ..
```

Generated structure:
```
PUGAN/input_2048_4X/test/
└── input_2048_4X/
    ├── input_2048/     # Low-res point clouds (2048 pts)
    └── gt_8192/        # Ground truth (8192 pts)
```

For more information, please refer to [Grad-PU](https://github.com/yunhe20/Grad-PU).

---

## Quick Start

Pretrained models are provided in the `pretrained_model/` folder.

### 1. Single Point Cloud Upsampling (4x)

```bash
python test_pufm.py --model pufm --test_input_path example/camel.xyz --up_rate 4
```

### 2. Arbitrary Upsampling (11x)

> **Note:** `test_pufm_arbitrary.py` accepts a **single file path**, not a directory.

```bash
python test_pufm_arbitrary.py     --model pufm_w_attn     --dataset pugan     --test_input_path ./PUGAN/input_2048_4X/test/input_2048_4X/input_2048/camel.xyz     --up_rate 11
```

### 3. Batch Evaluation on PUGAN (4x)

**Using PUFM:**
```bash
python eval_pufm.py     --model pufm     --dataset pugan     --test_input_path ./PUGAN/input_2048_4X/test/input_2048_4X/input_2048/     --test_gt_path ./PUGAN/input_2048_4X/test/input_2048_4X/gt_8192/     --up_rate 4     --ckpt_folder pretrained_model     --save_dir ./output/eval_4x
```

**Using PUFM_w_attn:**
```bash
python eval_pufm.py     --model pufm_w_attn     --dataset pugan     --test_input_path ./PUGAN/input_2048_4X/test/input_2048_4X/input_2048/     --test_gt_path ./PUGAN/input_2048_4X/test/input_2048_4X/gt_8192/     --up_rate 4     --ckpt_folder pretrained_model     --save_dir ./output/eval_4x_attn
```

### 4. Batch Evaluation (16x)

> **Note:** 16x evaluation requires 16x ground truth data (32768 pts). If you only have 4x data, regenerate with `--R 16` first.

```bash
# Generate 16x data
cd dataset
python prepare_data.py --input_pts_num 2048 --R 16 --mesh_dir ../PUGAN/test --save_dir ../PUGAN/input_2048_16X/test
cd ..

# Evaluate
python eval_pufm.py     --model pufm_w_attn     --dataset pugan     --test_input_path ./PUGAN/input_2048_16X/test/input_2048_16X/input_2048/     --test_gt_path ./PUGAN/input_2048_16X/test/input_2048_16X/gt_32768/     --up_rate 16     --ckpt_folder pretrained_model     --save_dir ./output/eval_16x
```

---

## Training

### Train on PU1K

```bash
python train_pufm.py --dataset pu1k
```

> **Note:** If you encounter `AttributeError: 'Namespace' object has no attribute 'up_rate'`, ensure you are using the fixed version of `train_pufm.py` in this repo. The original release had a missing parameter initialization bug.

---

## Known Fixes

This repository includes the following fixes compared to the original release:

| # | File | Issue | Fix |
|---|------|-------|-----|
| 1 | `test_pufm.py` | `sys.modules['pointops_cuda'] = None` forced JIT compilation, causing `RuntimeError: Ninja is required` | Removed the line; uses pre-compiled extension instead |
| 2 | `test_pufm.py` | `torch.cuda.is_available = lambda: False` forced CPU mode | Removed the line |
| 3 | `test_pufm_arbitrary.py` | Second `pcd_upsample` call lost batch dimension (`Expected 3 dims, got 2`) | Added `.unsqueeze(0)` before second call |
| 4 | `eval_pufm.py` | `parse_pc_args()` mistakenly parsed evaluation args, showing training help | Saved/restored `sys.argv` before calling `parse_pc_args()` |
| 5 | `train_pufm.py` | `model_args` missing `up_rate` and `num_points` attributes | Added fallback defaults after `reset_model_args()` |
| 6 | `prepare_data.py` | `args.noise_level` and `args.noise_type` used but not defined in `argparse` | Added the two missing arguments |
| 7 | `emd_assignment/` | Internal `import emd` failed because module is named `emd_assignment` | Changed to `import emd_assignment as emd` |
| 8 | Runtime | `libc10.so` / `libtorch_cpu.so` not found when importing compiled extensions | Set `LD_LIBRARY_PATH` to PyTorch lib directory |

---

## Acknowledgments

Our code is built upon the following repositories: [PU-GCN](https://github.com/guochengqian/PU-GCN), [PU-GAN](https://github.com/liruihui/PU-GAN) and [Grad-PU](https://github.com/yunhe20/Grad-PU). Thanks for their great work.

---

## Citation

If you find our project useful, please consider citing us:

```bibtex
@InProceedings{ZSLiu_2025,
    author    = {Zhi-Song Liu and Chenhang He and Lei Li},
    title     = {Efficient Point Clouds Upsampling via Flow Matching},
    booktitle = {arXiv:2501.15286},
    year      = {2025}
}
```
put_2048_16X/gt_32768/     --up_rate 16     --ckpt_folder pretrained_model     --save_dir ./output/eval_16x_attn
```
