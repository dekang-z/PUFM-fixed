# PUFM-Fixed: Bug-Free Reproduction of Efficient Point Cloud Upsampling via Flow Matching

This is a **bug-fixed and fully reproducible version** of the official PyTorch implementation for the paper:  
**Efficient Point Cloud Upsampling via Flow Matching** (arXiv:2501.15286) by Zhi-Song Liu, Chenhang He, and Lei Li.

> **Note:** This repository fixes all critical runtime bugs in the original release and has been verified to run out-of-the-box on **Python 3.9 + PyTorch 1.13 + CUDA 11.6**. See [Known Fixes](## Known Fixes (Patched in This Repository)) for a complete list of patched issues.

[Original Paper (arXiv)](https://arxiv.org/abs/2501.15286) | [Original Official Repository](https://github.com/Holmes-Alan/PUFM)

---

## Abstract (from the original paper)

Diffusion models are a powerful framework for tackling ill-posed problems, with recent advancements extending their use to point cloud upsampling. Despite their potential, existing diffusion models struggle with inefficiencies as they map Gaussian noise to real point clouds, overlooking the geometric information inherent in sparse point clouds. To address these inefficiencies, the original authors propose PUFM, a flow matching approach to directly map sparse point clouds to their high-fidelity dense counterparts. Their method first employs midpoint interpolation to sparse point clouds, resolving the density mismatch between sparse and dense point clouds. Since point clouds are unordered representations, they introduce a pre-alignment method based on Earth Mover's Distance (EMD) optimization to ensure coherent interpolation between sparse and dense point clouds, which enables a more stable learning path in flow matching. Experiments on synthetic datasets demonstrate that their method delivers superior upsampling quality but with fewer sampling steps. Further experiments on ScanNet and KITTI also show that their approach generalizes well on RGB-D point clouds and LiDAR point clouds, making it more practical for real-world applications.

---

## Installation

### Verified Environment
**Recommended:** Python 3.9 + PyTorch 1.13 (CUDA 11.6)  
This combination has been tested and confirmed to work without compatibility issues.

```bash
# Create and activate conda environment
conda create -n pufm python=3.9 -y
conda activate pufm

# Install PyTorch with CUDA 11.6 (exact version match required)
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

# Install all required dependencies (including missing ones from original README)
pip install numpy==1.25.2 open3d==0.17.0 einops==0.3.2 scikit-learn==1.3.1 tqdm==4.62.3 h5py==3.6.0 plyfile ninja
```

### CUDA Extension Compilation
Compile the three custom CUDA extensions required for point cloud operations:

```bash
cd models/pointops
python setup.py install
cd ../../Chamfer3D
python setup.py install
cd ../emd_assignment
python setup.py install
cd ../..
```

> **Critical Fix Tip:** If you encounter `libc10.so: cannot open shared object file` when importing extensions, set the library path:
> ```bash
> # Temporary (current terminal only)
> export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/torch/lib:$LD_LIBRARY_PATH
> 
> # Permanent (auto-apply when activating environment)
> mkdir -p $CONDA_PREFIX/etc/conda/activate.d
> echo "export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/torch/lib:$LD_LIBRARY_PATH" >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
> ```

---

## Data Preparation

Download the required datasets from their official repositories:
- [PU1K Dataset](https://github.com/guochengqian/PU-GCN)
- [PUGAN Dataset](https://github.com/liruihui/PU-GAN)

Generate test point clouds from mesh files using the fixed `prepare_data.py` script:

```bash
cd dataset

# Generate PUGAN 4x test set (2048 input points → 8192 ground truth points)
python prepare_data.py --input_pts_num 2048 --R 4 --mesh_dir ../PUGAN/test --save_dir ../PUGAN/input_2048_4X/test

# Generate PUGAN 16x test set (2048 → 32768 points, for 16x evaluation)
python prepare_data.py --input_pts_num 2048 --R 16 --mesh_dir ../PUGAN/test --save_dir ../PUGAN/input_2048_16X/test

cd ..
```

Generated directory structure:
```
PUGAN/input_2048_4X/test/
└── input_2048_4X/
    ├── input_2048/     # Low-resolution input point clouds
    └── gt_8192/        # High-resolution ground truth point clouds
```

For additional data preparation details, refer to [Grad-PU](https://github.com/yunhe20/Grad-PU).

---

## Quick Start

Pretrained models provided by the original authors are included in the `pretrained_model/` folder.

### 1. Single Point Cloud Upsampling (4x)
```bash
python test_pufm.py --model pufm --test_input_path example/camel.xyz --up_rate 4
```

### 2. Arbitrary Upsampling (11x Example)
> **Note:** This script accepts a **single point cloud file path**, not a directory.

```bash
python test_pufm_arbitrary.py \
    --model pufm_w_attn \
    --dataset pugan \
    --test_input_path ./PUGAN/input_2048_4X/test/input_2048_4X/input_2048/camel.xyz \
    --up_rate 11
```

### 3. Batch Evaluation on PUGAN (4x)
**Using base PUFM model:**
```bash
python eval_pufm.py \
    --model pufm \
    --dataset pugan \
    --test_input_path ./PUGAN/input_2048_4X/test/input_2048_4X/input_2048/ \
    --test_gt_path ./PUGAN/input_2048_4X/test/input_2048_4X/gt_8192/ \
    --up_rate 4 \
    --ckpt_folder pretrained_model \
    --save_dir ./output/eval_4x
```

**Using PUFM with attention mechanism:**
```bash
python eval_pufm.py \
    --model pufm_w_attn \
    --dataset pugan \
    --test_input_path ./PUGAN/input_2048_4X/test/input_2048_4X/input_2048/ \
    --test_gt_path ./PUGAN/input_2048_4X/test/input_2048_4X/gt_8192/ \
    --up_rate 4 \
    --ckpt_folder pretrained_model \
    --save_dir ./output/eval_4x_attn
```

### 4. Batch Evaluation (16x)
> **Note:** 16x evaluation requires 16x ground truth data (32768 points). Generate it first using the `--R 16` command above.

```bash
python eval_pufm.py \
    --model pufm_w_attn \
    --dataset pugan \
    --test_input_path ./PUGAN/input_2048_16X/test/input_2048_16X/input_2048/ \
    --test_gt_path ./PUGAN/input_2048_16X/test/input_2048_16X/gt_32768/ \
    --up_rate 16 \
    --ckpt_folder pretrained_model \
    --save_dir ./output/eval_16x
```

---

## Training

### Train on PU1K Dataset
```bash
python train_pufm.py --dataset pu1k
```

> **Note:** The original release had a bug where `model_args` was missing `up_rate` and `num_points` attributes. This has been fixed in this repository. If you encounter `AttributeError`, ensure you are using the `train_pufm.py` from this repo.

---

## Known Fixes (Patched in This Repository)
This repository addresses all critical runtime bugs present in the original official release:

| # | File | Original Issue | Fix Applied |
|---|------|----------------|-------------|
| 1 | `test_pufm.py` | `sys.modules['pointops_cuda'] = None` forced JIT compilation, causing `RuntimeError: Ninja is required` | Removed the problematic line; uses pre-compiled extensions instead |
| 2 | `test_pufm.py` | Hardcoded `torch.cuda.is_available = lambda: False` forced CPU mode | Removed the line to allow normal GPU detection |
| 3 | `test_pufm_arbitrary.py` | Second `pcd_upsample` call lost batch dimension, causing `Expected 3 dims, got 2` | Added `.unsqueeze(0)` before the second call to restore batch dimension |
| 4 | `eval_pufm.py` | `parse_pc_args()` mistakenly parsed evaluation arguments, showing training help instead | Saved/restored `sys.argv` before calling `parse_pc_args()` to isolate parameter parsing |
| 5 | `train_pufm.py` | `model_args` missing `up_rate` and `num_points` attributes, causing `AttributeError` | Added fallback default values after `reset_model_args()` |
| 6 | `prepare_data.py` | Used `args.noise_level` and `args.noise_type` but did not define them in argparse | Added the two missing arguments with sensible defaults |
| 7 | `emd_assignment/` | Internal `import emd` failed because compiled module is named `emd_assignment` | Changed to `import emd_assignment as emd` to match compiled module name |
| 8 | Runtime | `libc10.so` / `libtorch_cpu.so` not found when importing compiled extensions | Added explicit `LD_LIBRARY_PATH` configuration instructions |

---

## Disclaimer
1. **Copyright Notice**: This repository is for **academic research purposes only**. All core algorithm code, pretrained models, and intellectual property rights belong to the original authors of the PUFM paper. This repository strictly follows the MIT License of the original project.
2. **Takedown Request**: If there is any copyright infringement, please contact me via GitHub Issues, and I will immediately take down this repository.
3. **Debugging Notes**: Due to the long debugging and fixing cycle of the original code, some detailed implementation notes of the fixes may not be fully documented.
4. **Community Feedback**: If you encounter any new bugs or issues while using this repository, please feel free to submit an Issue or Pull Request. Your feedback is highly appreciated.

---

## Acknowledgments
- The original PUFM implementation is built upon [PU-GCN](https://github.com/guochengqian/PU-GCN), [PU-GAN](https://github.com/liruihui/PU-GAN), and [Grad-PU](https://github.com/yunhe20/Grad-PU). We thank the authors of these repositories for their excellent open-source work.
- We sincerely thank the original PUFM authors for releasing their research and code to the community.

---

## Citation
If you find this work useful, please cite the original paper:
```bibtex
@InProceedings{ZSLiu_2025,
    author    = {Zhi-Song Liu and Chenhang He and Lei Li},
    title     = {Efficient Point Clouds Upsampling via Flow Matching},
    booktitle = {arXiv:2501.15286},
    year      = {2025}
}
```

If you use this bug-fixed version in your research, please also consider citing this repository:
```bibtex
@misc{pufm-fixed-2026,
  title={PUFM-Fixed: Bug-Free Reproduction of Efficient Point Cloud Upsampling via Flow Matching},
  author={Dekang Zhu},
  year={2026},
  howpublished={\url{https://github.com/DekangZhu/PUFM-Fixed}}
}
```

---

## License
This repository follows the MIT License, consistent with the original project. All core algorithm code is copyright © 2025 the original PUFM authors.
