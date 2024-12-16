# SITCOM_phase_retrieval
To address the instability issue in SITCOM phase retrieval, we drew inspiration from DAPS and incorporated an ODE solver to replace the Tweedie formula update, enabling a more robust and accurate solution. The revised approach integrates an Euler solver, enhanced with a data consistency projection, ensuring the ODE solution maintains greater consistency throughout the process. Additionally, we implemented a data consistency update guided by a Lagrange multiplier, further refining the overall stability and performance of the method. The updated code is provided in the sampler class DiffusionSampler for the eulder method.


## Getting started

#### 1. Prepare the Environment

- python 3.8
- PyTorch 2.3
- CUDA 12.1

Lower version of PyTorch with proper CUDA should work but not be fully tested.

```
# in SITCOM_phase_retrieval folder

conda create -n SITCOM_phase_retrieval python=3.8
conda activate SITCOM_phase_retrieval

pip install -r requirements.txt

# (optional) install PyTorch with proper CUDA
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```

#### 2. Prepare the pretrained checkpoint

Download the public available FFHQ checkpoint (ffhq_10m.pt) [here](https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh).

```
# in SITCOM_phase_retrieval folder

mkdir checkpoint
mv {DOWNLOAD_DIR}/ffqh_10m.pt checkpoint/ffhq256.pt
```



#### 3.  (Optional) Prepare the dataset (or use provided examples)

You can add any FFHQ256 images you like to `dataset/demo` folder



#### 4. Sample

Make a folder to save results:

```
mkdir results
```

##### Phase Retrieval

Now you are ready for run. For **phase retrieval** with SITCOM in 4 runs for $10$ demo images in `dataset/demo`:

```
python posterior_sample.py \
+data=demo \
+model=ffhq256ddpm \
+task=phase_retrieval \
+sampler=edm_daps \
save_dir=results \
num_runs=4 \
sampler.diffusion_scheduler_config.num_steps=5 \
sampler.annealing_scheduler_config.num_steps=200 \
batch_size=10 \
data.start_id=0 data.end_id=10 \
name=phase_retrieval_demo \
gpu=0
```

This code is based on DAPS [here](https://github.com/zhangbingliang2019/DAPS/tree/main)
