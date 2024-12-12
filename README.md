# SITCOM_phase_retrieval
To address the instability issue in SITCOM phase retrieval, we drew inspiration from DAPS and incorporated an ODE solver to replace the Tweedie formula update, enabling a more robust and accurate solution. The revised approach integrates an Euler solver, enhanced with a data consistency projection, ensuring the ODE solution maintains greater consistency throughout the process. Additionally, we implemented a data consistency update guided by a Lagrange multiplier, further refining the overall stability and performance of the method. The updated code is provided in the sampler class DiffusionSampler for the eulder method.

    def _euler(self, model, x_start,operator, measurement, SDE=False, record=False, verbose=False):
        """
            Euler's method for sampling from the diffusion process.
            
        """
        if record:
            self.trajectory = Trajectory()
        pbar = tqdm.trange(self.scheduler.num_steps) if verbose else range(self.scheduler.num_steps)
        scale = 0.1
        x = x_start
        for step in pbar:
            sigma, factor = self.scheduler.sigma_steps[step], self.scheduler.factor_steps[step]
            score = model.score(x, sigma)
            if SDE:
                epsilon = torch.randn_like(x)
                x = x + factor * score + np.sqrt(factor) * epsilon
            else:
                x = x + factor * score * 0.5
            x_hat = x.clone().detach().requires_grad_(True)
            difference = operator.error(x_hat,measurement)
            norm = torch.linalg.norm(difference)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_hat)[0]
            # record
            x -= norm_grad*scale
            if record:
                if SDE:
                    self._record(x, score, sigma, factor, epsilon)
                else:
                    self._record(x, score, sigma, factor)
        return x
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


