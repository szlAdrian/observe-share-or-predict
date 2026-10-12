# Habitat-LLM Installation Instructions

### Requirements:
- Conda or Mamba

### Create and activate Conda environment
```bash
conda create -n habitat-llm  python=3.9.2 cmake=3.14.0 -y
conda activate habitat-llm
```

### Initialize third party submodules
```bash
git submodule sync
git submodule update --init --recursive
```

### Install dependencies and requirements
```bash
# Adjust the cuda version depending on your hardware stack
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia -y
# Install habitat-sim version 0.3.3
conda install habitat-sim=0.3.3 withbullet headless -c conda-forge -c aihabitat -y
# NOTE: If the above fails, packages may not be available for your system. Install from source (see https://github.com/facebookresearch/habitat-sim).
pip install -e ./third_party/habitat-lab/habitat-lab
pip install -e ./third_party/habitat-lab/habitat-baselines
pip install -e ./third_party/transformers-CFG
pip install -r requirements.txt
```
If you have issues with library linking make sure that the conda libraries are in your LD_LIBRARY_PATH (e.g `export LD_LIBRARY_PATH=/path/to/anaconda/envs/myenv/lib:$LD_LIBRARY_PATH`)

### Download datasets
```bash
# You may have to re-run downloader commands in case of network errors.
python -m habitat_sim.utils.datasets_download --uids rearrange_task_assets hab_spot_arm hab3-episodes habitat_humanoids --data-path data/ --no-replace --no-prune

# Download ovmm objects
git clone https://huggingface.co/datasets/ai-habitat/OVMM_objects data/objects_ovmm --recursive
```

### Setup HSSD scene dataset
```bash
# Download and link the data.
git clone -b partnr https://huggingface.co/datasets/hssd/hssd-hab data/versioned_data/hssd-hab
cd data/versioned_data/hssd-hab
git lfs pull
cd ../../..
ln -s versioned_data/hssd-hab data/hssd-hab
```

### Download task datasets and neural network skill checkpoints

```bash
# Download the data
git clone https://huggingface.co/datasets/ai-habitat/partnr_episodes data/versioned_data/partnr_episodes
cd data/versioned_data/partnr_episodes
git lfs pull
cd ../../..

# Link task datasets
mkdir -p data/datasets
ln -s ../versioned_data/partnr_episodes data/datasets/partnr_episodes

# Link skill checkpoints
ln -s versioned_data/partnr_episodes/checkpoints data/models
```

### (Optional) Install pybullet for IK based controllers
```bash
pip install pybullet==3.0.4
```

### Quantized LLMs (bitsandbytes)
`bitsandbytes` is installed by `requirements.txt` (used by `llm.load_in_4bit`
in `hf` mode). If you're serving a `--quantization bitsandbytes` model through
the separate vLLM venv below, it needs to be installed **there too** (it runs
inside the vLLM server process) — `requirements_vllm.txt` covers that.

### (Optional) Set up a vLLM server for fast/memory-efficient inference
`inference_mode: vllm` (`habitat_llm/conf/llm/vllm.yaml`) talks to a vLLM
OpenAI-compatible server over HTTP instead of loading the model in-process.
Since it's a separate process, **install vLLM in its own conda/venv env**
rather than the `habitat-llm` env — it doesn't need to share habitat-llm's
torch 2.4.1 pin, and mixing them risks dependency conflicts.

```bash
python3.11 -m venv ~/.venvs/vllm
source ~/.venvs/vllm/bin/activate
pip install "vllm>=0.6.5"  # xgrammar guided-decoding backend needs >=0.6.5
pip install "bitsandbytes>=0.45.3"  # only needed if serving with --quantization bitsandbytes
```

If your NVIDIA driver is capped below what a plain `pip install vllm` requires
(e.g. `RuntimeError: The NVIDIA driver on your system is too old`), check the
max CUDA version your driver supports with `nvidia-smi` (top-right of the
header) and, if you can't upgrade the driver (e.g. no root on a shared
cluster), pin vLLM and torch to a release line built for that CUDA version
instead. For a driver capped at CUDA 12.4, `requirements_vllm.txt` pins a
working combination (torch 2.6.0+cu124, vllm 0.8.5, transformers 4.51.1,
bitsandbytes):
```bash
pip install -r requirements_vllm.txt
# Only if a newer vllm was installed in this env before: remove its leftover
# flashinfer/tvm_ffi build (built against the newer torch's ABI, and vllm==0.8.5
# already disables flashinfer's sampler anyway):
pip uninstall -y flashinfer flashinfer-python tvm-ffi tvm_ffi torch-c-dlpack-ext torch_c_dlpack_ext
```
(vLLM's own CUDA-version-specific wheels — see the "Install vLLM with pip for
other CUDA versions" section of the
[vLLM docs](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html) —
are another option if a matching prebuilt combo exists for your CUDA version.)

Start the server (see `run.sh` for the exact commands, 8B full-precision and
70B bitsandbytes 4-bit variants), then point `habitat-llm` at it with
`llm.host`/`llm.port` (defaults to `localhost:8000`) and
`llm.generation_params.engine` matching `--served-model-name`. Grammar-
constrained baselines (`constrained_generation: True`) require the server to
be started with `--guided-decoding-backend xgrammar`. On vllm>=0.12.0 this
flag was renamed to `--structured-outputs-config.backend xgrammar` — use
whichever matches your installed vLLM version.

### Install pre-commit
```bash
pip install pre-commit && pre-commit install
```

### Install the habitat-llm library
```bash
pip install -e .
```

### Setup api keys if needed:
```bash
# Add the following to your ~/.bashrc file
export OPENAI_API_KEY=...
```

### Run the tests
```bash
# make sure to use bash shell if on zsh
bash

# Download and link the data.
git clone https://huggingface.co/datasets/ai-habitat/hssd-partnr-ci data/versioned_data/hssd-partnr-ci
ln -s versioned_data/hssd-partnr-ci data/hssd-partnr-ci
cd data/hssd-partnr-ci
git lfs pull
cd ../..

# link RAG testing data
ln -s versioned_data/partnr_episodes/test_rag data/test_rag

# then, run the tests
python -m pytest habitat_llm/tests [-v]
```

#### Troubleshooting

If the tests mentioned above fail due to scipy, numpy, pandas or opencv incompatibility, try installing the following alternate versions of selected packages which worked for us on Ubuntu 22.04.5 LTS:
```
pip install scipy==1.12.0 # make sure to run this before numpy or pandas install
pip install numpy==1.22.0
pip install pandas==2.0.3
pip install opencv-python==4.10.0.82
```
