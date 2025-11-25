# Mistral Environment Setup on Scratch

This README sets up a Python environment for running Mistral models on a scratch filesystem. Copy-paste the entire block below into your terminal.

```bash
# Create a folder for your environment and set up a virtual environment
mkdir -p $SCRATCH/mistral-env
python -m venv $SCRATCH/mistral-env

# Activate the environment
source $SCRATCH/mistral-env/bin/activate

# Configure Hugging Face cache
export HF_HOME=$SCRATCH/huggingface
source ~/.bash_profile
mkdir -p $HF_HOME

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support
pip install --prefix=$SCRATCH/mistral-env torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Hugging Face libraries
pip install --prefix=$SCRATCH/mistral-env transformers accelerate bitsandbytes huggingface_hub

# Set PYTHONPATH (replace python3.10 with your Python version)
export PYTHONPATH=$SCRATCH/mistral-env/lib/python3.10/site-packages:$PYTHONPATH

# Verify installation
python -c "import torch; print(torch.cuda.is_available())"
python -c "import transformers; print(transformers.__version__)"

# Login to Hugging Face
huggingface-cli login

# Download the Mistral model
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.2 --local-dir /pscratch/sd/n/nataraj2/mistral/7b --local-dir-use-symlinks False

# Optionally, set Hugging Face home to a persistent location
export HF_HOME=/pscratch/sd/n/nataraj2/.hf


