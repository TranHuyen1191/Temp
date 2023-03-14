Ubuntu 20.04:     

# Install sublime-text and anaconda 

	sudo apt update
	sudo apt install apt-transport-https ca-certificates curl software-properties-common
	curl -fsSL https://download.sublimetext.com/sublimehq-pub.gpg | sudo apt-key add -
	sudo add-apt-repository "deb https://download.sublimetext.com/ apt/stable/"
	sudo apt update
	sudo apt install sublime-text
	wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
	sh Anaconda3-2020.11-Linux-x86_64.sh -b -f 
	~/anaconda3/bin/conda init bash

# Check NVIDIA 
	inxi -Gx
	################ OUTPUT ############# 
	Graphics: 	Device-1: NVIDIA vendor: ZOTAC driver: nvidia v: 470.57.02 bus ID: 65:00.0
							Display: x11 server: X.Org 1.20.11 driver: nvidia unloaded: fbdev,modesetting,nouveau,vesa 
						 resolution: 2560x1440~60Hz, 1920x1080~60Hz 
						 OpenGL: renderer: NVIDIA GeForce RTX 3090/PCIe/SSE2 v: 4.6.0 NVIDIA 470.57.02 direct render: Yes 
	#############################

# Install driver

	wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
	sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
	sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
	sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
	sudo apt-get update
	wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/nvidia-machine-learning-repo-ubuntu2004_1.0.0-1_amd64.deb
	sudo apt install ./nvidia-machine-learning-repo-ubuntu2004_1.0.0-1_amd64.deb
	sudo apt-get update



## Check here to know the suitable version of driver https://www.nvidia.com/Download/Find.aspx 
## Replace *** in the below command by the version
	sudo apt-get install --no-install-recommends nvidia-driver-***

**For example: sudo apt-get install nvidia-driver-471

### COMMON ERROR:

	nvidia-driver-460 : Depends: nvidia-dkms-460 (= 460.91.03-0ubuntu1)
                      Depends: libnvidia-extra-460 (= 460.91.03-0ubuntu1) but it is not going to be installed
Solution: Manually install packages: 
	
	sudo apt-get install libnvidia-extra-460  nvidia-compute-utils-460


## Reboot. Check if GPU is visible using the command: 

	sudo reboot
	nvidia-smi

## Install CUDA

	wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/libnccl2_2.8.3-1+cuda11.0_amd64.deb
	sudo apt install ./libnccl2_2.8.3-1+cuda11.0_amd64.deb
	wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/libnccl-dev_2.8.3-1+cuda11.0_amd64.deb
	sudo apt install ./libnccl-dev_2.8.3-1+cuda11.0_amd64.deb
	sudo apt-get update
	echo 'Installing CUDA development and runtime libraries (~4GB)'
	sudo apt-get install --no-install-recommends  cuda-11-0  libcudnn8   libcudnn8-dev
	sudo apt install nvidia-cuda-toolkit
	nvcc -V
	
## Option 1: Manually create environment using Conda
	conda create -n tfmixer pip python=3.8 numpy=1.18.5 tensorflow=2.3
	pip install flax
	pip install tqdm
	pip install clu==0.0.1-alpha.2
	pip install flax==0.2.2
	pip install ml_collections==0.1.0
	pip install numpy==1.18.5
	pip install tensorflow-datasets==4.0.1
	pip install tensorflow-probability==0.11.1
	pip install google-colab
	pip install einops
	pip install ml_collections
	python3 -m pip uninstall -y jupyter jupyter_core jupyter-client jupyter-console jupyterlab_pygments notebook qtconsole nbconvert nbformat tornado
	pip install notebook tornado
	wget https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz
	conda install -c intel mkl
	conda install numpy
	pip install --upgrade jax jaxlib==0.1.67+cuda111 -f https://storage.googleapis.com/jax-releases/jax_releases.html
	pip install gsutil
	conda install -c anaconda ipython
	python -m ipykernel install --user --name=tfmixer
	jupyter kernelspec list
	jupyter kernelspec uninstall unwanted-kernel
	
	
### Check jax device 
	python3 -c 'import tensorflow as tf; print(tf.__version__);tf.test.gpu_device_name()'  # for Python 3
	python3 -c 'import jax ; print(jax.devices())'  # for Python 3

## Option 2: Create environment using transmmiss_env.yml file:
	conda env create -f transmmiss_env.yml

## Activate the environment:
	conda activate tfmixer
					
## Run jupyter notebook
	jupyter notebook

## Open and Run .ipynb files  (refer to https://www.youtube.com/watch?v=2V5Gq_iYqsY to see how to run example.ipynb)
    Step1_finetuning.ipynb: Fine tuning model parameters
    Step2_calPerformance.ipynb: Calculate the performance of TransMiSS for the training and test sets with different number of epochs
    Step3_savePrediction.ipynb: Extract some examples of image segmentation
    Step4_Threshold_1.ipynb: Calculate the performance of Threshold_1
    Step4_Threshold_2.ipynb: Calculate the performance of Threshold_2
