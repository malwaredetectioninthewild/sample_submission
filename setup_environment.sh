conda create -y sample_submission python=3.8 pip
conda activate sample_submission

conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install scikit-learn