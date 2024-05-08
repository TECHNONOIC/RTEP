# for nvidia gpus on windows only 
python -m venv ./venv
./venv/Scripts/activate
# cd C:\Users\ayush\miniconda3\envs\gpuenv
# ./Scripts/activate
python -m pip install --upgrade pip
pip install dataclasses
pip install jupyterlab
# for loading arff
pip install numpy
pip install matplotlib
pip install seaborn
pip install scikit-learn
pip install scipy
# for dataframe and some error
pip install pandas
pip install pyarrow
# for xlsx conversion
pip install openpyxl
# pytorch
pip install torchvision
pip install torch
# tensorflow
pip install tensorflow 
pip install -q git+https://github.com/tensorflow/docs
pip install imageio
