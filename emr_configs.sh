# AWS EMR 5.4.0 bootstrap script for installing Anaconda

# ----------------------------------------------------------------------
# move /usr/lib to /mnt/usr-moved/lib to avoid running out of space on /
# ----------------------------------------------------------------------
sudo mkdir /mnt/usr-moved
sudo mv /usr/local /mnt/usr-moved/
sudo ln -s /mnt/usr-moved/local /usr/
sudo mv /usr/share /mnt/usr-moved/
sudo ln -s /mnt/usr-moved/share /usr/

# ----------------------------------------------------------------------
#              Install Anaconda (Python 3) & Set To Default              
# ----------------------------------------------------------------------
wget https://repo.continuum.io/archive/Anaconda3-4.3.1-Linux-x86_64.sh -O ~/anaconda.sh
bash ~/anaconda.sh -b -p $HOME/anaconda
echo -e '\nexport PATH=$HOME/anaconda/bin:$PATH' >> $HOME/.bashrc && source $HOME/.bashrc

# ----------------------------------------------------------------------
#                    Install Additional Packages              
# ----------------------------------------------------------------------
conda install -y psycopg2 gensim
pip install textblob selenium 

# ----------------------------------------------------------------------
#   Install Tensorflow (CPU only and installs Keras to act as backend)              
# ----------------------------------------------------------------------
conda create -yn tensorflow
source activate tensorflow
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.0.1-cp36-cp36m-linux_x86_64.whl
source deactivate

# ----------------------------------------------------------------------
#                    Install Theano            
# ----------------------------------------------------------------------
conda install -y theano pygpu