# python installation
sudo apt install python3

# anaconda 3 installation
if ! [ -x "$(command -v conda)" ]; then
    curl -O https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh;
    chmod +x Anaconda3-5.2.0-Linux-x86_64.sh;
    ./Anaconda3-5.2.0-Linux-x86_64.sh;
    export PATH="/home/$(whoami)/anaconda3/bin:$PATH"
fi

# preparing conda environment
conda create --name deep_learning python=3
source activate deep_learning

# data science and computing libraries installation
echo "INSTALLING REQUIRED LIBRARIES ..."
echo "installing scikit-learn [1/4] ..."
conda install scikit-learn
echo "installing keras [2/4] ..."
conda install -c conda-forge keras
echo "installing pandas [3/4]"
conda install pandas
echo "installing mysql connector [4/4]"
conda install -c anaconda mysql-connector-python

echo "INSTALLATION FINISHED, YOU ARE READY TO GO!!!"


