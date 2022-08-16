cd ..
pip install streamlit loguru vimms stable-baselines3 optuna plotly kaleido cloudpickle==2.1.0
pip install git+https://github.com/DLR-RM/stable-baselines3
apt-get install -y htop

# cd /notebooks/bin
# wget https://github.com/dropbox/dbxcli/releases/download/v3.0.0/dbxcli-linux-amd64
# mv dbxcli-linux-amd64 dbxcli
# chmod +x dbxcli

# wget https://github.com/feklee/dbx-tools/archive/refs/tags/v0.7.1.tar.gz
# tar xvzf v0.7.1.tar.gz
# rm v0.7.1.tar.gz
# mv dbx-tools-0.7.1/* .
# rm -rf dbx-tools-0.7.1