git clone git@github.com:ultimagen/prometheus-stack
cd prometheus-stack/
git switch local
./start.sh

git clone git@github.com:ultimagen/VariantCalling
cd VariantCalling/
git switch annotate-featuremap

curl https://pyenv.run | bash
vi ~/.bashrc
exec $SHELL
sudo apt install libreadline-dev
sudo apt install libsqlite3-dev
pyenv install 3.11.9
pyenv virtualenv 3.11.9 annotate-featuremap
pyenv activate annotate-featuremap
pyenv local annotate-featuremap

pip install uv -U pip
uv pip install -r requirements.in