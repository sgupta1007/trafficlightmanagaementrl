sudo apt-get install sumo sumo-tools sumo-doc 
echo 'export SUMO_HOME="/usr/share/sumo"' >> ~/.bashrc
source ~/.bashrc
git clone https://github.com/LucasAlegre/sumo-rl
cd sumo-rl
pip install -e .
