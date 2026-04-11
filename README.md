# s4_flight
s4 (Structured State Spaces for Sequence Modeling) official repository but with adaptation to my flight data. Original code not mine - https://github.com/state-spaces/s4/tree/main

before trying to run please run the next commands in order

module load python/3.10.14
python -m venv .env
source .env/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu116