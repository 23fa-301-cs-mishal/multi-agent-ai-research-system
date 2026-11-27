conda create -n langagent python=3.11 -y

conda activate langagent

pip install -r requirements.txt

python -m playwright install chromium

playwright install
pip install pymupdf
