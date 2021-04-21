## Requirements
cd so_textprocessing
pip3 install .
cd ..
pip3 install -r requirements.txt

echo "Running data/getdata.py"
python3 data/getdata.py

echo "Running RQ1"
python3 rq1/rq1.py

echo "Running RQ2"
python3 rq2/rq2.py

echo "Running RQ3"
python3 rq3/rq3.py

echo "Running RQ4"
python3 rq4/rq4.py

echo "Done - results saved to ./outputs"
