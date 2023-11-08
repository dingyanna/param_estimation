cd ..
declare exp=topo
declare max_iter=2000
python3 run.py --experiment $exp --dynamics eco --max_iter $max_iter
python3 run.py --experiment $exp --dynamics gene --max_iter $max_iter