cd ..

declare exp=plot_loss
declare i=2
python3 run.py --experiment ${exp} --topology real --data net8 --dynamics eco --seed $i --tol 0.001 --ss_solver full
python3 run.py --experiment ${exp} --topology real --data net8 --dynamics eco --seed $i --tol 0.001 --ss_solver mfa
 