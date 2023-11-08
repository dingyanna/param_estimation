cd ..
declare exp=network_size
for i in `seq 0 0`;
do 
    python3 run.py --experiment ${exp} --topology er --dynamics gene --seed $i --tol 1e-8 --ss_solver mfa
    python3 run.py --experiment ${exp} --topology sf --dynamics gene --seed $i --tol 1e-8 --ss_solver mfa

    python3 run.py --experiment ${exp} --topology er --dynamics epi --seed $i --tol 1e-8 --ss_solver mfa
    python3 run.py --experiment ${exp} --topology sf --dynamics epi --seed $i --tol 1e-8 --ss_solver mfa
done

