 
cd .. 

declare exp=adv_recover
for i in `seq 0 0`;
do 
    python3 run.py --experiment ${exp} --topology real --data net8 --dynamics eco --seed $i --tol 0.001 --noise_type 2 --noise_level 0.05 --ss_solver mfa
    python3 run.py --experiment ${exp} --topology real --data net6 --dynamics eco --seed $i --tol 0.001 --noise_type 2 --noise_level 0.05 --ss_solver mfa
    python3 run.py --experiment ${exp} --topology real --data tya --dynamics gene --seed $i --tol 1e-8 --ub 3 --noise_type 2 --noise_level 0.05 --ss_solver mfa
    python3 run.py --experiment ${exp} --topology real --data mec --dynamics gene --seed $i --tol 1e-8 --ub 3 --noise_type 2 --noise_level 0.05 --ss_solver mfa
    python3 run.py --experiment ${exp} --topology real --data arenas-email --dynamics epi --seed $i --tol 1e-8 --ub 1 --noise_type 2 --noise_level 0.05 --ss_solver mfa
    python3 run.py --experiment ${exp} --topology real --data infect-dublin --dynamics epi --seed $i --tol 1e-8 --ub 1 --noise_type 2 --noise_level 0.05 --ss_solver mfa
done



