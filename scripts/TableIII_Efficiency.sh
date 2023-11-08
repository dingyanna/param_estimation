cd .. 
declare exp=efficiency
declare max_iter=2
declare prob_theta0=(n u) 
for i in `seq 0 29`;
do  
    for j in ${!prob_theta0[@]}
    do 
        python3 run.py --experiment ${exp} --topology real --data net8 --dynamics eco --seed $i --max_iter $max_iter --prob_theta0 ${prob_theta0[$j]}
        python3 run.py --experiment ${exp} --topology real --data tya --dynamics gene --seed $i --ub 3   --max_iter $max_iter --prob_theta0 ${prob_theta0[$j]}
        python3 run.py --experiment ${exp} --topology real --data mec --dynamics gene --seed $i --ub 3 --max_iter $max_iter --prob_theta0 ${prob_theta0[$j]}
        python3 run.py --experiment ${exp} --topology real --data infect-dublin --dynamics epi --seed $i --ub 1 --max_iter $max_iter  --prob_theta0 ${prob_theta0[$j]}
        python3 run.py --experiment ${exp} --topology real --data arenas-email --dynamics epi --seed $i --ub 1  --max_iter $max_iter  --prob_theta0 ${prob_theta0[$j]}
        python3 run.py --experiment ${exp} --topology real --data net6 --dynamics eco --seed $i --max_iter $max_iter  --prob_theta0 ${prob_theta0[$j]}
    done  
done


