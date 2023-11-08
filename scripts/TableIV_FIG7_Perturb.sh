cd ..
declare exp=perturb
for i in `seq 0 0`;
do 
    python3 run.py --experiment ${exp} --topology real --data net8 --dynamics eco --seed $i 
    python3 run.py --experiment ${exp} --topology real --data net6 --dynamics eco --seed $i   
    python3 run.py --experiment ${exp} --topology real --data tya --dynamics gene --seed $i 
    python3 run.py --experiment ${exp} --topology real --data mec --dynamics gene --seed $i 
    python3 run.py --experiment ${exp} --topology real --data arenas-email --dynamics epi --seed $i  
    python3 run.py --experiment ${exp} --topology real --data infect-dublin --dynamics epi --seed $i  
done


