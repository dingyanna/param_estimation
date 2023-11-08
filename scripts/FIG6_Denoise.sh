cd ..

declare exp=noise_level
for i in `seq 0 100`;
do 
    python3 run.py --experiment ${exp} --topology er --dynamics gene --n 200 --k 8  --ub 2.5 --seed $i --prob_theta0 n
    python3 run.py --experiment ${exp} --topology sf --dynamics gene --n 200 --gamma 2.1  --ub 2.5 --seed $i --prob_theta0 n
    
    python3 run.py --experiment ${exp} --topology sf --dynamics epi --n 200 --gamma 2.1 --ub 1 --seed $i --prob_theta0 n
    python3 run.py --experiment ${exp} --topology er --dynamics epi --n 200 --k 8 --ub 1 --seed $i --prob_theta0 n

    python3 run.py --experiment ${exp} --topology er --dynamics eco --n 200 --k 8 --seed $i --prob_theta0 n
    python3 run.py --experiment ${exp} --topology sf --dynamics eco --n 200 --gamma 2.1 --seed $i --prob_theta0 n
done
