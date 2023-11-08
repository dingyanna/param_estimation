cd ..
declare size=(100 1000 3000 5000 10000)
declare ss_solver=(mfa)
declare dyn=(eco gene epi)
declare seed=7
for i in ${!size[@]}
do 
    for j in ${!dyn[@]}
    do 
        for k in ${!ss_solver[@]}
        do 
            python3 run.py --experiment compute_ss --topology er --n ${size[$i]} --dynamics ${dyn[$j]} --ss_solver ${ss_solver[$k]} --seed $seed  
        done
    done 
done 

