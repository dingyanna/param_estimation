cd .. 
declare exp=brn_vs_true 

python3 run.py --experiment ${exp} --topology er --dynamics gene --n 200 --k 12   
python3 run.py --experiment ${exp} --topology er --dynamics gene --n 2000 --k 16   
python3 run.py --experiment ${exp} --topology sf --dynamics gene --n 200 --gamma 2.1  
python3 run.py --experiment ${exp} --topology sf --dynamics gene --n 2000 --gamma 2.1   

python3 run.py --experiment ${exp} --topology er --dynamics epi --n 200 --k 12  
python3 run.py --experiment ${exp} --topology er --dynamics epi --n 2000 --k 16  
python3 run.py --experiment ${exp} --topology sf --dynamics epi --n 200 --gamma 2.1  
python3 run.py --experiment ${exp} --topology sf --dynamics epi --n 2000 --gamma 2.1 

python3 run.py --experiment ${exp} --topology er --dynamics eco --n 200 --k 12  
python3 run.py --experiment ${exp} --topology er --dynamics eco --n 2000 --k 16 
python3 run.py --experiment ${exp} --topology sf --dynamics eco --n 200 --gamma 2.1  
python3 run.py --experiment ${exp} --topology sf --dynamics eco --n 2000 --gamma 2.1  
 