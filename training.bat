@echo off

cd  C:\Users\DL_ws\Desktop\end2end_master
echo start training
start cmd /k "cd  C:\Users\DL_ws\Desktop\end2end_master\data_gen && echo Data generator windows && python server.py --time 5 --batch 32 --port 5557"  

start cmd /k "cd  C:\Users\DL_ws\Desktop\end2end_master\data_gen && echo Data generator windows && python server.py --time 5 --batch 32 --validation --port 5556" 
 

start cmd /k " echo Tensorboard panel  && tensorboard --logdir="logs"  "