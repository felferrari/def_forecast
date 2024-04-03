import subprocess

for i in range(2325): #2325
    subprocess.run([f"./run.sh mlp_features_{i}"], shell = True, executable="/bin/bash") 
    subprocess.run([f"./run.sh transformer_features_{i}"], shell = True, executable="/bin/bash") 