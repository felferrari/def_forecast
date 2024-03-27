import subprocess

for i in range(12951): #12951
    subprocess.run([f"./run.sh mlp_features_{i}"], shell = True, executable="/bin/bash") 
    subprocess.run([f"./run.sh transformer_features_{i}"], shell = True, executable="/bin/bash") 