import subprocess
import pandas as pd
import time
"""Main Script for batched training 
Unessicary unless you're me and cant solve memory problem
"""



for high in range(10000,20000, 5000):
    tag = 0   

    print(f'Training Started: {high}')


    subprocess.run(["python3", "train_gpt.py", "--data_subset_range_low", str(high-5000), "--data_subset_range_high", str(high), "--tag", str(tag) ])

    print(f'Training Complete: {high}')

    time.sleep(120)

