import numpy as np
import os
import glob

with open("synth_idx.txt", "w") as fw:
    for name in glob.iglob("./synth/uniform/images/*.png"):
        fw.write(os.path.abspath(name) + "\n")
        print(os.path.abspath(name))
