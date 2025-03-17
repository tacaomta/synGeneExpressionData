import pandas as pd
import numpy as np

sizes = [10, 50, 100]
steps = [10, 20, 30, 40, 50]
for size in sizes:
    header = 'Time\t'
    for i in range(size):
        header+=f'G{i+1}\t'
    header=header[:-1]
    for st in steps:
        timestep = [i for i in range(50)]
        for sp in range(1, 11):
            filename = fr'C:\caocao\gnw-master\tave_gen\hybrid\s{size}\synthetic\s{st}\fake{size}_{st}st_{sp}.csv'

            dt = pd.read_csv(filename, header=None, index_col=None)
            dt.insert(0, 'Time', timestep)
            outpath = fr'C:\caocao\gnw-master\tave_gen\hybrid\s{size}\synthetic\s{st}\fake{size}_{st}st_{sp}.txt'
            np.savetxt(outpath, dt, delimiter='\t', header=header, fmt='%s')