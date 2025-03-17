from gen_network import Generation
import pandas as pd

#size = [i for i in range(10, 101, 10)]
size = [100]
sample = [i for i in range(201, 401, 1)]
for si in size:
    index=0
    raw_csv = {}
    for sam in sample:
        folder=fr'C:\caocao\gnw-master\ANN\toy dataset\training'
        gen = Generation(size=si,timestep_number=50, bits=(0, 1, 2), timelag=1, folder=folder)
        network = gen.generate(f'sample{sam}', False)
        for regulator, target, reaction in network.goldstandard:
            d = {'size': si, 'sample':sam, 'regulator':regulator, 'target':target, 'interaction': reaction}
            for node in network.nodes:
                if node.name==regulator:
                    regulator_node = node
                    break
            for node in network.nodes:
                if node.name==target:
                    target_node = node
                    break
            for i, v in enumerate(regulator_node.values):
                d[f'r_step{i}'] = v
            for i, v in enumerate(target_node.values):
                d[f't_step{i}'] = v
            raw_csv[index] = d
            index+=1
        print(f'sample {sam} is done...')
    print('conversion is starting...')
    data = pd.DataFrame.from_dict(raw_csv, orient='index')
    data.to_csv(r'C:\caocao\gnw-master\ANN\toy dataset\test\toy_size100_2.csv')
print('process is completely done...')    