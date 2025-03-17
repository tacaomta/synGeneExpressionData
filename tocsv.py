from gene import GeneNetwork


sizes=[10, 50, 100]
sps=[i for i in range(1, 2)]
timesteps = [(0, 10), (0, 20), (0, 30), (0, 40), (0, 50)]

for size in sizes:
    for sp in sps:
        file = fr'C:\caocao\gnw-master\tave_gen\hybrid\s{size}\sample1_network.txt'
        network = GeneNetwork(file, timeseries=False)
        for start, end in timesteps:
            sn = network.getSubTimeStepsData(start, end)
            filename = f'{network.folder}\{network.filename}_{end}.csv'
            sn.to_csv(filename)

