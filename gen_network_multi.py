from gen_network import Generation

#size = [i for i in range(10, 101, 10)]
size = [10, 50, 100]
sample = [i for i in range(1, 11, 1)]
for si in size:
    for sam in sample:
        folder=fr'C:\caocao\gnw-master\tave_gen\new dataset\boolean\size{si}'
        gen = Generation(size=si,timestep_number=100, bits=(0, 1), timelag=1, folder=folder)
        gen.generate(f'sample{sam}')