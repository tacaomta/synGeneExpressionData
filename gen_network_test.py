from gen_network import Generation

size = 100
folder=fr'C:\caocao\gnw-master\tave_gen\hybrid\s{size}'
gen = Generation(size=size,timestep_number=50, bits=(0, 1, 2), timelag=1, folder=folder)
gen.generate('sample1')