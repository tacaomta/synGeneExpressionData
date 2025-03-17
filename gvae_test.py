from gene_vae import GVAE
from gene import GeneNetwork

network = GeneNetwork(fr"C:\caocao\gnw-master\tave_gen\hybrid\s10\sample1_network.txt", 
                      fr"C:\caocao\gnw-master\tave_gen\hybrid\s10\sample1_goldstandard.txt",None, False, 
                                  print_out=False)

outpath = fr'C:\caocao\2023\CodeSpace\BooleanInference\myMIBNI\networks\size10.txt'

vae = GVAE(network)
vae.fit()
vae.sample(outpath, rows=network.timestepsNumber)