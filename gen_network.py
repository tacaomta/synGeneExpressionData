import random
from gene import GeneNetwork, Gene
import copy
import numpy as np

class Generation():
    '''
    Generate discrete network
    '''
    def __init__(self, size=10, timestep_number=50, bits=(0, 1), timelag=1, folder=None) -> None:
        self.size = size
        self.timestep_number = timestep_number
        self.bits = bits
        self.timelag = timelag
        self.folder = folder
    
    def generate(self, sample_name=None, save=True):
        '''
        sample_name is useful when generate many networks at one
        '''
        links = [1, 2, 3, 4, 5]
        weights = [0.25, 0.35, 0.32, 0.05, 0.03]
        genes = [f'G{i+1}' for i in range(self.size)]
        incomming_links = {}
        for gene in genes:
            candidates = copy.deepcopy(genes)
            candidates.remove(gene)           
            incomming_links[gene] = random.sample(candidates, random.choices(links, weights=weights, k=1)[0])
        
        updateTables = {target:{} for target in genes}
        # for target, regulators in incomming_links.items():
        #     updateTables[target] = self.getUpdateTable(len(regulators))
        # print(updateTables)
        goldstandard = []
        for target, regulators in incomming_links.items():
            for rg in regulators:
                goldstandard.append((rg, target, 1))
        for target, regulators in incomming_links.items():
            for g in genes:                
                if g!=target and g not in regulators:
                    goldstandard.append((g, target, 0))
        # Initialize the first timestep
        nodes = []
        for i in genes:
            nodes.append(Gene(i, [random.choice(self.bits)]))
        #update table
        for step in range(1, self.timestep_number):
            for node in nodes:
                key=''
                for regulator in incomming_links[node.name]:
                    for n in nodes:
                        if n.name==regulator:
                            key+=str(n.values[step-self.timelag])
                            break
                if key not in updateTables[node.name].keys():
                    
                    #val = random.choice(self.bits)
                    if len(updateTables[node.name].keys())==0:
                        val = random.choice(self.bits)
                    else:
                        vls = {i:0 for i in self.bits}
                        for i in updateTables[node.name].values():
                            vls[i]+=1
                        vls_reverse = dict(sorted(vls.items(), key=lambda x:x[1], reverse=False))
                        vls = dict(sorted(vls.items(), key=lambda x:x[1], reverse=True))
                        total = sum(vls.values())
                        probabilities = [i/total for i in vls.values()]
                        val = np.random.choice(list(vls_reverse.keys()), p=probabilities)
                    node.values.append(val)
                    updateTables[node.name][key] = val
                else:       
                    node.values.append(updateTables[node.name][key])
        # Save the result
        if save:
            goldstandard_path = f'{self.folder}/{sample_name}_goldstandard.txt'
            structure_path = f'{self.folder}/{sample_name}_structure.txt'
            updateTables_path = f'{self.folder}/{sample_name}_updateTables.txt'
            network_path = f'{self.folder}/{sample_name}_network.txt'
            with open(goldstandard_path, 'w+') as f:
                for regulator, target, reaction in goldstandard:
                    f.write(f'{regulator}\t{target}\t{reaction}\n')
            with open(structure_path, 'w+') as f:
                for target, regulators in incomming_links.items():
                    f.write(f'{target} <=== {regulators}\n')
            with open(updateTables_path, 'w+') as f:
                for target, table in updateTables.items():
                    f.write(f'*{target}===\n')
                    for k, v in table.items():
                        f.write(f'{k} ==> {v}\n')
                    f.write('=============\n')
        
        # Wrapping the result into network format
        network = GeneNetwork()
        network.nodes = nodes
        network.names = genes
        network.timesteps = [i for i in range(self.timestep_number)]
        network.goldstandard = goldstandard
        network.goldstandard_signed = None
        network.size = self.size
        network.timestepsNumber = self.timestep_number
        network.dict_names = {i:i for i in genes}
        network.folder = self.folder
        network.filename = sample_name
        if save:
            with open(network_path, 'w+') as f:
                f.write('#Time\t')
                for gene_name in network.names[:-1]:
                    f.write(f'{gene_name}\t')
                f.write(f'{network.names[-1]}\n')
                for step in range(network.timestepsNumber):
                    f.write(f'{network.timesteps[step]}\t')
                    for node in network.nodes[:-1]:
                        f.write(f'{node.values[step]}\t')
                    f.write(f'{network.nodes[-1].values[step]}\n')
        return network


    def getUpdateTable(self, incomming_number=1):
        bits_pairs = {}
        for i in range(len(self.bits)):
            for j in range(i+1, len(self.bits)):
                bits_pairs[str(self.bits[i])+str(self.bits[j])] = (self.bits[i], self.bits[j])
        pair_tables = []
        for k, v in bits_pairs.items():
            cols = []
            for i in range(incomming_number):
                if i==0:
                    unit = k
                else:
                    unit = k[0]*2**i+k[1]*2**i
                cols.append(unit*(2**(incomming_number-1-i)))
            pair_tables.append(cols)

        tables = []
        for i in range(incomming_number):
            col=''
            for t in pair_tables:
                col+=t[i]
            tables.append(col)

        keys = []
        for i in range(len(tables[0])):
            k=''
            for t in tables:
                k+=t[i]
            keys.append(k)
        keys = set(keys)
        updateTables = {key:random.choice(self.bits) for key in keys}
        return updateTables  

