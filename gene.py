from sklearn.metrics import mutual_info_score
from scipy.stats import entropy
import math
from path import Path
import os
import copy
import pandas as pd
import numpy as np
import random
import ast

class Gene:
    '''
    The information of a single gene
    It includes the header: ex: mglB and the expression values
    '''
    def __init__(self, name, values) -> None:
        self.name = name
        self.values = values
    
    def __str__(self) -> str:
        ev = ''
        for i in self.values:
            ev+=f'{i} '
        return f'{self.name}: {ev}'
    
    def clone(self):
        return Gene(self.name, self.values)
    
    def getEntropy(self, timelag=1):
        return entropy(self.values[timelag:], base=2)
    
    def getBasicEntropy(self, timelag = 1):
        '''
        Get entropy of base = 2
        '''
        return -sum(p * math.log2(p) for p in self.values[timelag:] if p != 0)
    
    def getMI(self, gene, timelag = 1):
        '''
        Get MI between two genes with the time lag
        self: target gene
        gene: candidate regulator
        Note: This function used only for desretized gene
        '''
        return mutual_info_score(self.values[timelag:], gene.values[:-timelag])
    


class GeneNetwork:
    '''
    Gene network structure that contains a list of Gene
    '''
    def __init__(self, path=None, goldstandard=None, goldstandard_signed=None, network_structure=None, update_table=None, timeseries=True, print_out=False) -> None:
        self.nodes, self.names, self.timesteps = GeneNetwork.readNetwork(path, timeseries)
        self.goldstandard = GeneNetwork.readGoldStandard(goldstandard)
        self.goldstandard_signed = GeneNetwork.readGoldStandard(goldstandard_signed)
        self.size = len(self.nodes)
        self.timestepsNumber = len(self.nodes[0].values) if len(self.nodes)!=0 else 0
        self.dict_names = {f'{self.names[i]}':f'G{i+1}' for i in range(self.size)}
        self.folder = os.path.dirname(path) if path is not None else ''
        self.filename = Path(path).stem if path is not None else ''
        self.incomming_links = GeneNetwork.readNetworkStructure(network_structure)
        self.updateTables = GeneNetwork.readUpdateTables(update_table)
        if print_out:
            self.display()

        
            
    def normalize(self, bits=(0, 1)):
        '''
        This function is called only when the input data generating by Autoencoder. Some values of gene are out of range
        so this function fixes, forces values into the range of discretization 
        '''
        for n in range(len(self.nodes)):
            for i in range(len(self.nodes[n].values)):
                if self.nodes[n].values[i] > bits[-1]:
                    self.nodes[n].values[i] = bits[-1]
                elif self.nodes[n].values[i] < bits[0]:
                    self.nodes[n].values[i] = bits[0]
                

    def readDiscreteNetwork(self, path, goldstandard):
        '''
        This function is used to load the discrete network: no header, no index
        the header and timestep will be added automatically after reading
        '''
        txt = []
        with open(path, 'r') as f:
            for x in f:
                arr = x[:-2].split("\t")
                txt.append(arr)
        self.size = len(txt[0])
        self.timestepsNumber = len(txt)
        self.nodes=[]
        for i in range(self.size):
            values=[txt[j][i] for j in range(self.timestepsNumber)]
            self.nodes.append(Gene(f'G{i+1}', values))
        self.names = [f'G{i+1}' for i in range(self.size)]
        self.timesteps = [i+1 for i in range(self.timestepsNumber)]
        self.folder = os.path.dirname(path) if path is not None else ''
        self.filename = Path(path).stem if path is not None else ''
        self.dict_names = {f'G{i+1}':f'G{i+1}' for i in range(self.size)}
        gd = []
        with open(goldstandard, "r") as f:
            for x in f:
                gd.append(tuple(x[:-1].split('\t')))
        gold = []
        for regulator, target, reaction in gd:
            gold.append((regulator, target))
    
    def getSubTimeStepsData(self, start=0, end=10):
        gnt = GeneNetwork()
        gnt.nodes = copy.deepcopy(self.nodes)
        gnt.names = copy.deepcopy(self.names)
        gnt.timesteps = copy.deepcopy(self.timesteps[start:end])
        gnt.goldstandard = copy.deepcopy(self.goldstandard)
        gnt.goldstandard_signed = copy.deepcopy(self.goldstandard_signed)
        gnt.dict_names = copy.deepcopy(self.dict_names)
        gnt.size = self.size
        gnt.timestepsNumber = len(gnt.timesteps)
        gnt.folder = self.folder
        gnt.filename = self.filename

        new_notes = []
        for node in self.nodes:
            new_notes.append(Gene(node.name, node.values[start:end]))
        gnt.nodes = new_notes
        return gnt
    
    def addNoise(self, rate=10):
        gnt = GeneNetwork()
        gnt.nodes = copy.deepcopy(self.nodes)
        gnt.names = copy.deepcopy(self.names)
        gnt.timesteps = copy.deepcopy(self.timesteps)
        gnt.goldstandard = copy.deepcopy(self.goldstandard)
        gnt.goldstandard_signed = copy.deepcopy(self.goldstandard_signed)
        gnt.dict_names = copy.deepcopy(self.dict_names)
        gnt.size = self.size
        gnt.timestepsNumber = len(gnt.timesteps)
        gnt.folder = self.folder
        gnt.filename = self.filename

        new_notes = []
        for node in self.nodes:
            new_notes.append(Gene(node.name, copy.deepcopy(node.values)))
        gnt.nodes = new_notes

        noises_account = int(gnt.size*gnt.timestepsNumber*rate/100)
        node_index = [i for i in range(self.size)]
        noises_in_each_nodes_index = [random.choice(node_index) for _ in range(noises_account)]

        val_list = [0, 1, 2]
        for node_modified_index in noises_in_each_nodes_index:
            val_noise_index = random.randint(0, gnt.timestepsNumber-1)
            true_value = gnt.nodes[node_modified_index].values[val_noise_index]
            noise_value_list = [i for i in val_list if i!=true_value]
            noise_value = random.choice(noise_value_list)
            gnt.nodes[node_modified_index].values[val_noise_index] = noise_value
        return gnt
    
    def to_csv(self, path=None):
        dict_nodes = {}
        for node in self.nodes:
            dict_nodes[node.name] = node.values
        df = pd.DataFrame(dict_nodes)
        if path==None:
            filename = f'{self.folder}\{self.filename}.csv'
        else: 
            filename = path
        df.to_csv(filename, header=None, index=None)
    
    def to_txt(self, path=None):
        network = []
        for i in range(self.timestepsNumber):
            steps = [i]
            for node in self.nodes:
                steps.append(node.values[i])
            network.append(steps)
        network = np.array(network)
        if path is None:
            filename = f'{self.folder}\{self.filename}.txt'
        else:
            filename = path
        header = 'Time'
        for i in range(self.size):
            header+='\t'+f'G{i+1}'
        np.savetxt(filename, network, delimiter='\t', header=header, fmt='%s')

    def to_latents(self, timelag=1):
        '''
        This function to convert network into latent input for variational autoencoder...
        '''
        latents = []
        for i in range(self.timestepsNumber-timelag):
            latent = []
            for node in self.nodes:
                latent.append(node.values[i])
            for node in self.nodes:
                latent.append(node.values[i+timelag])
            latents.append(latent)
        return np.array(latents)
    
    def get_vt_vt_plus_1(self, timelag=1):
        '''
        Chuyển dữ liệu network về dạng input v(t) cho ra v(t+1). Dữ liệu này dùng để training autoencoder 
        cho dự đoán bước tiếp theo của gene khi biết dữ liệu của bước trước đó
        '''
        vts = []
        vt_plus_timelag = []
        for i in range(self.timestepsNumber-timelag):
            vt=[]
            vt_plus=[]
            for node in self.nodes:
                vt.append(node.values[i])
                vt_plus.append(node.values[i+timelag])
            vts.append(vt)
            vt_plus_timelag.append(vt_plus)
        return np.array(vts), np.array(vt_plus_timelag)

    def getReadableGoldStandard(self):
        '''
        Reformat goldstand file in form that a list of regulator for each target gene a
        '''            
        gsd = {}
        for node in self.nodes:
            regulators = []
            for r, t, re in self.goldstandard:
                if node.name ==t and re=='1':
                    regulators.append(r)
            gsd[node.name] = regulators
        return gsd
    
    def display(self):
        print(f'#Time\t{self.getHeader()}')
        for i in range(self.timestepsNumber):
            print(self.timesteps[i], end='\t')
            for n in self.nodes:
                print(n.values[i], end='\t')
            print()                
        
    def clone(self):
        gnt = GeneNetwork()
        gnt.nodes = copy.deepcopy(self.nodes)
        gnt.names = copy.deepcopy(self.names)
        gnt.timesteps = copy.deepcopy(self.timesteps)
        gnt.goldstandard = copy.deepcopy(self.goldstandard)
        gnt.goldstandard_signed = copy.deepcopy(self.goldstandard_signed)
        gnt.dict_names = copy.deepcopy(self.dict_names)
        gnt.size = self.size
        gnt.timestepsNumber = self.timestepsNumber
        gnt.folder = self.folder
        gnt.filename = self.filename
        return gnt
    
    def compareTo(self, other: object) -> float:
        matrix = np.zeros((self.timestepsNumber, len(self.nodes)))
        differences = 0
        for i in range(len(self.nodes)):
            for j in range(self.timestepsNumber):
                if self.nodes[i].values[j]!=other.nodes[i].values[j]:
                    differences+=1
                    matrix[j, i] = 1
        return differences/(len(self.nodes)*(self.timestepsNumber)), matrix

    def getHeader(self):
        header = ''
        for i, n in enumerate(self.names):
            if i == 0:
                    header += n
            else:
                header += f'\t{n}'
        return header
    
    def getMIMatrix(self, timelag=1):
        '''
        Function returns the MI matrix of the network
        '''
        matrixMI = []
        for i, target in enumerate(self.nodes):
            row = []
            for j, regulator in enumerate(self.nodes):
                if i==j:
                    row.append(-1)
                else:
                    row.append(target.getMI(regulator, timelag))
            matrixMI.append(row)
        return matrixMI
    
    def extendTimeStep(self, timestep=10, timelag=1, bits=(0, 1, 2)):
        '''
        This function extends the network with cutains timestep more with timelag
        Arguments:
        - timestep: the number of timestep should be extended of the network
        - timelag: default value always set 1
        Returns:
        - A new network with updated settings
        '''
        for step in range(self.timestepsNumber, self.timestepsNumber+timestep):
            for node in self.nodes:
                key=''
                for regulator in self.incomming_links[node.name]:
                    for n in self.nodes:
                        if n.name==regulator:
                            key+=str(n.values[step-timelag])
                            break
                if key not in self.updateTables[node.name].keys():
                    val = random.choice(bits)
                    node.values.append(val)
                    self.updateTables[node.name][key] = val
                else:       
                    node.values.append(self.updateTables[node.name][key])        
        self.timesteps.extend([i for i in range(self.timestepsNumber, self.timestepsNumber+timestep)])
        self.timestepsNumber+=timestep

    def saveNetwork(self, folder, sample_name):
        '''
        Save the network and other settings
        '''
        # Save the result
        goldstandard_path = f'{folder}/{sample_name}_goldstandard.txt'
        structure_path = f'{folder}/{sample_name}_structure.txt'
        updateTables_path = f'{folder}/{sample_name}_updateTables.txt'
        network_path = f'{folder}/{sample_name}_network.txt'
        with open(goldstandard_path, 'w+') as f:
            for regulator, target, reaction in self.goldstandard:
                f.write(f'{regulator}\t{target}\t{reaction}\n')
        with open(structure_path, 'w+') as f:
            for target, regulators in self.incomming_links.items():
                f.write(f'{target} <=== {regulators}\n')
        with open(updateTables_path, 'w+') as f:
            for target, table in self.updateTables.items():
                f.write(f'*{target}===\n')
                for k, v in table.items():
                    f.write(f'{k} ==> {v}\n')
                f.write('=============\n')
        # Wrapping the result into network format
        with open(network_path, 'w+') as f:
            f.write('#Time\t')
            for gene_name in self.names[:-1]:
                f.write(f'{gene_name}\t')
            f.write(f'{self.names[-1]}\n')
            for step in range(self.timestepsNumber):
                f.write(f'{self.timesteps[step]}\t')
                for node in self.nodes[:-1]:
                    f.write(f'{node.values[step]}\t')
                f.write(f'{self.nodes[-1].values[step]}\n')

    @staticmethod
    def readNetwork(path, timeseries=True):
        '''
        Arguments:
        path: the localtion of the file
        timeseries: indicates whether the file is timeseries genereated by GWN tool or discretized(discretized by Kmean)
        Structural of the file:
        * file dream4_timeseries.tsv
            - line1: header
            - line2: space
            - line3-n: data
            - first column: time steps
        * discretized file
            - line1: header
            - line2-n: data
            - first column: time steps
        
        Outputs:
            1. list of Gene
            2. header list
            3. timesteps list
        '''
        if path==None:
            return [], [], []
        f = open(path, "r")
        txt = []
        nodes = []
        
        if timeseries:
            for x in f:
                arr = x[:-1].split("\t")
                txt.append(arr)
            f.close()
            header = txt[0][1:]
            timesteps = [int(txt[j][0]) for j in range(2, len(txt))]
            geneNumber = len(txt[2][1:])
            for i in range(geneNumber):            
                exp_val = []
                for j in range(2, len(txt)):
                    try:
                        exp_val.append(float(txt[j][i+1]))
                    except:
                        continue
                if len(exp_val) > 0:
                    nodes.append(Gene(header[i], exp_val))
        else:
            for x in f:
                arr = x[:-1].split("\t")
                txt.append(arr)
            f.close()
            header = txt[0][1:]            
            timesteps = [int(txt[j][0]) for j in range(1, len(txt))]
            geneNumber = len(txt[0][1:])
            for i in range(geneNumber):            
                exp_val = []
                for j in range(1, len(txt)):
                    try:
                        exp_val.append(int(txt[j][i+1]))
                    except:
                        continue
                if len(exp_val) > 0:
                    nodes.append(Gene(header[i], exp_val))
        return nodes, header, timesteps
    
    @staticmethod
    def readGoldStandard(path):
        '''
        Read the goldstandard file
        Returns:
        List of tuple that contains a pair of (regulator and target)
        '''
        if path==None:
            return None
        goldstandard = []
        with open(path, "r") as f:
            for x in f:
                goldstandard.append(tuple(x[:-1].split('\t')))
        return goldstandard
    
    @staticmethod
    def readNetworkStructure(path):
        '''
        Read the network structure
        Returns:
        Dictionary that contains target gene <---< list of regulators
        '''
        if path is None:
            return None
        incomming_link = {}
        with open(path, 'r') as f:
            for x in f:
                info = x.split('<===')
                incomming_link[info[0][0:-1]] = ast.literal_eval(info[1][1:-1])        
        return incomming_link
        
    @staticmethod
    def readUpdateTables(path):
        '''
        read the update table
        Returns:
        A dictionary that has structure {target gene: {a dictionary of combinations input and output}}
        '''
        if path is None:
            return None
        updateTables = {}
        with open(path, 'r') as f:
            for x in f:
                if '*G' in x:
                    key = x[1:-4]
                    updateTables[key] = []
                    rules = {}
                else:
                    if '====' in x: 
                        updateTables[key] = rules
                        continue
                    rules_break = x.split('==>')
                    rules[rules_break[0][:-1]] = int(rules_break[1][1:])
        return updateTables
# s=200
# sp=1   
# nwt = GeneNetwork(fr'C:\caocao\gnw-master\tave\size{s}\sample{sp}\original\Ecoli{s}-{sp}_dream4_timeseries.tsv', 
# fr'C:\caocao\gnw-master\tave\size{s}\sample{sp}\original\Ecoli{s}-{sp}_goldstandard.tsv')
#nwt = GeneNetwork(r'C:\caocao\gnw-master\tave_gen\size200\sample1_network.txt', timeseries=False)
#sub_nwt = nwt.getSubTimeStepsData()
#nwt.display()
#print(nwt.getMIMatrix(timelag=2))

# print(g.getEntropy2())