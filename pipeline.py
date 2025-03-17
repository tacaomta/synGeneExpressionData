from gene import Gene, GeneNetwork
from mibni import MIBNI
from evaluation import Evaluation
from kmeans import Kmeans
import time

class Pipeline:
    '''
    It follows a pipe as:
    Input: a Gene Network
    1. Kmeans
    2. MIFS
    3. SWAP
    4. Evalulation
    Output: Precision, Recall, Structural, Dynamics Accuracy
    '''

    def __init__(self, networkinfo, kmeans, mibni, evaluation, experiment) -> None:
        self.network = networkinfo
        self.kmeans = kmeans
        self.mibni = mibni
        self.evaluation = evaluation
        self.experiment = experiment
    

    def execute(self):
        start_time = time.time()
        print('Reading the input network...')
        network = GeneNetwork(path=self.network['timeseries'],goldstandard=self.network['goldstandard'], 
                                  goldstandard_signed= self.network['goldstandard_signed'], timeseries=self.network['real_value'], 
                                  print_out=self.network['print_out'])
        print(f'The input network is read successfully: network size - {network.size}, filename - {network.filename}')
        if self.network['real_value']:
            print('The network is feeding to Kmeans...')
            k = Kmeans(network, self.kmeans['k'])
            network = k.execute(with_header=self.kmeans['save_with_header'], graph=self.kmeans['graph'], print_out=self.kmeans['print_out'])
            print('The input network is done with disretization...')

        print('The inference process is starting...')
        sample = {}
        if self.experiment['timesteps']:            
            for start, end in self.experiment['steps']:
                infering_network = network.getSubTimeStepsData(start, end)
                mibni = MIBNI(network=infering_network, timelag=self.mibni['timelag'], K=self.mibni['K_max'])
                inference = mibni.mifs_with_swap()
                pred = mibni.getPrediction()
                statistic = Evaluation(infering_network)
                precision, recall, structural = statistic.evaluate(pred, print_out=self.evaluation['print_out'])
                dynamics = statistic.dynamics(inference, print_out=True)
                sample[f'steps({start}, {end})'] = {'precison': precision, 'recall': recall, 'structural': structural, 'dynamics':dynamics}
                print(f'Done for experiment with timesteps from {start} to {end}')  
                
        mibni = MIBNI(network=network, timelag=self.mibni['timelag'], K=self.mibni['K_max'])
        inference = mibni.mifs_with_swap()
        pred = mibni.getPrediction()
        if self.mibni['print_out']:
            print('The inference results are shown as belows:')
            mibni.getReferenceResult(virtual_name=False)
            print('==============with numerical name==================')
            mibni.getReferenceResult(virtual_name=True)
        # For test
        # print('Goldstandard:')
        # for k, v in raw_network.getReadableGoldStandard().items():
        #     print(f'{k} <-- {v}')  
        # End
        mibni.saveInference()
        print('The network is successfully recontructed...')
        print('The evaluation process is going...')
        statistic = Evaluation(network)
        precision, recall, structural = statistic.evaluate(pred, print_out=self.evaluation['print_out'])
        dynamics = statistic.dynamics(inference, print_out=True)        
        end_time = time.time()
        print(f'The whole pipeline is completed (in {1000 * (end_time - start_time):.2f} milliseconds)...')
        
        return precision, recall, structural, dynamics, sample

