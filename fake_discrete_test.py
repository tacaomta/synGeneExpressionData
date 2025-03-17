from pipeline import Pipeline
import pandas as pd

#sizes=[10, 30, 50, 70, 100, 120, 150, 180, 200]
sizes=[10, 50, 100]
sps=[i for i in range(1, 11)]
timesteps = [10, 20, 30, 40, 50]

for size in sizes:    
    for st in timesteps:
        result = {}
        for sp in sps:        
            experiment = {'timesteps': False, 'steps': timesteps}

            network_info = {'timeseries': fr'C:\caocao\gnw-master\tave_gen\hybrid\s{size}\synthetic\s{st}\fake{size}_{st}st_{sp}.txt',
                            'goldstandard': fr'C:\caocao\gnw-master\tave_gen\hybrid\s{size}\sample1_goldstandard.txt',
                            'goldstandard_signed': None,
                            'real_value': False,
                            'print_out': False}

            kmean_params = {}

            mibni_params = {'timelag': 1, 'K_max': 5, 'print_out': False}

            evaluation_params = {'print_out': True}

            precision, recall, structural, dynamics, sample = Pipeline(network_info, kmean_params, mibni_params, evaluation_params, experiment).execute()
            print('precision = ', precision)
            if len(sample)!=0:
                sample = pd.DataFrame(sample)
                sample.to_csv(fr'C:\caocao\gnw-master\tave_gen\size{size}\analysis_step_sample{sp}.csv')    
            result[f'sample{sp}'] = {'precison': precision, 'recall': recall, 'structural': structural, 'dynamics':dynamics}
        result = pd.DataFrame(result)
        result.to_csv(fr'C:\caocao\gnw-master\tave_gen\hybrid\s{size}\synthetic\inference_st_{st}.csv')   

