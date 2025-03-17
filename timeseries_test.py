from pipeline import Pipeline
import pandas as pd

#sizes=[10, 30, 50, 70, 100, 120, 150, 180, 200]
sizes=[50]
sps=[i for i in range(1, 2)]
timesteps = [(0, 10), (0, 20), (0, 30), (0, 40), (0, 50)]

for size in sizes:
    result = {}
    m_max = 4 if size==10 else 5 if size<=50 else 8
    for sp in sps:
        experiment = {'timesteps': False, 'steps': timesteps}

        network_info = {'timeseries': fr'C:\caocao\gnw-master\model test\Ecoli50-1_dream4_timeseries.tsv',
                        'goldstandard': fr'C:\caocao\gnw-master\model test\Ecoli50-1_goldstandard.tsv',
                        'goldstandard_signed': fr'C:\caocao\gnw-master\model test\Ecoli10-1_goldstandard_signed.tsv',
                        'real_value': True,
                        'print_out': False}

        kmean_params = {'k': 2, 'save_with_header': True, 'graph': False, 'subtimestep':False, 'print_out': False}

        mibni_params = {'timelag': 1, 'K_max': m_max, 'print_out': True}

        evaluation_params = {'print_out': True}

        precision, recall, structural, dynamics, sample = Pipeline(network_info, kmean_params, mibni_params, evaluation_params, experiment).execute()
        if len(sample)!=0:
            sample = pd.DataFrame(sample)
            #sample.to_csv(fr'C:\caocao\gnw-master\tave\size{size}\sample{sp}\analysis.csv')    
        result[f'sample{sp}'] = {'precison': precision, 'recall': recall, 'structural': structural, 'dynamics':dynamics}
    result = pd.DataFrame(result)
    #result.to_csv(fr'C:\caocao\gnw-master\tave\size{size}\analysis.csv')    