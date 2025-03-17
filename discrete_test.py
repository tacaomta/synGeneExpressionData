from pipeline import Pipeline
import pandas as pd

#sizes=[10, 30, 50, 70, 100, 120, 150, 180, 200]
sizes=[10, 50, 100]
sps=[i for i in range(1,11, 1)]
timesteps = [(0, i) for i in range(10, 101, 10)]

for size in sizes:
    result = {}
    for sp in sps:
        experiment = {'timesteps': True, 'steps': timesteps}

        network_info = {'timeseries': fr'C:\caocao\gnw-master\tave_gen\simple test\extent_timesteps100\size{size}\ext_sample{sp}_network.txt',
                        'goldstandard': fr'C:\caocao\gnw-master\tave_gen\simple test\extent_timesteps100\size{size}\ext_sample{sp}_goldstandard.txt',
                        'goldstandard_signed': None,
                        'real_value': False,
                        'print_out': False}

        kmean_params = {}

        mibni_params = {'timelag': 1, 'K_max': 5, 'print_out': False}

        evaluation_params = {'print_out': True}

        precision, recall, structural, dynamics, sample = Pipeline(network_info, kmean_params, mibni_params, evaluation_params, experiment).execute()
        if len(sample)!=0:
            sample = pd.DataFrame(sample)
            sample.to_csv(fr'C:\caocao\gnw-master\tave_gen\simple test\extent_timesteps100\size{size}\analysis_step_sample{sp}.csv')    
        result[f'sample{sp}'] = {'precison': precision, 'recall': recall, 'structural': structural, 'dynamics':dynamics}
    result = pd.DataFrame(result)
    result.to_csv(fr'C:\caocao\gnw-master\tave_gen\simple test\extent_timesteps100\analysis_full_size{size}.csv')  

