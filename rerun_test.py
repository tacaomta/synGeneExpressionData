from gen_network import Generation
from ae_prediction import Vt_prediction
from gene import GeneNetwork
import numpy as np
from pipeline import Pipeline
import pandas as pd

sizes = [100]
samples = range(10,11,1)

def experiment01(size, sample, epochs, lr, folder_network, folder_model, folder_loss, folder_network_synthetic, folder_performance):
    '''
    Huấn luyện autoencoder trên tập dữ liệu gốc để sinh thêm các step mới
    Lưu network vừa sinh ra
    Inffer với MIDNI

    '''
    result={}
    network = GeneNetwork(fr'{folder_network}\size{size}\sample{sample}_network.txt', 
                                 None, timeseries=False)
    steps = [(0,i) for i in range(10,101,10)]
    for start, end in steps:
        sub = network.getSubTimeStepsData(start, end)
        md_vt = Vt_prediction(sub)        
        md_vt.fit(epochs=epochs, lr=lr , model_save=fr'{folder_model}\ae_{size}_{sample}_0_{end}.h5',
                 loss_figure=fr'{folder_loss}\loss_{size}_{sample}_{start}_{end}_lr{lr}.png')
        
        syn_network = []
        syn_network.append(list(md_vt.v_t[0]))
        for i in md_vt.v_t_plus_1:
            syn_network.append(list(i.copy()))
        #Sinh ra các dòng dữ liệu mới và bổ sung vào mạng gốc
        for i in range(network.timestepsNumber - end):
            vt = np.reshape(np.array(syn_network[-1]), (-1,network.size))
            vt_plus = md_vt.predict(vt)
            vt_plus = np.abs(np.round(vt_plus))
            print(vt_plus)
            vt_plus = np.clip(vt_plus, 0, 2)
            vt_plus = vt_plus.astype(int)
            syn_network.append(list(vt_plus[0]))
        # Add the header and save to txt file
        for i, row in enumerate(syn_network):
            row.insert(0, i)
        syn_network = np.array(syn_network)
        path = fr'{folder_network_synthetic}\samp{sample}_0_{end}.txt'
        header = 'Time'
        for i in range(network.size):
            header+='\t'+f'G{i+1}'
        np.savetxt(path, syn_network, delimiter='\t', header=header, fmt='%s')
        # Kiểm tra với MIDNI mạng vừa mới sinh ra
        experiment = {'timesteps': False, 'steps': None}
        network_info = {'timeseries': fr'{folder_network_synthetic}\samp{sample}_0_{end}.txt',
                        'goldstandard': fr'{folder_network}\size{size}\sample{sample}_goldstandard.txt',
                        'goldstandard_signed': None,
                        'real_value': False,
                        'print_out': False}

        kmean_params = {}

        mibni_params = {'timelag': 1, 'K_max': 5, 'print_out': False}

        evaluation_params = {'print_out': True}

        p, r, st, dy, sp = Pipeline(network_info, kmean_params, mibni_params, evaluation_params, experiment).execute()
        result[f'st_0_{end}'] = {'precison': p, 'recall': r, 'structural': st, 'dynamics':dy}
    result1 = pd.DataFrame(result)
    result1.to_csv(fr'{folder_performance}\size{size}_samp{sample}.csv')
        
def experiment02():
    '''
    Chạy bộ dữ liệu các size với hàm experiment01
    '''
    folder_network = r'C:\caocao\gnw-master\tave_gen\new dataset\input'
    folder_model = r'C:\caocao\gnw-master\tave_gen\new dataset\trained_model'
    folder_loss = r'C:\caocao\gnw-master\tave_gen\new dataset\history'
    folder_performance = r'C:\caocao\gnw-master\tave_gen\new dataset\evaluation\synthetic'
    lr = 0.0001
    for size in sizes:
        folder_network_synthetic = fr'C:\caocao\gnw-master\tave_gen\new dataset\output\size{size}'
        for sample in samples:
            epochs = 1000 if size==10 else 2000 if size==50 else 10000
            experiment01(size, sample, epochs, lr, folder_network, folder_model, folder_loss, folder_network_synthetic, folder_performance)


def experiment03():
    '''
    Chạy MIDNI với bộ dữ liệu gốc, mạng 3 level discretization
    '''
    stps = [(0,i) for i in range(10,101,10)]
    for size in sizes:
        for sp in samples:
            experiment = {'timesteps': True, 'steps': stps}
            network_info = {'timeseries': fr'C:\caocao\gnw-master\tave_gen\new dataset\input\size{size}\sample{sp}_network.txt',
                            'goldstandard': fr'C:\caocao\gnw-master\tave_gen\new dataset\input\size{size}\sample{sp}_goldstandard.txt',
                            'goldstandard_signed': None,
                            'real_value': False,
                            'print_out': False}

            kmean_params = {}

            mibni_params = {'timelag': 1, 'K_max': 5, 'print_out': False}

            evaluation_params = {'print_out': True}

            precision, recall, structural, dynamics, sample = Pipeline(network_info, kmean_params, mibni_params, evaluation_params, experiment).execute()
            if len(sample)!=0:
                result = pd.DataFrame(sample)
                result.to_csv(fr'C:\caocao\gnw-master\tave_gen\new dataset\evaluation\origin\size{size}\samp{sp}.csv') 

def experiment04():
    '''
    Chạy MIDNI với bộ dữ liệu gốc, mạng 2 level discretization
    '''
    stps = [(0,i) for i in range(10,101,10)]
    for size in sizes:
        for sp in samples:
            experiment = {'timesteps': True, 'steps': stps}
            network_info = {'timeseries': fr'C:\caocao\gnw-master\tave_gen\new dataset\boolean\size{size}\sample{sp}_network.txt',
                            'goldstandard': fr'C:\caocao\gnw-master\tave_gen\new dataset\boolean\size{size}\sample{sp}_goldstandard.txt',
                            'goldstandard_signed': None,
                            'real_value': False,
                            'print_out': False}

            kmean_params = {}

            mibni_params = {'timelag': 1, 'K_max': 5, 'print_out': False}

            evaluation_params = {'print_out': True}

            precision, recall, structural, dynamics, sample = Pipeline(network_info, kmean_params, mibni_params, evaluation_params, experiment).execute()
            if len(sample)!=0:
                result = pd.DataFrame(sample)
                result.to_csv(fr'C:\caocao\gnw-master\tave_gen\new dataset\evaluation\boolean\origin\size{size}\samp{sp}.csv') 


def experiment05():
    '''
    Chạy bộ dữ liệu các size với hàm experiment01, bộ dữ liệu Boolean
    '''
    folder_network = r'C:\caocao\gnw-master\tave_gen\new dataset\boolean'
    folder_model = r'C:\caocao\gnw-master\tave_gen\new dataset\boolean\trained_model'
    folder_loss = r'C:\caocao\gnw-master\tave_gen\new dataset\boolean\history'
    folder_performance = r'C:\caocao\gnw-master\tave_gen\new dataset\evaluation\boolean\synthetic'
    lr = 0.0001
    for size in sizes:
        folder_network_synthetic = fr'C:\caocao\gnw-master\tave_gen\new dataset\boolean\synthetic networks\size{size}'
        for sample in samples:
            epochs = 1000 if size==10 else 2000 if size==50 else 10000
            experiment01(size, sample, epochs, lr, folder_network, folder_model, folder_loss, folder_network_synthetic, folder_performance)

experiment05()
            