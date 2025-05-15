from gene_vae import GVAE
from gene import GeneNetwork
from pipeline import Pipeline
import pandas as pd
from tensorflow.keras.models import load_model
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
from gen_network import Generation
from ae_prediction import Vt_prediction
from gene_gan import GENEGAN


def experiment01(time_series_file, gold_standard_file, output_file):
    '''
    This experiment generates a network from an existing network.
    Args:
    time_series_file - Path of a time series file (input network)
    gold_standard_file - Path of a gold standard file
    output_file - Path of the output file.
    '''
    network = GeneNetwork(time_series_file, gold_standard_file ,None, False, print_out=False)
    vae = GVAE(network)
    #vae.fit(epochs=10000)
    vae.sample(output_file, rows=network.timestepsNumber)


def experiment02(time_series_file, gold_standard_file):
    '''
    Evaluate generated networks with MIDNI algorithms
    Args:
    time_series_file - Path of a generated network
    gold_standard_file - Path of a gold standard file
    '''
    
    experiment = {'timesteps': False, 'steps': None}
    network_info = {'timeseries': time_series_file,
                    'goldstandard': gold_standard_file,
                    'goldstandard_signed': None,
                    'real_value': False,
                    'print_out': False}

    kmean_params = {}

    mibni_params = {'timelag': 1, 'K_max': 5, 'print_out': False}

    evaluation_params = {'print_out': True}

    precision, recall, structural, dynamics, sample = Pipeline(network_info, kmean_params, mibni_params, evaluation_params, experiment).execute()
    print('precision = ', precision, 'recall = ', recall, 'structural = ', structural, 'dynamics = ',dynamics) 


def experiment03(time_series_folder, gold_standard_folder, output_folder):
    '''
    Run new experiment 1 on different datasets
    Args:
    time_series_folder - The folder where the timeseries files are located.
    gold_standard_folder - The folder where the gold standard files are located.
    output_folder - the output folder
    '''
    sizes=[i for i in range(10, 101, 10)]
    sps=[i for i in range(1,11, 1)]
    for size in sizes:
        for sp in sps:

            time_series_file = fr"{time_series_folder}\size{size}\sample{sp}_network.txt"
            gold_standard_file = fr"{gold_standard_folder}\size{size}\sample{sp}_goldstandard.txt"
            network = GeneNetwork(time_series_file, gold_standard_file ,None, False, print_out=False)
            outpath = fr'{output_folder}\size{size}\f_sample{sp}_network.txt'
            vae = GVAE(network)
            vae.fit(epochs=10000)
            vae.sample(outpath, rows=network.timestepsNumber)

def experiment04(time_series_folder, gold_standard_folder, output_folder):
    '''
    Evaluate generated networks with MIDNI algorithms
    Args:
    time_series_folder - The folder where the timeseries files are located.
    gold_standard_folder - The folder where the gold standard files are located.
    output_folder - the output folder
    '''
    #sizes=[i for i in range(10, 101, 10)]
    sizes = [10, 50, 100]
    sps=[i for i in range(1,11, 1)]
    for size in sizes:
        result = {}
        for sp in sps:
            experiment = {'timesteps': False, 'steps': None}
            network_info = {'timeseries': fr'{time_series_folder}\size{size}\sample{sp}.txt',
                            'goldstandard': fr'{gold_standard_folder}\size{size}\sample{sp}_goldstandard.txt',
                            'goldstandard_signed': None,
                            'real_value': False,
                            'print_out': False}

            kmean_params = {}

            mibni_params = {'timelag': 1, 'K_max': 5, 'print_out': False}

            evaluation_params = {'print_out': True}

            precision, recall, structural, dynamics, sample = Pipeline(network_info, kmean_params, mibni_params, evaluation_params, experiment).execute()
            result[f'sample{sp}'] = {'precision': precision, 'recall': recall, 'structural': structural, 'dynamics':dynamics}

        result = pd.DataFrame(result)
        result.to_csv(fr'{output_folder}\f_size{size}_analysis.csv')  


def experiment05(time_series_folder, gold_standard_folder, output_folder):
    '''
    This experiment is based on the input network, the model will be trained based on 
    different timesteps from 10, 20,...N and generate synthetic networks with 
    the same number of timesteps as the original network.
    Args:
    time_series_folder - The folder where the timeseries files are located.
    gold_standard_folder - The folder where the gold standard files are located.
    output_folder - the output folder
    '''
    sizes=[i for i in range(10, 101, 10)]
    sps=[i for i in range(1,11, 1)]
    stps = [(0,i) for i in range(10,51,10)]

    for size in sizes:
        for sp in sps:
            network = GeneNetwork(fr"{time_series_folder}\size{size}\sample{sp}_network.txt", 
                                fr"{gold_standard_folder}\size{size}\sample{sp}_goldstandard.txt",None, False, 
                                            print_out=False)
            for start, end in stps:
                sub_network = network.getSubTimeStepsData(start=start, end=end)
                sub_network.filename = f'{start}_{end}_{sub_network.filename}'

                outpath = fr'{output_folder}\size{size}\f_sample{sp}_{start}_{end}.txt'

                vae = GVAE(sub_network)
                vae.fit(epochs=10000)
                vae.sample(outpath, rows=network.timestepsNumber)

def experiment06(time_series_folder, gold_standard_folder, output_folder):
    '''
    Evaluate generated networks with MIDNI algorithms
    Args:
    time_series_folder - The folder where the timeseries files are located.
    gold_standard_folder - The folder where the gold standard files are located.
    output_folder - the output folder
    '''
    #sizes=[i for i in range(10, 101, 10)]
    sizes=[10, 50, 100]
    sps=[i for i in range(1,11, 1)]
    stps = [(0,i) for i in range(10,51,10)]
    for size in sizes:
        for sp in sps:
            result = {}
            for start, end in stps:
                experiment = {'timesteps': False, 'steps': None}
                network_info = {'timeseries': fr'{time_series_folder}\size{size}\syn_samp{sp}_0_{end}.txt',
                                'goldstandard': fr'{gold_standard_folder}\size{size}\sample{sp}_goldstandard.txt',
                                'goldstandard_signed': None,
                                'real_value': False,
                                'print_out': False}

                kmean_params = {}

                mibni_params = {'timelag': 1, 'K_max': 5, 'print_out': False}

                evaluation_params = {'print_out': True}

                precision, recall, structural, dynamics, sample = Pipeline(network_info, kmean_params, mibni_params, evaluation_params, experiment).execute()
                result[f'st_0_{end}'] = {'precision': precision, 'recall': recall, 'structural': structural, 'dynamics':dynamics}

            result = pd.DataFrame(result)
            result.to_csv(fr'{output_folder}\size{size}\newf_size{size}_samp{sp}_0_{end}.csv')  


def experiment07(time_series_folder, gold_standard_folder, output_folder):
    '''
    The experiment compares the generation of two networks from a trained model. One network is generated by predicting
    an original network, and the other network is generated by predicting a normal matrix of the same size as the original network.
    This experiment compares two random network generation methods.
    -------------------------------------
    Args:
    time_series_folder - The folder where the timeseries files are located.
    gold_standard_folder - The folder where the gold standard files are located.
    output_folder - the output folder
    '''
    size = 50
    sample = [i for i in range(1,11, 1)]
    for sp in sample:
        network = GeneNetwork(fr'{time_series_folder}\size{size}\sample{sp}_network.txt', 
                            fr'{gold_standard_folder}\size{size}\sample{sp}_goldstandard.txt', timeseries=False)
        autoencoder = load_model(fr'models/ae_model_50_sample{sp}_network.h5')
        # Predict the normal matrix
        generated_latent_points = np.random.normal(size=(50, network.size*2))
        synthetic_data = autoencoder.predict(generated_latent_points)
        synthetic_data = np.abs(np.round(synthetic_data))
        synthetic_data = np.clip(synthetic_data, 0, 2)
        synthetic_data = synthetic_data.astype(int)
        syn_network = []
        for i in range(len(synthetic_data)):
            syn_network.append(list(synthetic_data[i])[:network.size])
            if i==len(synthetic_data)-1:
                syn_network.append(list(synthetic_data[i])[network.size:])

        for i, row in enumerate(syn_network):
            row.insert(0, i)
        syn_network = np.array(syn_network)
        path = fr'{output_folder}\nw{size}_{sp}_normal.txt'
        header = 'Time'
        for i in range(network.size):
            header+='\t'+f'G{i+1}'
        np.savetxt(path, syn_network, delimiter='\t', header=header, fmt='%s')
        # Predict the original
        orginal_predict_synthetic_data = autoencoder.predict(network.to_latents())
        orginal_predict_synthetic_data = np.abs(np.round(orginal_predict_synthetic_data))
        orginal_predict_synthetic_data = np.clip(orginal_predict_synthetic_data, 0, 2)
        orginal_predict_synthetic_data = orginal_predict_synthetic_data.astype(int)
        syn_network = []
        for i in range(len(orginal_predict_synthetic_data)):
            syn_network.append(list(orginal_predict_synthetic_data[i])[:network.size])
            if i==len(orginal_predict_synthetic_data)-1:
                syn_network.append(list(orginal_predict_synthetic_data[i])[network.size:])

        for i, row in enumerate(syn_network):
            row.insert(0, i)
        syn_network = np.array(syn_network)
        path = fr'{output_folder}\nw{size}_{sp}_orginal.txt'
        np.savetxt(path, syn_network, delimiter='\t', header=header, fmt='%s')


def experiment08(time_series_folder, gold_standard_folder, output_folder):
    '''
    Run MIDNI with the generated networks from experiment 07
    -------------------------------------
    Args:
    time_series_folder - The folder where the timeseries files are located.
    gold_standard_folder - The folder where the gold standard files are located.
    output_folder - the output folder
    '''
    size = 50
    sps=[i for i in range(1,11, 1)]
    types =['normal', 'original']
    for t in types:
        result = {}
        for sp in sps:            
            experiment = {'timesteps': False, 'steps': None}
            network_info = {'timeseries': fr'{time_series_folder}\nw{size}_{sp}_{t}.txt',
                            'goldstandard': fr'{gold_standard_folder}\size{size}\sample{sp}_goldstandard.txt',
                            'goldstandard_signed': None,
                            'real_value': False,
                            'print_out': False}

            kmean_params = {}

            mibni_params = {'timelag': 1, 'K_max': 5, 'print_out': False}

            evaluation_params = {'print_out': True}

            precision, recall, structural, dynamics, sample = Pipeline(network_info, kmean_params, mibni_params, evaluation_params, experiment).execute()
            result[f'sample_{sp}'] = {'precision': precision, 'recall': recall, 'structural': structural, 'dynamics': dynamics}

        result = pd.DataFrame(result)
        result.to_csv(fr'{output_folder}\f_size{size}_{t}.csv')  

def experiment09(time_series_folder, gold_standard_folder, output_folder):
    '''
    Generate networks from models trained on timestep-restricted original datasets. 
    The generated network has the same timestep as the original one, without generating more.
    -------------------------------------
    Args:
    time_series_folder - The folder where the timeseries files are located.
    gold_standard_folder - The folder where the gold standard files are located.
    output_folder - the output folder
    '''
    sizes=[i for i in range(10, 101, 10)]
    sps=[i for i in range(1,11, 1)]
    stps = [(0,i) for i in range(10,51,10)]
    for size in sizes:
        for sp in sps:
            network = GeneNetwork(fr'{time_series_folder}\size{size}\sample{sp}_network.txt', 
                            fr'{gold_standard_folder}\size{size}\sample{sp}_goldstandard.txt', timeseries=False)
            
            for start, end in stps:
                autoencoder = load_model(fr'models/ae_model_{size}_0_{end}_sample{sp}_network.h5')
                generated_latent_points = network.getSubTimeStepsData(start=start, end=end).to_latents()
                synthetic_data = autoencoder.predict(generated_latent_points)

                synthetic_data = np.abs(np.round(synthetic_data))
                synthetic_data = np.clip(synthetic_data, 0, 2)
                synthetic_data = synthetic_data.astype(int)
                syn_network = []
                for i in range(len(synthetic_data)):
                    syn_network.append(list(synthetic_data[i])[:network.size])
                    if i==len(synthetic_data)-1:
                        syn_network.append(list(synthetic_data[i])[network.size:])

                for i, row in enumerate(syn_network):
                    row.insert(0, i)
                syn_network = np.array(syn_network)
                path = fr'{output_folder}\size{size}\syn_samp{sp}_{start}_{end}.txt'
                header = 'Time'
                for i in range(network.size):
                    header+='\t'+f'G{i+1}'
                np.savetxt(path, syn_network, delimiter='\t', header=header, fmt='%s')


def experiment10(time_series_folder, gold_standard_folder, output_folder):
    '''
    Testing the artificial life from the model trained on the original data sets, limited by the timestep. 
    The generated network has more timesteps than the original trained network. 
    The data is generated in a step-by-step manner. Use another trained autoencoder to generate each of these steps...
    -------------------------------------
    Args:
    time_series_folder - The folder where the timeseries files are located.
    gold_standard_folder - The folder where the gold standard files are located.
    output_folder - the output folder
    '''
    sizes=[10, 50, 100]
    sps=[i for i in range(1,11, 1)]
    stps = [(0,i) for i in range(10,51,10)]
    for size in sizes:
        for sp in sps:            
            network = GeneNetwork(fr'{time_series_folder}\size{size}\sample{sp}_network.txt', 
                            fr'{gold_standard_folder}\size{size}\sample{sp}_goldstandard.txt', timeseries=False)
            
            for start, end in stps:
                autoencoder = load_model(fr'models/ae_model_{size}_0_{end}_sample{sp}_network.h5')
                generated_latent_points = network.getSubTimeStepsData(start=start, end=end).to_latents()
                synthetic_data = autoencoder.predict(generated_latent_points)

                synthetic_data = np.abs(np.round(synthetic_data))
                synthetic_data = np.clip(synthetic_data, 0, 2)
                synthetic_data = synthetic_data.astype(int)
                syn_network = []
                for i in range(len(synthetic_data)):
                    syn_network.append(list(synthetic_data[i])[:network.size])
                    if i==len(synthetic_data)-1:
                        syn_network.append(list(synthetic_data[i])[network.size:])

                # print(syn_network[-1])
                # vt = np.array(syn_network[-1])
                # vt = np.reshape(vt, (-1,network.size))
                # print(vt)
                # Generate next steps using a trained autoencoder to predict the value of the next steps
                ae_vt = load_model(fr'\vt_step_models\ae_{size}_{sp}_{start}_{end}.h5')
                for i in range(network.timestepsNumber - end):
                    vt = np.reshape(np.array(syn_network[-1]), (-1,network.size))
                    vt_plus = ae_vt.predict(vt)
                    vt_plus = np.abs(np.round(vt_plus))
                    vt_plus = np.clip(vt_plus, 0, 2)
                    vt_plus = vt_plus.astype(int)
                    syn_network.append(list(vt_plus[0]))
                for i, row in enumerate(syn_network):
                    row.insert(0, i)
                syn_network = np.array(syn_network)
                path = fr'{output_folder}\size{size}\syn_samp{sp}_{start}_{end}.txt'
                header = 'Time'
                for i in range(network.size):
                    header+='\t'+f'G{i+1}'
                np.savetxt(path, syn_network, delimiter='\t', header=header, fmt='%s')

def experiment11():
    '''
    Từ các mạng dữ liệu đầu vào, module này add thêm tỷ lệ nhiễu để tạo ra các mạng mới,
    phục vụ cho các thử nghiệm trên các tập dữ liệu có nhiễu
    Tỷ lệ nhiễu thêm vào là 5%, 10%
    '''
    sizes=[10, 50, 100]
    sps=[i for i in range(1,11, 1)]
    noise_rates = [5, 10]
    stps = [(0,i) for i in range(10,51,10)]
    for size in sizes:
        for sp in sps:            
            network = GeneNetwork(fr'C:\caocao\gnw-master\tave_gen\simple test\size{size}\sample{sp}_network.txt', 
                            fr'C:\caocao\gnw-master\tave_gen\simple test\size{size}\sample{sp}_goldstandard.txt', timeseries=False)
            for start, end in stps:
                for noise_rate in noise_rates:
                    sub_network = network.getSubTimeStepsData(start, end)
                    sub_network = sub_network.addNoise(noise_rate)
                    sub_network.to_txt(fr'C:\caocao\gnw-master\tave_gen\simple test\withnoise\size{size}\sp{sp}_noise_{noise_rate}_{start}_{end}.txt')
        

def experiment12():
    '''
    Thí nghiệm kiểm tra thuật toán truy hồi đối với nhiều mạng được chia nhỏ và bổ sung vào đó các tỷ lệ nhiễu
    '''
    #sizes=[i for i in range(10, 101, 10)]
    sizes=[10, 50, 100]
    sps=[i for i in range(1,11, 1)]
    stps = [(0,i) for i in range(10,51,10)]
    noise_rates = [5, 10]
    for size in sizes:        
        for rate in noise_rates:
            result={}
            for sp in sps:            
                for start, end in stps:
                    experiment = {'timesteps': False, 'steps': None}
                    network_info = {'timeseries': fr'C:\caocao\gnw-master\tave_gen\simple test\withnoise\size{size}\sp{sp}_noise_{rate}_{start}_{end}.txt',
                                    'goldstandard': fr'C:\caocao\gnw-master\tave_gen\simple test\size{size}\sample{sp}_goldstandard.txt',
                                    'goldstandard_signed': None,
                                    'real_value': False,
                                    'print_out': False}

                    kmean_params = {}

                    mibni_params = {'timelag': 1, 'K_max': 5, 'print_out': False}

                    evaluation_params = {'print_out': True}

                    precision, recall, structural, dynamics, sample = Pipeline(network_info, kmean_params, mibni_params, evaluation_params, experiment).execute()
                    result[f'sample{sp}_rate{rate}_{start}_{end}'] = {'precison': precision, 'recall': recall, 'structural': structural, 'dynamics':dynamics}
            result = pd.DataFrame(result)
            result.to_csv(fr'C:\caocao\gnw-master\tave_gen\simple test\withnoise\size{size}\full_size{size}_noise_{rate}.csv')               

def experiment13():
    '''
    Thí nghiệm tạo ra các mô hình autoencoder được huấn luyện trên các mạng được cho thêm nhiễu vào
    '''
    sizes=[10, 50, 100]
    sps=[i for i in range(1,11, 1)]
    stps = [(0,i) for i in range(10,51,10)]
    noise_rates = [5, 10]
    for size in sizes:        
        for rate in noise_rates:
            for sp in sps:            
                for start, end in stps:
                    network = GeneNetwork(fr"C:\caocao\gnw-master\tave_gen\simple test\withnoise\size{size}\sp{sp}_noise_{rate}_{start}_{end}.txt", 
                                        None,None, False, print_out=False)
                    
                    epochs = 5000 if size==10 else 8000 if size==50 else 10000
                    path = fr'C:\caocao\gnw-master\tave_gen\simple test\withnoise\models\ae_{size}_samp{sp}_noise{rate}_{start}_{end}.h5'
                    vae = GVAE(network)
                    vae.fit(epochs=epochs, model_save=path)

def experiment14():
    '''
    Thử nghiệm sinh mạng giả từ mô hình đã được huấn luyện trên các tập dữ liệu
    gốc hạn chế theo timestep. Mạng sinh ra có timestep nhiều hơn so với mạng gốc được huấn luyện
    Dữ liệu sinh ra theo phương pháp từng step một. Dùng một autoencoder khác đã được huấn luyện để
    sinh ra từng step này...Mạng giả được sinh ra dựa trên mô hình đã được huấn luyện trên các
    tập dữ liệu có nhiễu (noise_rate = 5,10%)
    '''
    sizes=[10, 50, 100]
    sps=[i for i in range(1,11, 1)]
    stps = [(0,i) for i in range(10,51,10)]
    noise_rates = [5, 10]
    for size in sizes:
        for sp in sps:            
            for rate in noise_rates:            
                for start, end in stps:
                    network = GeneNetwork(fr'C:\caocao\gnw-master\tave_gen\simple test\withnoise\size{size}\sp{sp}_noise_{rate}_{start}_{end}.txt', 
                                None, timeseries=False)
                    autoencoder = load_model(fr'C:\caocao\gnw-master\tave_gen\simple test\withnoise\models\ae_{size}_samp{sp}_noise{rate}_{start}_{end}.h5')
                    generated_latent_points = network.to_latents()
                    synthetic_data = autoencoder.predict(generated_latent_points)

                    synthetic_data = np.abs(np.round(synthetic_data))
                    synthetic_data = np.clip(synthetic_data, 0, 2)
                    synthetic_data = synthetic_data.astype(int)
                    syn_network = []
                    for i in range(len(synthetic_data)):
                        syn_network.append(list(synthetic_data[i])[:network.size])
                        if i==len(synthetic_data)-1:
                            syn_network.append(list(synthetic_data[i])[network.size:])

                    # print(syn_network[-1])
                    # vt = np.array(syn_network[-1])
                    # vt = np.reshape(vt, (-1,network.size))
                    # print(vt)
                    # Sinh tiếp các bước tiếp theo sử dụng autoencoder đã được huấn luyện để dự đoán giá trị của những bước tiếp theo
                    ae_vt = load_model(fr'C:\caocao\gnw-master\tave_gen\simple test\withnoise\models_vt\ae_{size}_{sp}_noise_{rate}_{start}_{end}.h5')
                    for i in range(network.timestepsNumber - end):
                        vt = np.reshape(np.array(syn_network[-1]), (-1,network.size))
                        vt_plus = ae_vt.predict(vt)
                        vt_plus = np.abs(np.round(vt_plus))
                        vt_plus = np.clip(vt_plus, 0, 2)
                        vt_plus = vt_plus.astype(int)
                        syn_network.append(list(vt_plus[0]))
                    for i, row in enumerate(syn_network):
                        row.insert(0, i)
                    syn_network = np.array(syn_network)
                    path = fr'C:\caocao\gnw-master\tave_gen\simple test\withnoise\synthetic\size{size}\syn_samp{sp}_noise{rate}_{start}_{end}.txt'
                    header = 'Time'
                    for i in range(network.size):
                        header+='\t'+f'G{i+1}'
                    np.savetxt(path, syn_network, delimiter='\t', header=header, fmt='%s')

def experiment15():
    '''
    Thí nghiệm kiểm tra thuật toán truy hồi đối với nhiều mạng được giả được sinh ra từ các mạng gốc có nhiễu
    '''
    #sizes=[i for i in range(10, 101, 10)]
    sizes=[10, 50, 100]
    sps=[i for i in range(1,11, 1)]
    stps = [(0,i) for i in range(10,51,10)]
    noise_rates = [5, 10]
    for size in sizes:        
        for rate in noise_rates:
            result={}
            for sp in sps:            
                for start, end in stps:
                    experiment = {'timesteps': False, 'steps': None}
                    network_info = {'timeseries': fr'C:\caocao\gnw-master\tave_gen\simple test\withnoise\synthetic\size{size}\syn_samp{sp}_noise{rate}_{start}_{end}.txt',
                                    'goldstandard': fr'C:\caocao\gnw-master\tave_gen\simple test\size{size}\sample{sp}_goldstandard.txt',
                                    'goldstandard_signed': None,
                                    'real_value': False,
                                    'print_out': False}

                    kmean_params = {}

                    mibni_params = {'timelag': 1, 'K_max': 5, 'print_out': False}

                    evaluation_params = {'print_out': True}

                    precision, recall, structural, dynamics, sample = Pipeline(network_info, kmean_params, mibni_params, evaluation_params, experiment).execute()
                    result[f'sample{sp}_rate{rate}_{start}_{end}'] = {'precison': precision, 'recall': recall, 'structural': structural, 'dynamics':dynamics}
            result = pd.DataFrame(result)
            result.to_csv(fr'C:\caocao\gnw-master\tave_gen\simple test\withnoise\synthetic\size{size}\full_size{size}_noise_{rate}.csv')     


def experiment16():
    '''
    Thí nghiệm so sánh sự khác biệt giữa 2 mạng: mạng gốc và mạng giả
    Kết quả trả về là giá trị được đo bằng số lần khác biệt trên mỗi gen / tổng số lần giá trị trong mạng
    '''
    sizes=[10, 50, 100]
    sps=[i for i in range(1,11, 1)]
    stps = [(0,i) for i in range(10,51,10)]
    for size in sizes:   
        result = {}    
        for sp in sps:        
            original_network = GeneNetwork(fr'C:\caocao\gnw-master\tave_gen\simple test\size{size}\sample{sp}_network.txt', 
                                None, timeseries=False)  
            for start, end in stps:
                synthetic_network = GeneNetwork(fr'C:\caocao\gnw-master\tave_gen\simple test\ae_ct_network\size{size}\syn_samp{sp}_{start}_{end}.txt', 
                                None, timeseries=False)
                result[f'sp{sp}_{start}_{end}'], matrix = synthetic_network.compareTo(original_network)
                plt.clf()
                sns.heatmap(matrix, annot=False, cmap='YlGnBu')
                plt.title(f'Differences between [sp{sp}_{start}_{end}] - [sp{sp}]')
                plt.savefig(fr'C:\caocao\gnw-master\tave_gen\simple test\ae_ct_network\size{size}\differences\sp{sp}_{start}_{end}.png')
        
        result = pd.DataFrame(result, index=[0])
        result.to_csv(fr'C:\caocao\gnw-master\tave_gen\simple test\ae_ct_network\size{size}\differences\compare_detail.csv')                       



def experiment17():
    '''
    Thí nghiệm so sánh sự khác biệt giữa 2 mạng: mạng gốc và mạng giả
    Kết quả trả về là giá trị được đo bằng số lần khác biệt trên mỗi gen / tổng số lần giá trị trong mạng
    Thí nghiệm được thực hiện với mạng dữ liệu có nhiễu
    '''
    sizes=[10, 50, 100]
    sps=[i for i in range(1,11, 1)]
    stps = [(0,i) for i in range(10,51,10)]
    noise_rates = [5, 10]
    for size in sizes:           
        for rate in noise_rates:  
            result = {}  
            for sp in sps:                        
                for start, end in stps:
                    original_network = GeneNetwork(fr'C:\caocao\gnw-master\tave_gen\simple test\withnoise\size{size}\sp{sp}_noise_{rate}_{start}_{end}.txt', 
                                    None, timeseries=False)  
                    synthetic_network = GeneNetwork(fr'C:\caocao\gnw-master\tave_gen\simple test\withnoise\synthetic\size{size}\syn_samp{sp}_noise{rate}_{start}_{end}.txt', 
                                    None, timeseries=False)
                    result[f'sp{sp}_{start}_{end}'], matrix = synthetic_network.compareTo(original_network)
                    plt.clf()
                    sns.heatmap(matrix, annot=False, cmap='YlGnBu')
                    plt.title(f'Differences between [sp{sp}_{start}_{end}] with noise {rate}')
                    plt.savefig(fr'C:\caocao\gnw-master\tave_gen\simple test\withnoise\synthetic\size{size}\differences\sp{sp}_{start}_{end}_noise{rate}.png')
            
            result = pd.DataFrame(result, index=[0])
            result.to_csv(fr'C:\caocao\gnw-master\tave_gen\simple test\withnoise\synthetic\size{size}\differences\compare_detail_noise{rate}.csv')


########################## Điều chỉnh các tham số khi huấn luyện tập dữ liệu không có nhiễu #######################
def experiment18(size, sample, step, epochs):
    '''
    Điều chỉnh lại một số trường hợp được chọn
    '''

    network = GeneNetwork(fr"C:\caocao\gnw-master\tave_gen\simple test\size{size}\sample{sample}_network.txt", 
                        None,None, False, print_out=False)
    
    path = fr'C:\caocao\gnw-master\tave_gen\simple test\ae_ct_network\size{size}\adjustment\models\ae_{size}_samp{sample}_0_{step}.h5'
    vae = GVAE(network)
    vae.fit(epochs=epochs, model_save=path)

    autoencoder = load_model(path)
    generated_latent_points = network.to_latents()
    synthetic_data = autoencoder.predict(generated_latent_points)

    synthetic_data = np.abs(np.round(synthetic_data))
    synthetic_data = np.clip(synthetic_data, 0, 2)
    synthetic_data = synthetic_data.astype(int)
    syn_network = []
    for i in range(len(synthetic_data)):
        syn_network.append(list(synthetic_data[i])[:network.size])
        if i==len(synthetic_data)-1:
            syn_network.append(list(synthetic_data[i])[network.size:])
    for i, row in enumerate(syn_network):
        row.insert(0, i)
    syn_network = np.array(syn_network)
    path = fr'C:\caocao\gnw-master\tave_gen\simple test\ae_ct_network\size{size}\adjustment\networks\samp{sample}_{0}_{step}.txt'
    header = 'Time'
    for i in range(network.size):
        header+='\t'+f'G{i+1}'
    np.savetxt(path, syn_network, delimiter='\t', header=header, fmt='%s')

    # Evaluation
    experiment = {'timesteps': False, 'steps': None}
    network_info = {'timeseries': fr'C:\caocao\gnw-master\tave_gen\simple test\ae_ct_network\size{size}\adjustment\networks\samp{sample}_{0}_{step}.txt',
                    'goldstandard': fr'C:\caocao\gnw-master\tave_gen\simple test\size{size}\sample{sample}_goldstandard.txt',
                    'goldstandard_signed': None,
                    'real_value': False,
                    'print_out': False}

    kmean_params = {}

    mibni_params = {'timelag': 1, 'K_max': 5, 'print_out': False}

    evaluation_params = {'print_out': True}

    Pipeline(network_info, kmean_params, mibni_params, evaluation_params, experiment).execute()

    # Headmap
    synthetic_network = GeneNetwork(fr'C:\caocao\gnw-master\tave_gen\simple test\ae_ct_network\size{size}\adjustment\networks\samp{sample}_0_{step}.txt', 
                    None, timeseries=False)
    a, matrix = synthetic_network.compareTo(network)
    plt.clf()
    sns.heatmap(matrix, annot=False, cmap='YlGnBu')
    plt.title(f'Network distance')
    plt.savefig(fr'C:\caocao\gnw-master\tave_gen\simple test\ae_ct_network\size{size}\adjustment\differences\sp{sample}_0_{step}.png')

#experiment18(size=10, sample=2, step=50, epochs=100)
#experiment18(size=10, sample=4, step=30, epochs=100)
#experiment18(size=10, sample=4, step=40, epochs=100)
#experiment18(size=10, sample=4, step=50, epochs=100)

#experiment18(size=50, sample=3, step=50, epochs=500)
#experiment18(size=50, sample=2, step=50, epochs=500)
#experiment18(size=100, sample=10, step=50, epochs=500)

#experiment18(size=100, sample=1, step=50, epochs=400)
#experiment18(size=100, sample=3, step=50, epochs=200)



def experiment19():
    '''
    Thí nghiệm sinh ra tập dữ liệu có 100 timesteps
    '''
    size = [10, 50, 100, 200]
    sample = [i for i in range(1, 11, 1)]
    for si in size:
        for sam in sample:
            folder=fr'C:\caocao\gnw-master\tave_gen\simple test\timestep100\size{si}'
            gen = Generation(size=si,timestep_number=100, bits=(0, 1, 2), timelag=1, folder=folder)
            gen.generate(f'sample{sam}')

def experiment20():
    '''
    Thí nghiệm kiểm tra thuật toán truy hồi đối với nhiều mạng được sinh ra trên bộ dữ liệu 100 timestep gốc
    '''
    #sizes=[i for i in range(10, 101, 10)]
    sizes=[10, 50, 100, 200]
    sps=[i for i in range(1,11, 1)]
    stps = [(0,i) for i in range(10,101,10)]
    for size in sizes:
        for sp in sps:
            experiment = {'timesteps': True, 'steps': stps}
            network_info = {'timeseries': fr'C:\caocao\gnw-master\tave_gen\simple test\timestep100\size{size}\sample{sp}_network.txt',
                            'goldstandard': fr'C:\caocao\gnw-master\tave_gen\simple test\timestep100\size{size}\sample{sp}_goldstandard.txt',
                            'goldstandard_signed': None,
                            'real_value': False,
                            'print_out': False}

            kmean_params = {}

            mibni_params = {'timelag': 1, 'K_max': 5, 'print_out': False}

            evaluation_params = {'print_out': True}

            precision, recall, structural, dynamics, sample = Pipeline(network_info, kmean_params, mibni_params, evaluation_params, experiment).execute()
            #result[f'st_0_{end}'] = {'precison': precision, 'recall': recall, 'structural': structural, 'dynamics':dynamics}

            result = pd.DataFrame(sample)
            result.to_csv(fr'C:\caocao\gnw-master\tave_gen\simple test\timestep100\size{size}\sp{sp}.csv')  


def experiment21(size, sample, epochs):
    '''
    Thí nghiệm sinh giả mạng, tiến hành đánh giá trên bộ dữ liệu 100 timesteps
    '''
    result={}
    network = GeneNetwork(fr"C:\caocao\gnw-master\tave_gen\simple test\timestep100\size{size}\sample{sample}_network.txt",
                          fr'C:\caocao\gnw-master\tave_gen\simple test\timestep100\size{size}\sample{sample}_goldstandard.txt' 
                         ,None, False, print_out=False)
    for end in range(10, 101, 10):
        sub = network.getSubTimeStepsData(0, end)

        path = fr'C:\caocao\gnw-master\tave_gen\simple test\timestep100\models\ae_{size}_samp{sample}_0_{end}.h5'
        vae = GVAE(sub)
        vae.fit(epochs=epochs, model_save=path)

        autoencoder = load_model(path)
        generated_latent_points = sub.to_latents()
        synthetic_data = autoencoder.predict(generated_latent_points)

        synthetic_data = np.abs(np.round(synthetic_data))
        synthetic_data = np.clip(synthetic_data, 0, 2)
        synthetic_data = synthetic_data.astype(int)
        syn_network = []
        for i in range(len(synthetic_data)):
            syn_network.append(list(synthetic_data[i])[:network.size])
            if i==len(synthetic_data)-1:
                syn_network.append(list(synthetic_data[i])[network.size:])
        
        ae_vt = load_model(fr'C:\caocao\gnw-master\tave_gen\simple test\timestep100\vt_models\ae_{size}_{sample}_0_{end}.h5')
        for i in range(network.timestepsNumber - end):
            vt = np.reshape(np.array(syn_network[-1]), (-1,network.size))
            vt_plus = ae_vt.predict(vt)
            vt_plus = np.abs(np.round(vt_plus))
            vt_plus = np.clip(vt_plus, 0, 2)
            vt_plus = vt_plus.astype(int)
            syn_network.append(list(vt_plus[0]))
            
        for i, row in enumerate(syn_network):
            row.insert(0, i)
        syn_network = np.array(syn_network)
        path = fr'C:\caocao\gnw-master\tave_gen\simple test\timestep100\synthetics\size{size}\samp{sample}_0_{end}.txt'
        header = 'Time'
        for i in range(network.size):
            header+='\t'+f'G{i+1}'
        np.savetxt(path, syn_network, delimiter='\t', header=header, fmt='%s')

        # Evaluation
        experiment = {'timesteps': False, 'steps': None}
        network_info = {'timeseries': fr'C:\caocao\gnw-master\tave_gen\simple test\timestep100\synthetics\size{size}\samp{sample}_0_{end}.txt',
                        'goldstandard': fr'C:\caocao\gnw-master\tave_gen\simple test\timestep100\size{size}\sample{sample}_goldstandard.txt',
                        'goldstandard_signed': None,
                        'real_value': False,
                        'print_out': False}

        kmean_params = {}

        mibni_params = {'timelag': 1, 'K_max': 5, 'print_out': False}

        evaluation_params = {'print_out': True}

        p, r, st, dy, sp = Pipeline(network_info, kmean_params, mibni_params, evaluation_params, experiment).execute()
        result[f'st_0_{end}'] = {'precison': p, 'recall': r, 'structural': st, 'dynamics':dy}
        

        # Headmap
        synthetic_network = GeneNetwork(fr'C:\caocao\gnw-master\tave_gen\simple test\timestep100\synthetics\size{size}\samp{sample}_0_{end}.txt', 
                        None, timeseries=False)
        a, matrix = synthetic_network.compareTo(network)
        plt.clf()
        sns.heatmap(matrix, annot=False, cmap='YlGnBu')
        plt.title(f'Network distance')
        plt.savefig(fr'C:\caocao\gnw-master\tave_gen\simple test\timestep100\synthetics\size{size}\differences2\sp{sample}_0_{end}.png')
    result = pd.DataFrame(result)
    result.to_csv(fr'C:\caocao\gnw-master\tave_gen\simple test\timestep100\synthetics\size{size}\sp{sample}.csv') 

def experiment22():
    '''
    Thí nghiệm chạy hàng loạt hàm 21
    '''
    for size in [10, 50, 100, 200]:
        for samp in range(1,11, 1):
            epoch = 500 if size<50 else 1000 if size<100 else 5000 if size<200 else 10000
            experiment21(size, samp, epoch)


def experiment23(size, sample, epochs, lr):
    '''
    Thí nghiệm huấn luyện với mẫu tinh chỉnh cho mô hình dự đoán các bước step tiếp theo

    '''
    network = GeneNetwork(fr'C:\caocao\gnw-master\tave_gen\simple test\timestep100\size{size}\sample{sample}_network.txt', 
                                 None, timeseries=False)
    #size = [100]
    #sample = [i for i in range(7, 8, 1)]
    steps = [(0,i) for i in range(10,101,10)]
    for start, end in steps:
        sub = network.getSubTimeStepsData(start, end)
        md_vt = Vt_prediction(sub)
        #md_vt.fit(epochs=epochs, model_save=fr'C:\caocao\gnw-master\tave_gen\simple test\timestep100\vt_models\ae_{size}_{sample}_{start}_{end}.h5')
        md_vt.fit(epochs=epochs, lr=lr , model_save=None,
                  loss_figure=fr'C:\caocao\gnw-master\tave_gen\simple test\timestep100\history\loss_{size}_{sample}_{start}_{end}_lr{lr}.png')
        

def experiment24():
    '''
    Load network and extent its timesteps
    '''
    sizes=[10, 50, 100]
    sps=[i for i in range(1,11, 1)]
    for size in sizes:
        for sp in sps:
            network = GeneNetwork(path=fr'C:\caocao\gnw-master\tave_gen\simple test\size{size}\sample{sp}_network.txt',
                                goldstandard=fr'C:\caocao\gnw-master\tave_gen\simple test\size{size}\sample{sp}_goldstandard.txt',
                                goldstandard_signed=None,
                                network_structure=fr'C:\caocao\gnw-master\tave_gen\simple test\size{size}\sample{sp}_structure.txt',
                                update_table=fr'C:\caocao\gnw-master\tave_gen\simple test\size{size}\sample{sp}_updateTables.txt', timeseries=False)
            network.extendTimeStep(50)
            network.saveNetwork(fr'C:\caocao\gnw-master\tave_gen\simple test\extent_timesteps100\size{size}', sample_name=f'ext_sample{sp}')


def experiment25():
    '''
    Thí nghiệm kiểm tra thuật toán truy hồi đối với nhiều mạng được sinh ra bởi thuật toán sinh ra mạng.
    Các mạng này được coi là các mạng gốc
    '''
    #sizes=[i for i in range(10, 101, 10)]
    sizes=[10, 50, 100]
    sps=[i for i in range(1,11, 1)]
    stps = [(0,i) for i in range(10,101,10)]
    for size in sizes:
        for sp in sps:
            experiment = {'timesteps': True, 'steps': stps}
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
                result = pd.DataFrame(sample)
                result.to_csv(fr'C:\caocao\gnw-master\tave_gen\simple test\extent_timesteps100\size{size}\analysis_samp{sp}.csv')  
#experiment22()
#experiment23(100, 7, 1000, 0.0001)
#experiment21(100, 7, 20000)

def experiment26(size, sample, epochs, folder_network_input, folder_model1, folder_model2, folder_network_synthetic,
                 folder_distance, folder_performance, train_model1=True):
    '''
    Thí nghiệm sinh giả mạng, tiến hành đánh giá trên bộ dữ liệu 100 timesteps
    '''
    result={}
    network = GeneNetwork(path= fr"{folder_network_input}\size{size}\ext_sample{sample}_network.txt",
                          goldstandard=fr'{folder_network_input}\size{size}\ext_sample{sample}_goldstandard.txt', 
                         goldstandard_signed=None, timeseries=False, print_out=False)
    for end in range(10, 101, 10):
        sub = network.getSubTimeStepsData(0, end)

        path = fr'{folder_model1}\ae_{size}_samp{sample}_0_{end}.h5'
        if train_model1:
            vae = GVAE(sub)
            vae.fit(epochs=epochs, model_save=path)

        autoencoder = load_model(path)
        generated_latent_points = sub.to_latents()
        synthetic_data = autoencoder.predict(generated_latent_points)

        synthetic_data = np.abs(np.round(synthetic_data))
        synthetic_data = np.clip(synthetic_data, 0, 2)
        synthetic_data = synthetic_data.astype(int)
        syn_network = []
        for i in range(len(synthetic_data)):
            syn_network.append(list(synthetic_data[i])[:network.size])
            if i==len(synthetic_data)-1:
                syn_network.append(list(synthetic_data[i])[network.size:])
        
        ae_vt = load_model(fr'{folder_model2}\ae_{size}_{sample}_0_{end}.h5')
        for i in range(network.timestepsNumber - end):
            vt = np.reshape(np.array(syn_network[-1]), (-1,network.size))
            vt_plus = ae_vt.predict(vt)
            vt_plus = np.abs(np.round(vt_plus))
            vt_plus = np.clip(vt_plus, 0, 2)
            vt_plus = vt_plus.astype(int)
            syn_network.append(list(vt_plus[0]))
            
        for i, row in enumerate(syn_network):
            row.insert(0, i)
        syn_network = np.array(syn_network)
        path = fr'{folder_network_synthetic}\size{size}_samp{sample}_0_{end}.txt'
        header = 'Time'
        for i in range(network.size):
            header+='\t'+f'G{i+1}'
        np.savetxt(path, syn_network, delimiter='\t', header=header, fmt='%s')

        # Evaluation
        experiment = {'timesteps': False, 'steps': None}
        network_info = {'timeseries': fr'{folder_network_synthetic}\size{size}_samp{sample}_0_{end}.txt',
                        'goldstandard': fr'{folder_network_input}\size{size}\ext_sample{sample}_goldstandard.txt',
                        'goldstandard_signed': None,
                        'real_value': False,
                        'print_out': False}

        kmean_params = {}

        mibni_params = {'timelag': 1, 'K_max': 5, 'print_out': False}

        evaluation_params = {'print_out': True}

        p, r, st, dy, sp = Pipeline(network_info, kmean_params, mibni_params, evaluation_params, experiment).execute()
        result[f'st_0_{end}'] = {'precison': p, 'recall': r, 'structural': st, 'dynamics':dy}
        

        # Headmap
        synthetic_network = GeneNetwork(path=fr'{folder_network_synthetic}\size{size}_samp{sample}_0_{end}.txt', 
                        goldstandard=None, timeseries=False)
        a, matrix = synthetic_network.compareTo(network)
        plt.clf()
        sns.heatmap(matrix, annot=False, cmap='YlGnBu')
        plt.title(f'Network distance')
        plt.savefig(fr'{folder_distance}\size{size}_samp{sample}_0_{end}.png')
    result1 = pd.DataFrame(result)
    result1.to_csv(fr'{folder_performance}\size{size}_samp{sample}.csv')
    return result


def experiment27():
    '''
    chạy nhiều lần với hàm 26
    '''
    sizes=[10, 50, 100]
    sps=[i for i in range(1,11, 1)]
    for size in sizes:
        #result[f'st_0_{end}'] = {'precison': p, 'recall': r, 'structural': st, 'dynamics':dy}
        average = {f'st_0_{end}':{'precison': 0, 'recall': 0, 'structural': 0, 'dynamics':0} for end in range(10,101,10)}
        epochs = 1000 if size==10 else 2000 if size==50 else 5000
        for sp in sps:
            result = experiment26(size=size, sample=sp, epochs=epochs, 
                         folder_network_input=fr'C:\caocao\gnw-master\tave_gen\simple test\extent_timesteps100', 
                         folder_model1=fr'C:\caocao\gnw-master\tave_gen\simple test\extent_timesteps100\synthetic\model1', 
                         folder_model2=fr'C:\caocao\gnw-master\tave_gen\simple test\extent_timesteps100\synthetic\model2', 
                         folder_network_synthetic=fr'C:\caocao\gnw-master\tave_gen\simple test\extent_timesteps100\synthetic\networks',
                        folder_distance=fr'C:\caocao\gnw-master\tave_gen\simple test\extent_timesteps100\synthetic\distances', 
                        folder_performance=fr'C:\caocao\gnw-master\tave_gen\simple test\extent_timesteps100\synthetic\performance',
                        train_model1=False)
            for k, v in result.items():
                for subkey in v.keys():
                    average[k][subkey]+=v[subkey]
        for k, v in average.items():
            for subkey in v.keys():
                average[k][subkey]/=10
        average = pd.DataFrame(average)
        average.to_csv(fr'C:\caocao\gnw-master\tave_gen\simple test\extent_timesteps100\synthetic\performance\average_size{size}.csv')

def experiment28(size, sample, epochs, lr, folder_network, folder_model, folder_loss):
    '''
    Thí nghiệm huấn luyện với mẫu tinh chỉnh cho mô hình dự đoán các bước step tiếp theo trên dữ liệu extent

    '''
    network = GeneNetwork(fr'{folder_network}\size{size}\ext_sample{sample}_network.txt', 
                                 None, timeseries=False)
    steps = [(0,i) for i in range(10,101,10)]
    for start, end in steps:
        sub = network.getSubTimeStepsData(start, end)
        md_vt = Vt_prediction(sub)        
        md_vt.fit(epochs=epochs, lr=lr , model_save=fr'{folder_model}\ae_{size}_{sample}_0_{end}.h5',
                  loss_figure=fr'{folder_loss}\loss_{size}_{sample}_{start}_{end}_lr{lr}.png')

def experiment29():
    '''
    Chạy một lúc nhiều hàm 28
    '''
    sizes=[100]
    sps=[i for i in range(8,11, 1)]
    for size in sizes:
        epochs = 1000 if size==10 else 2000 if size==50 else 5000
        for sp in sps:
            experiment28(size, sp, epochs, lr=0.0005,folder_network=r'C:\caocao\gnw-master\tave_gen\simple test\extent_timesteps100', 
                         folder_model=r'C:\caocao\gnw-master\tave_gen\simple test\extent_timesteps100\synthetic\ model2', 
                         folder_loss=r'C:\caocao\gnw-master\tave_gen\simple test\extent_timesteps100\synthetic\history')



def experiment30():
    '''
    Ví dụ sinh network từ mạng GAN
    '''
    network = GeneNetwork(fr'C:\caocao\gnw-master\tave_gen\simple test\extent_timesteps100\size10\ext_sample4_network.txt', 
                                 goldstandard=None, timeseries=False)
    gene_gan = GENEGAN(network=network)
    gene_gan.fit(lr=0.0001, history_path=fr'C:\caocao\gnw-master\tave_gen\simple test\extent_timesteps100\gan_model\history\{network.filename}.png')
    gene_gan.sample(number_sample=50, path=fr'C:\caocao\gnw-master\tave_gen\simple test\extent_timesteps100\gan_model\networks\{network.filename}.txt')

#experiment29()
#experiment27()

#experiment28(100, 7, 5000, lr=0.0005,folder_network=r'C:\caocao\gnw-master\tave_gen\simple test\extent_timesteps100', 
#                         folder_model=r'C:\caocao\gnw-master\tave_gen\simple test\extent_timesteps100\synthetic\adjustment', 
#                         folder_loss=r'C:\caocao\gnw-master\tave_gen\sim ple test\extent_timesteps100\synthetic\adjustment')


# experiment26(size=100, sample=7, epochs=5000, 
#                          folder_network_input=fr'C:\caocao\gnw-master\tave_gen\simple test\extent_timesteps100', 
#                          folder_model1=fr'C:\caocao\gnw-master\tave_gen\simple test\extent_timesteps100\synthetic\model1', 
#                          folder_model2=fr'C:\caocao\gnw-master\tave_gen\simple test\extent_timesteps100\synthetic\adjustment', 
#                          folder_network_synthetic=fr'C:\caocao\gnw-master\tave_gen\simple test\extent_timesteps100\synthetic\adjustment',
#                         folder_distance=fr'C:\caocao\gnw-master\tave_gen\simple test\extent_timesteps100\synthetic\adjustment', 
#                         folder_performance=fr'C:\caocao\gnw-master\tave_gen\simple test\extent_timesteps100\synthetic\adjustment')

def experiment31():
    '''
    tạo bộ dữ liệu Bolean để test các thuật toán truy hồi Bolean network
    Bộ dữ liệu cũng gồm có 3 group size: 10, 50, 100 với timestep là 100 
    giống như bộ dữ liệu triple level
    '''
    size = [10, 50, 100]
    sample = [i for i in range(1, 11, 1)]
    for si in size:
        for sam in sample:
            folder=fr'C:\caocao\gnw-master\tave_gen\simple test\extent_timesteps100\boolean\size{si}'
            gen = Generation(size=si,timestep_number=100, bits=(0, 1), timelag=1, folder=folder)
            gen.generate(f'sample{sam}')

def experiment32():
    '''
    Truy hồi trên bộ dữ liệu Boolean gốc
    '''
    sizes=[100]
    sps=[i for i in range(10,11, 1)]
    stps = [(0,i) for i in range(10,101,10)]
    for size in sizes:
        for sp in sps:
            experiment = {'timesteps': True, 'steps': stps}
            network_info = {'timeseries': fr'C:\caocao\gnw-master\tave_gen\simple test\extent_timesteps100\boolean\size{size}\sample{sp}_network.txt',
                            'goldstandard': fr'C:\caocao\gnw-master\tave_gen\simple test\extent_timesteps100\boolean\size{size}\sample{sp}_goldstandard.txt',
                            'goldstandard_signed': None,
                            'real_value': False,
                            'print_out': False}

            kmean_params = {}

            mibni_params = {'timelag': 1, 'K_max': 5, 'print_out': False}

            evaluation_params = {'print_out': True}

            precision, recall, structural, dynamics, sample = Pipeline(network_info, kmean_params, mibni_params, evaluation_params, experiment).execute()
            if len(sample)!=0:
                result = pd.DataFrame(sample)
                result.to_csv(fr'C:\caocao\gnw-master\tave_gen\simple test\extent_timesteps100\boolean\size{size}\analysis_samp{sp}.csv') 


def experiment33():
    '''
    Huấn luyện model2 trên bộ dữ liệu boolean
    '''
    folder_model = r'C:\caocao\gnw-master\tave_gen\simple test\extent_timesteps100\boolean\vt_models'
    folder_loss = r'C:\caocao\gnw-master\tave_gen\simple test\extent_timesteps100\boolean\loss'
    lr = 0.0001
    sizes=[100]
    sps=[i for i in range(1,11, 1)]
    for size in sizes:
        epochs = 1000 if size==10 else 2000 if size==50 else 10000
        for sp in sps:
            network = GeneNetwork(fr'C:\caocao\gnw-master\tave_gen\simple test\extent_timesteps100\boolean\size{size}\sample{sp}_network.txt', timeseries=False)
            steps = [(0,i) for i in range(10,101,10)]
            for start, end in steps:
                sub = network.getSubTimeStepsData(start, end)
                md_vt = Vt_prediction(sub)        
                md_vt.fit(epochs=epochs, lr=lr , model_save=fr'{folder_model}\ae_{size}_{sp}_0_{end}.h5',
                        loss_figure=fr'{folder_loss}\loss_{size}_{sp}_{start}_{end}_lr{lr}.png')


def experiment34(size, sample, epochs, folder_network_input, folder_model1, folder_model2, folder_network_synthetic,
                 folder_distance, folder_performance, train_model1=True):
    '''
    Thí nghiệm sinh giả mạng, tiến hành đánh giá trên bộ dữ liệu 100 timesteps: Boolean networks
    '''
    result={}
    network = GeneNetwork(path= fr"{folder_network_input}\size{size}\sample{sample}_network.txt",
                          goldstandard=fr'{folder_network_input}\size{size}\sample{sample}_goldstandard.txt', 
                         goldstandard_signed=None, timeseries=False, print_out=False)
    for end in range(10, 101, 10):
        sub = network.getSubTimeStepsData(0, end)

        path = fr'{folder_model1}\ae_{size}_samp{sample}_0_{end}.h5'
        if train_model1:
            vae = GVAE(sub)
            vae.fit(epochs=epochs, model_save=path)

        autoencoder = load_model(path)
        generated_latent_points = sub.to_latents()
        synthetic_data = autoencoder.predict(generated_latent_points)

        synthetic_data = np.abs(np.round(synthetic_data))
        synthetic_data = np.clip(synthetic_data, 0, 1)
        synthetic_data = synthetic_data.astype(int)
        syn_network = []
        for i in range(len(synthetic_data)):
            syn_network.append(list(synthetic_data[i])[:network.size])
            if i==len(synthetic_data)-1:
                syn_network.append(list(synthetic_data[i])[network.size:])
        
        ae_vt = load_model(fr'{folder_model2}\ae_{size}_{sample}_0_{end}.h5')
        for i in range(network.timestepsNumber - end):
            vt = np.reshape(np.array(syn_network[-1]), (-1,network.size))
            vt_plus = ae_vt.predict(vt)
            vt_plus = np.abs(np.round(vt_plus))
            vt_plus = np.clip(vt_plus, 0, 1)
            vt_plus = vt_plus.astype(int)
            syn_network.append(list(vt_plus[0]))
            
        for i, row in enumerate(syn_network):
            row.insert(0, i)
        syn_network = np.array(syn_network)
        path = fr'{folder_network_synthetic}\size{size}_samp{sample}_0_{end}.txt'
        header = 'Time'
        for i in range(network.size):
            header+='\t'+f'G{i+1}'
        np.savetxt(path, syn_network, delimiter='\t', header=header, fmt='%s')

        # Evaluation
        experiment = {'timesteps': False, 'steps': None}
        network_info = {'timeseries': fr'{folder_network_synthetic}\size{size}_samp{sample}_0_{end}.txt',
                        'goldstandard': fr'{folder_network_input}\size{size}\sample{sample}_goldstandard.txt',
                        'goldstandard_signed': None,
                        'real_value': False,
                        'print_out': False}

        kmean_params = {}

        mibni_params = {'timelag': 1, 'K_max': 5, 'print_out': False}

        evaluation_params = {'print_out': True}

        p, r, st, dy, sp = Pipeline(network_info, kmean_params, mibni_params, evaluation_params, experiment).execute()
        result[f'st_0_{end}'] = {'precison': p, 'recall': r, 'structural': st, 'dynamics':dy}
        

        # Headmap
        synthetic_network = GeneNetwork(path=fr'{folder_network_synthetic}\size{size}_samp{sample}_0_{end}.txt', 
                        goldstandard=None, timeseries=False)
        a, matrix = synthetic_network.compareTo(network)
        plt.clf()
        sns.heatmap(matrix, annot=False, cmap='YlGnBu')
        plt.title(f'Network distance')
        plt.savefig(fr'{folder_distance}\size{size}_samp{sample}_0_{end}.png')
    result1 = pd.DataFrame(result)
    result1.to_csv(fr'{folder_performance}\size{size}_samp{sample}.csv')
    return result


def experiment35():
    '''
    chạy nhiều lần với hàm 34
    '''
    sizes=[100]
    sps=[i for i in range(1,11, 1)]
    for size in sizes:
        #result[f'st_0_{end}'] = {'precison': p, 'recall': r, 'structural': st, 'dynamics':dy}
        average = {f'st_0_{end}':{'precison': 0, 'recall': 0, 'structural': 0, 'dynamics':0} for end in range(10,101,10)}
        epochs = 1000 if size==10 else 2000 if size==50 else 5000
        for sp in sps:
           # if size==50 and sp<9: continue
            result = experiment34(size=size, sample=sp, epochs=epochs, 
                         folder_network_input=r'C:\caocao\gnw-master\tave_gen\simple test\extent_timesteps100\boolean', 
                         folder_model1=r'C:\caocao\gnw-master\tave_gen\simple test\extent_timesteps100\boolean\models', 
                         folder_model2=r'C:\caocao\gnw-master\tave_gen\simple test\extent_timesteps100\boolean\vt_models', 
                         folder_network_synthetic=r'C:\caocao\gnw-master\tave_gen\simple test\extent_timesteps100\boolean\synthetics\networks',
                        folder_distance=r'C:\caocao\gnw-master\tave_gen\simple test\extent_timesteps100\boolean\synthetics\distances', 
                        folder_performance=r'C:\caocao\gnw-master\tave_gen\simple test\extent_timesteps100\boolean\synthetics\performance',
                        train_model1=False)
            for k, v in result.items():
                for subkey in v.keys():
                    average[k][subkey]+=v[subkey]
        for k, v in average.items():
            for subkey in v.keys():
                average[k][subkey]/=10
        average = pd.DataFrame(average)
        average.to_csv(fr'C:\caocao\gnw-master\tave_gen\simple test\extent_timesteps100\boolean\synthetics\performance\average_size{size}.csv')

#
#experiment33()
#experiment35()
#experiment29()
experiment35()
