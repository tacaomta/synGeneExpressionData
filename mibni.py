from utilities import directory_making
from evaluation import Evaluation


class MIBNI:
    '''
    MIBNI algorithm:
    K must be >=1
    '''

    def __init__(self, network, timelag=1, K=5) -> None:
        self.network = network
        self.timelag = timelag
        self.K = K

    def mifs_with_swap(self):
        evaluation = Evaluation(self.network, self.timelag)
        inferred_result = {}
        for target in self.network.nodes:
            if target.getEntropy(self.timelag)==0:
                inferred_result[(target.name, tuple(target.values))] = []                
            else:
                candidates = []
                firstMaxMI = -100000000
                # for g in self.network.nodes:
                #     if g.name==target.name:
                #         continue
                #     firstMaxMI = target.getMI(g, self.timelag)
                #     break
                firstSelectionCandidate = None
                for g in self.network.nodes:
                    if g.name==target.name:
                        continue
                    candidates.append(g)
                    currentMI = target.getMI(g, self.timelag)
                    if currentMI > firstMaxMI:
                        firstMaxMI = currentMI
                        firstSelectionCandidate = g.clone()
                #Remove first selected candidate from candidate list
                for c in candidates:
                    if c.name==firstSelectionCandidate.name:
                        candidates.remove(c)
                        break
                # Select the next candidates
                selection_regulators = [firstSelectionCandidate]
                consistency = evaluation.gene_wise_dynamics_consistency((target.name, tuple(target.values)), selection_regulators)
                if consistency!=1:
                    while len(selection_regulators)<self.K:
                        #print(f'TRước vòng lặp tìm next: {[i.name for i in selection_regulators]}')
                        nextScoreEstimation = target.getMI(candidates[0], self.timelag) - MIBNI.getMIofSandW(candidates[0], selection_regulators, timelag=self.timelag)
                        nextSelectionCandidate = candidates[0].clone()
                        for w in candidates:
                            currentNextScore = target.getMI(w, self.timelag) - MIBNI.getMIofSandW(w, selection_regulators, timelag=self.timelag)
                            if currentNextScore>nextScoreEstimation:
                                nextScoreEstimation = currentNextScore
                                nextSelectionCandidate = w.clone()
                        # Remove the next selected candidate from the candidate list
                        for c in candidates:
                            if c.name == nextSelectionCandidate.name:
                                candidates.remove(c)
                                break
                        # Add the next selected candilate to the selected regulator list
                        selection_regulators.append(nextSelectionCandidate)
                        # print(f'Sau khi tìm next, selection: {[i.name for i in selection_regulators]}')
                        # print(f'Sau khi tìm next, candidate: {[i.name for i in candidates]}')
                        #Swap will be added here
                        consistency = evaluation.gene_wise_dynamics_consistency((target.name, tuple(target.values)), selection_regulators)
                        if consistency==1:
                            break
                        else:
                            # Swap routine
                            emax, selection_regulators, candidates = self.swapping(target, selection_regulators, candidates)
                            if emax==1:
                                break
                inferred_result[(target.name, tuple(target.values))] = selection_regulators    
            print(f'Done: {target.name} ~ {self.network.dict_names[target.name]}...')
        self.inference = inferred_result
        return inferred_result 

    def swapping(self, target, selected, unselected):
        selected = sorted(selected, key=lambda x:target.getMI(x), reverse=True)
        unselected = sorted(unselected, key=lambda x:target.getMI(x), reverse=True)
        evaluation = Evaluation(self.network, self.timelag)
        eMax = evaluation.gene_wise_dynamics_consistency((target.name, tuple(target.values)), selected)
        for i in range(1, len(selected)):            
            for j in range(len(unselected)):                
                gene_swap_int = unselected[j]
                gene_swap_out = selected[i]
             
                selected.remove(gene_swap_out)            
                selected.insert(i, gene_swap_int)

                unselected.remove(gene_swap_int)
                unselected.insert(j, gene_swap_out)
                current_eMax = evaluation.gene_wise_dynamics_consistency((target.name, tuple(target.values)), selected)
                if current_eMax>eMax:
                    eMax = current_eMax
                else:
                    selected.remove(gene_swap_int)
                    selected.insert(i, gene_swap_out)

                    unselected.remove(gene_swap_out)
                    unselected.insert(j, gene_swap_int)
        return eMax, selected, unselected

    def mifs_without_swap(self):
        inferred_result = {}
        for target in self.network.nodes:
            if target.getEntropy(self.timelag)==0:
                inferred_result[(target.name, tuple(target.values))] = []                
            else:
                candidates = []
                firstMaxMI = -100000000
                # for g in self.network.nodes:
                #     if g.name==target.name:
                #         continue
                #     firstMaxMI = target.getMI(g, self.timelag)
                #     break
                firstSelectionCandidate = None
                for g in self.network.nodes:
                    if g.name==target.name:
                        continue
                    candidates.append(g)
                    currentMI = target.getMI(g, self.timelag)
                    #print(f'current = {currentMI}, max = {firstMaxMI}')
                    if currentMI > firstMaxMI:
                        firstMaxMI = currentMI
                        firstSelectionCandidate = g.clone()
                #Remove first selected candidate from candidate list
                for c in candidates:
                    if c.name==firstSelectionCandidate.name:
                        candidates.remove(c)
                        break
                # Select the next candidates
                selection_regulators = [firstSelectionCandidate]
                while len(selection_regulators)<self.K:
                    nextScoreEstimation = target.getMI(candidates[0], self.timelag) - MIBNI.getMIofSandW(candidates[0], selection_regulators, timelag=self.timelag)
                    nextSelectionCandidate = candidates[0].clone()
                    for w in candidates:
                        currentNextScore = target.getMI(w, self.timelag) - MIBNI.getMIofSandW(w, selection_regulators, timelag=self.timelag)
                        if currentNextScore>nextScoreEstimation:
                            nextScoreEstimation = currentNextScore
                            nextSelectionCandidate = w.clone()
                    # Remove the next selected candidate from the candidate list
                    for c in candidates:
                        if c.name == nextSelectionCandidate.name:
                            candidates.remove(c)
                            break
                    # Add the next selected candilate to the selected regulator list
                    selection_regulators.append(nextSelectionCandidate)
                #inferred_result[target.name] = [g.name for g in selection_regulators] 
                inferred_result[(target.name, tuple(target.values))] = selection_regulators   
            print(f'Done: {target.name} ~ {self.network.dict_names[target.name]}...')
        self.inference = inferred_result


        for k, v in inferred_result.items():
            print(k, [i.name for i in v])
        return inferred_result  

    def saveInference(self):
        inference_folder = directory_making(f"{self.network.folder}/mibni_output/inference/")
        with open(f'{inference_folder}/network.txt', 'w+') as f:
            for target, regulators in self.inference.items():
                rls = [i.name for i in regulators]
                f.write(f'{target[0]} <=== {rls} \n')
            f.write('==============with numerical name==================\n')
            for target, regulators in self.inference.items():
                rgl = [self.network.dict_names[r.name] for r in regulators]
                f.write(f'{self.network.dict_names[target[0]]} <=== {rgl} \n')
        
        with open(f'{inference_folder}/standard.txt', 'w+') as f:
            for regulator, target, reaction in self.prediction_set:
                f.write(f'{regulator}\t{target}\t{reaction} \n')
        print(f'The inference results are saved at: {inference_folder}')
           

    def getReferenceResult(self, virtual_name=True): 
        if virtual_name:
            for k, v in self.inference.items():
                regulators = [self.network.dict_names[i.name] for i in v]
                print(f'{self.network.dict_names[k[0]]} <-- {regulators}')                   
        else:
            for k, v in self.inference.items():
                regulators = [i.name for i in v]
                print(f'{k[0]} <-- {regulators}')   

    def getInferenceSummary(self, virtual_name=False):
        '''
        Hàm này để áp dụng cho thí nghiệm vote khi tái cấu trúc với nhiều mạng cùng một goldstandard
        để đưa ra lựa chọn chính xác dựa trên xác suất
        '''
        infer_summary = {}
        if virtual_name:
            for k, v in self.inference.items():
                regulators = [self.network.dict_names[i.name] for i in v]
                infer_summary[self.network.dict_names[k[0]]] = regulators               
        else:
            for k, v in self.inference.items():
                regulators = [i.name for i in v]
                infer_summary[k[0]] = regulators
        return infer_summary
    
    @staticmethod
    def getPredictionSummary(infer_summary, k=5, saved=True, path_save=None):
        '''
        Hàm này là kết quả đánh giá tổng hợp từ việc infer nhiều network có cùng một goldstandard
        '''
        summary = {k: [] for k in list(infer_summary.values())[0].keys()}
        for network_index, infer in infer_summary.items():
            for target, regulators in infer.items():
                summary[target] +=regulators
        
        result = {}
        for target, regulators in summary.items():
            candidates = set(regulators)
            statistic = {i:0 for i in candidates}
            for i in regulators:
                statistic[i]+=1
            statistic = dict(sorted(statistic.items(), key=lambda x:x[1], reverse=True))
            result[target] = statistic
        if saved and path_save is not None:
            with open(f'{path_save}', 'w+') as f:
                for target, stas in result.items():
                    f.write(f'{target} <= {str(stas)}\n')
        truncate = {}
        for target, stas in result.items():
            truncate[target] = [i for i in list(stas.keys())[:k]]

        full_none_prediction = []
        for reg in truncate.keys():
            for tar in truncate.keys():
                if reg==tar:
                    continue
                full_none_prediction.append((reg, tar, 0))
        
        prediction = []
        for k, v in truncate.items():
            for i in v:
                prediction.append((i, k, 1))
        negative_set = []
        for r2, t2, re2 in full_none_prediction:
            found=False
            for r1, t1, re1 in prediction:            
                if r1==r2 and t1==t2:
                    found=True
                    break
            if(not found):
                negative_set.append((r2, t2, re2))
        # Sort positve and negative prediction
        positive_pred_set = sorted(prediction, key=lambda x:x[0])
        negative_pred_set = sorted(negative_set, key=lambda x:x[0])
        prediction_set = positive_pred_set + negative_pred_set
        return prediction_set

    def getPrediction(self):
        '''
        Returns the reference result in the format as that in goldstandard file
        '''
        full_none_prediction = []
        for reg in self.network.nodes:
            for tar in self.network.nodes:
                if reg.name==tar.name:
                    continue
                full_none_prediction.append((reg.name, tar.name, 0))

        prediction = []
        for k, v in self.inference.items():
            for i in v:
                prediction.append((i.name, k[0], 1))
        negative_set = []
        for r2, t2, re2 in full_none_prediction:
            found=False
            for r1, t1, re1 in prediction:            
                if r1==r2 and t1==t2:
                    found=True
                    break
            if(not found):
                negative_set.append((r2, t2, re2))
        # Sort positve and negative prediction
        positive_pred_set = sorted(prediction, key=lambda x:x[0])
        negative_pred_set = sorted(negative_set, key=lambda x:[0])
        prediction_set = positive_pred_set + negative_pred_set
        self.prediction_set = prediction_set
        return prediction_set
            

    def getSignedPrediction(self):
        pass
    
    @staticmethod
    def getMIofSandW(w, S, timelag=1):
        '''
        Calculate sum of the multual information of w (next candidate) and S (selection list)
        '''
        info = 0
        for s in S:
            info+=s.getMI(w, timelag)
        return info


# from evaluation import Evaluation
# #mibni = MIBNI(r'C:\caocao\gnw-master\tave\size10\sample1\original\Ecoli10-1_dream4_timeseries.tsv')
# nwt = GeneNetwork(r'C:\caocao\gnw-master\tave\size10\sample1\csv\k_mean2\values\sp1_k2.txt',
#                   r'C:\caocao\gnw-master\tave\size10\sample1\original\Ecoli10-1_goldstandard.tsv', timeseries=False)
# mibni = MIBNI(network=nwt)
# mibni.mifs_without_swap()
# pred = mibni.getPrediction()
# statistic = Evaluation(nwt)
# statistic.evaluate(pred)