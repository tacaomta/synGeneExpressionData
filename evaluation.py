from utilities import directory_making

class Evaluation:
    '''
    Evaluate the performance of the algorithm
    Arguments:
    - discretized network
    - goldstandard
    Returns:
    - precision
    - recall
    - structural accuracy
    - dynamics accuracy
    '''

    def __init__(self, network, timelage=1) -> None:
        self.network = network
        self.timelag = timelage

    def gene_wise_dynamics_consistency(self, target, regulators):    
        '''
        target = {name: values}
        regulators: list of Gene instances
        '''
        dyns = []           
        for i in range(self.network.timestepsNumber-self.timelag):
            consistency = ''  
            for regulator in regulators:
                consistency+=str(regulator.values[i])
            consistency+=str(target[1][i+self.timelag])
            dyns.append(consistency)
        unique_dyns = set(dyns)
        unique_dyns = list(unique_dyns)
        conflict = []
        for i in range(len(unique_dyns)-1):
            for j in range(i+1, len(unique_dyns)):
                if unique_dyns[i][:-1]==unique_dyns[j][:-1] and unique_dyns[i][-1]!=unique_dyns[j][-1]:
                    conflict.extend([unique_dyns[j], unique_dyns[i]])
        conflict=set(conflict)
        for cf in conflict:
            dyns.remove(cf)
        unique_dyns = set(dyns)
        dynamics_consistency = {i:0 for i in unique_dyns}
        for i in dyns:
            dynamics_consistency[i]+=1
        occurence = 0
        for occ in dynamics_consistency.values():
            occurence+=occ
        return occurence/(self.network.timestepsNumber-self.timelag)

    def dynamics(self, inference, print_out=True):
        gene_wise_dynamics_accuracy = []
        for target, regulators in inference.items():                 
            gene_wise_dynamics_accuracy.append(self.gene_wise_dynamics_consistency(target, regulators))
        dynamics_accuracy = sum(gene_wise_dynamics_accuracy)/self.network.size
        if print_out:
            print(f'Dynamics Accuracy = {dynamics_accuracy:.4f}')
        return dynamics_accuracy
    
    def evaluate(self, prediction, print_out=True):
        true_positive_set = []
        false_positive_set = []
        true_negative_set = []
        false_negative_set = []
        for regulator, target, reaction in self.network.goldstandard:
            for r, t, re in prediction:
                if regulator==r and target==t:
                    if reaction==str(re):
                        if re==1:
                            true_positive_set.append((r, t, re))
                        else:
                            true_negative_set.append((r, t, re))
                    else:
                        if re==0:
                            false_negative_set.append((r, t, re))
                        else:
                            false_positive_set.append((r, t, re))
                    break
        TP = len(true_positive_set)
        FP = len(false_positive_set)
        TN = len(true_negative_set)
        FN = len(false_negative_set)
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        structural = (TP+TN)/(TP+FP+TN+FN)
        if print_out:
            print(f'Precision = {precision:.4f}, recall = {recall:.4f}, structural = {structural:.4f}')
        evalulation_folder = directory_making(f"{self.network.folder}/mibni_output/evaluation/")
        with open(f'{evalulation_folder}/statistics.txt', 'w+') as f:
            f.write(f'Precision = {precision}\n')
            f.write(f'Recall = {recall}\n')
            f.write(f'Structural = {structural}\n')
            f.write(f'====== Details {TP+FP} predictions ====\n')
            f.write(f'- True positive predictions ({TP}):\n')
            for r, t, re in true_positive_set:
                f.write(f'{r} => {t}: {re}\n')
            f.write(f'- False positive predictions ({FP}):\n')
            for r, t, re in false_positive_set:
                f.write(f'{r} => {t}: {re}\n')
            f.write(f'- True negative predictions ({TN}):\n')
            for r, t, re in true_negative_set:
                f.write(f'{r} => {t}: {re}\n')
            f.write(f'- False negative predictions ({FN}):\n')
            for r, t, re in false_negative_set:
                f.write(f'{r} => {t}: {re}\n')
        print(f'The evaluation details are saved at: {evalulation_folder}')
        return precision, recall, structural
        
        


