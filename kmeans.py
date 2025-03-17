import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gene import Gene
from utilities import directory_making

class Kmeans:

    def __init__(self, network, k=2, bits={1: '0', 2:'1'}, colormap={1:'r', 2:'g'}):
        self.colmap = colormap
        self.bits = bits
        self.network = network
        self.k = k

    def execute(self, with_header=True, graph=False, print_out=False):
        if graph:
            discretization_graph_path = directory_making(f"{self.network.folder}/mibni_output/clustering/graph/")
        else:
            discretization_graph_path = None
        discretization_gene_path = directory_making(f"{self.network.folder}/mibni_output/clustering/value/")
        result = pd.DataFrame({})
        for g in self.network.nodes:
            df = self.K_mean(g.values, self.k, discretization_graph_path, graph)
            result[g.name] = df['closest'].map(lambda j: self.bits[j])
        
        
        # Create a discretized network
        discretized_network = self.network.clone()
        # Update nodes for discretized network
        discretized_nodes = []   
        for g in self.network.nodes:
            discretized_nodes.append(Gene(g.name, [int(i) for i in result[g.name].values]))
        discretized_network.nodes = discretized_nodes
        # Save the discretized network
        outpath = '%s/%s_k2.txt' % (discretization_gene_path, self.network.filename)
        result.insert(0, 'Time', self.network.timesteps)
        if with_header:
            np.savetxt(outpath, result, delimiter='\t', header="Time\t"+self.network.getHeader(), fmt='%s')
        else:
            np.savetxt(outpath, result, delimiter='\t', fmt='%s')
        if print_out:
            print('Discretized network as follows:')
            discretized_network.display()
        print(f'The discretized network is saved at: {discretization_gene_path}')
        return discretized_network


    def K_mean(self, x,k, graph_path=None, graph=False):
        '''
        x: an instance of Gene
        '''

        maxValueOfThisGene = max(x)
        minValueOfThisGene = min(x)
        df = pd.DataFrame({
            'x': self.network.timesteps,
            'y': x
        })
        """
        centroids initialization step 
        """
        interval = (maxValueOfThisGene - minValueOfThisGene) / k
        centroids = {
            j + 1: [max(x) / 2, minValueOfThisGene + (j + 0.5) * interval]
            for j in range(k)
        }
        # print("Khỏi tạo ================")
        # print(centroids)

        """
        assignment of points to each centroids
        """
        df = self.assignment(centroids, df)
        """
        fit step
        """
        while True:
            closest_centroids = df['closest'].copy(deep=True)
            for j in centroids.keys():
                centroids[j][1] = np.mean(df[df['closest'] == j]['y'])
            df = self.assignment(centroids, df)
            if closest_centroids.equals(df['closest']):
                break
        if graph:
            plt.clf()
            plt.figure(figsize=(5, 5))
            plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')
            for i in centroids.keys():
                plt.scatter(*centroids[i], color=self.colmap[i])
            plt.xlim(0, self.time_max())
            plt.ylim(0, self.gene_max(x))
            plt.savefig(graph_path)
            print(f'The visualization is saved at: {graph_path}')
        #text_display = '%s  - save discretized figure  -  gene %d' % (self.path, index + 1)
        #print(text_display)
        return df

    def assignment(self, centroids, df):
        for j in centroids.keys():
            df['distance_from_{}'.format(j)] = (
                np.sqrt(
                    (df['y'] - centroids[j][1]) ** 2
                )
            )
            df['x_distance_from_{}'.format(j)] = (
                    (df['y'] - centroids[j][1]) ** 2
            )
        centroid_distance_cols = ['distance_from_{}'.format(k) for k in centroids.keys()]
        df['closest'] = df.loc[:, centroid_distance_cols].idxmin(axis=1)
        df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
        df['color'] = df['closest'].map(lambda x: self.colmap[x])
        return df


# nwt = GeneNetwork(r'C:\caocao\gnw-master\tave\size10\sample1\original\Ecoli10-1_dream4_timeseries.tsv')
# k = Kmeans(nwt, k=2)
# k.execute()

