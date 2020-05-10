import numpy as np
import os

class Member:

    def __init__(self,tf_idf, label = None,doc_id = None ):  

        self.tf_idf = tf_idf
        self.label = label
        self.doc_id  = doc_id
        

class Cluster:

    def __init__(self):
           self._centroid = None
           self._members = []
    def reset_member(self):
          self._members= []
    def add_member(self,member):
        self._members.append(member)
    def set_centroid(self,centroid):
        self._centroid = centroid

class K_mean:

    def __init__ (self ,number_clusters,seed_value ):
        self.number_clusters = number_clusters
        self.centroids = []
        self.all_data = []
        self._label_count = dict()
        self._cluster= [ Cluster() for i in range(self.number_clusters)]
        self.seed_value = seed_value
        return
    def sparse_to_dense(self, dict_size, tf_idf):
        vetor = [0.0 for i in range(dict_size)]
        list_of_words  = tf_idf.split()
        for word in list_of_words:
            word_id,value = word.split(':')
            word_id = int(word_id)
            value = float(value)
            vetor[word_id] = value
        return vetor
        
    def load_data():
        
        file_dict = open('dict_idf.txt','r')
        words = file_dict.read().splitlines()
        self.dict_size = len(words)


        file_tf_idf = open('tf_idf.txt','r')
        docs =  file_tf_idf.read().splitlines()
        for doc in docs :
            feature = doc.split('<ffff>')
            label = int(feature[0])
            doc_id = int(feature[1])
            if(self._label_count.get(label,None) == None):
                self._label_count[label] = 1
            else:
                self._label_count[label ]+=1
            tf_idf = self.sparse_to_dense(self.dict_size, feature[2])
            self.all_data.append(Member(tf_idf  = tf_idf,label = label, doc_id = doc_id ))
        

    def random_init (self, seed_value):
        np.random.seed(seed_value)
        
        khoang = np.array(range(len(self.all_data)))
        np.random.shuffle(khoang)
        
        for i in range(self.number_clusters):
            self.centroids.append(self.all_data[khoang[i]].tf_idf)
            self._cluster[i].set_centroid(self.centroids[i]) 
        
    def compute_similar (self, member, centroid):
        distance = 0
        
        for i in range(self.dict_size):
            distance+= (member.tf_idf[i] - centroid[i] )**2
        distance =np.sqrt(distance)
        return distance
    def select_cluster_for(self,member):
        min_similar = 99999999
        best_fit_cluster = None
        for cluster in self._cluster:
            similar = self.compute_similar(member,cluster._centroid)
            if(similar< min_similar ):
                min_similar = similar
                best_fit_cluster = cluster
        best_fit_cluster.add_member(member)
        return min_similar
    def update_centroid(self, cluster):
        members_tf_idf = [member.tf_idf for member in cluster._members]
        avg_member_tf_idf = np.mean(members_tf_idf,axis = 0)
        sqrt_sum = 1
        sqrt_sum = np.sqrt(np.sum(avg_member_tf_idf**2))
        new_centroid = np.array([value/sqrt_sum for value in avg_member_tf_idf])
        cluster.set_centroid(new_centroid)
        
    def stop_condition( self,criterion,threshold):
        criteria = ['centroid', 'similarity','max_iters']
        assert criterion in criteria
        if(criterion == 'max_iters') :
            if(self._iteration >= threshold):
                return True
            else:
                return False
        elif(criterion == 'centroid'):
            e_new= [cluster._centroid for cluster in self._cluster]
            e_new_minus_e = [centroid for centroid in e_new if centroid not in self.centroids]
            self.centroids = e_new
            if(len(e_new_minus_e)>= threshold):
                return True
            else:
                return False



        elif(criterion == 'similarity'):
            new_s_minus_s = self._new_s - self._s
            self._s = self._new_s
            if(new_s_minus_s >= threshold):
                return True
            else:
                return False

        
    def run(self,criterion,threshold):
        self.load_data('tf_idf.txt','dicf_idf.txt')
        self.random_init(self.seed_value)
        self._iteration = 0
        self._s = 0
        while True:
         #   print(self._iteration)
            for cluster in self._cluster:
                cluster.reset_member()
            self._new_s = 0
            for member in self.all_data:
                min_s = self.select_cluster_for(member)
                self._new_s += min_s
            for cluster in self._cluster:
                self.update_centroid(cluster)
            self._iteration +=1
            if(self.stop_condition(criterion, threshold)):
                break

    def purity(self):
        sum =0
        
        for cluster in self._cluster:
            member_label = [member.label for member in cluster._members]
            
            max_count =max( [member_label.count(label) for label in self._label_count.keys()] )
            sum += max_count
        return sum*1.0/len(self.all_data)
    def NMI(self):
        i_value, h_omega, h_c,n = 0., 0., 0.,len(self.all_data)
        for cluster in self._cluster:
            wk = len(cluster._members)*1.
            h_omega += - wk/n*np.log10(wk/n)
            member_labels = [member.label for member in cluster._members]
            for label in self._label_count.keys():
                wk_cj = member_labels.count(label)*1.0
                cj = self._label_count[label]
                i_value += wk_cj/n * np.log10(n*wk_cj/(wk*cj)*1e-12)
        for label in self._label_count.keys():
            cj = self._label_count[label]*1.0
            h_c += -cj /n*np.log10(cj/n)

        return i_value*2./(h_omega + h_c)

#nope = K_mean(2,1)
#nope.run('max_iters',3)




