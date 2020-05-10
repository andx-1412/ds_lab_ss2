from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
import numpy as np
def sparse_to_dense(dict_size, tf_idf):
        vetor = [0.0 for i in range(dict_size)]
        list_of_words  = tf_idf.split()
        for word in list_of_words:
            word_id,value = word.split(':')
            word_id = int(word_id)
            value = float(value)
            vetor[word_id] = value
        return np.array(vetor)
        
def load_data(filepath_tf_idf, filepath_vocab):
        file_dict= open(filepath_vocab,'r')
        file_tf_idf= open(filepath_tf_idf,'r')
        words = file_dict.read().splitlines()
        dict_size = len(words)
        data = []
        labels = []
        
        
        docs =  file_tf_idf.read().splitlines()
        for doc in docs :
            feature = doc.split('<ffff>')
            label = int(feature[0])
            doc_id = int(feature[1])
            labels.append(label)
          
               
            tf_idf = sparse_to_dense(dict_size, feature[2])
            data.append(tf_idf)
        return np.array(data), np.array(labels)
        



  
def clustering_with_kmean():
    data,label = load_data('tf_idf.txt','dict_idf.txt')
    
    print("_____________________________________________")
    kmeans = KMeans(n_clusters = len(set(label)), init ='random',n_init = 5, tol = 1e-3, random_state = 2018).fit(data)
    labels = kmeans.labels_
    
def mean_squared_accurate(predicted,expected):
    matches = np.equal(predicted,expected)
    accu = np.sum(matches.astype(float))/expected.shape[0]
    return accu
    
def clustering_with_svm():
    train_x, train_y = load_data('tf_idf.txt','dict_idf.txt')
    classifier = LinearSVC(C = 10.0, tol=0.001,verbose = True)

    classifier.fit(train_x,train_y)
    test_x, test_y = load_data('test_tf_idf.txt','dict_idf.txt')
    predict_y = classifier.predict(test_x)
    accurate=  mean_squared_accurate(predicted = predict_y, expected = test_y)
    
    return
clustering_with_svm()