'''
Build Text Mining Methods (for personal use)

TODOs:
    - Quick plotting methods
    - Method to incorporate N-grams
    - Build tidytext emulator (to weild text data in that way)
'''

from pandas import read_csv, DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import datetime as dt
import re

class topicModel:
    '''
    Parent text class for running text models.
    '''

    def __init__(self,data=None,text_var = "", no_features=5000):
        """Initialization method the the topicModel class. Generic use case class
        for processing text data efficiently in Python.

        Parameters
        ----------
        data : Pandas Data Frame
            Pandas data frame
        text_variable : str
            Text variable name.
        no_features : int
            Number of text features to extract.
        """
        self.data = data
        self.text = self.data[text_var].values.tolist()
        self.no_features = no_features
        self.stop_words = read_csv("Data/stop_words.csv").word.values.tolist()
        self.is_vectorized = False


    def tokenize(self,text):
        '''
        Quick tokenization method that only extracts letters, dropping all other naming convention issues.
        '''
        return ["".join(re.split("[^a-zA-Z]*", word)) for word in text.lower().split(' ')
                if len("".join(re.split("[^a-zA-Z]*", word))) > 2 and
                ("".join(re.split("[^a-zA-Z]*", word)) not in self.stop_words)]

    def exclude_words(self,word_list):
        '''
        Add words to exclude to the stop word list.
        '''
        self.stop_words.extend([w.lower().strip() for w in word_list])


    def word_count(self,n=25):
        '''
        Generate word counts from fit text as data.
        '''
        results = DataFrame(self.vector_fit.toarray(), columns=self.feature_names)
        return results.apply(sum).sort_values(ascending=False).iloc[:n].reset_index().rename( columns={0:self.vec_type})


    def display_topics(self,n_words=10):
        '''
        Print display topics output from the main model.
        '''
        for topic_idx, topic in enumerate(self.model_output.components_):
            print(f"Topic {topic_idx+ 1}")
            print("\t"," ".join([self.feature_names[i] for i in topic.argsort()[:-n_words - 1:-1]]),end="\n\n")


    def export_topics(self,file_name="",n_words=10):
        '''
        Export topic listing to a text file for independent review.
        '''
        with open(file_name,mode='wt',encoding='UTF-8') as file:
            file.writelines(f'Ran Non-negative Matrix Factorization where k = {self.model_output.components_.shape[0]}\n')
            file.writelines(f'Topics generated on {dt.datetime.now().strftime("%d/%m/%Y %H:%M:%S")} \n\n')
            for topic_idx, topic in enumerate(self.model_output.components_):
                file.writelines(f"Topic {topic_idx+ 1}:\n\t\t --> ")
                file.writelines(" ".join([self.feature_names[i] for i in topic.argsort()[:-n_words - 1:-1]]))
                file.writelines("\n\n")

    def topic_assignments(self,auto_name = False):
        """Return the topic assignments back to original data structure.

        Parameters
        ----------
        auto_name : boolean
            Should the topics be auto named with the top two most prevalent words assigned to the topic.

        Returns
        -------
        DataFrame
            Returns the original input data frame with the topic entries mapped on.
        """
        # Name the topics
        var_names = []
        for topic_idx, topic in enumerate(self.model_output.components_):
            if auto_name:
                var_names.append("_".join([self.feature_names[i] for i in topic.argsort()[:-2 - 1:-1]]))
            else:
                var_names.append(f"topic_{topic_idx+1}")


        # Topic assignments to documents
        W = self.model_output.transform(self.vector_fit)

        # Weights as proportions of all topical categories.
        W_prop = W/W.sum(axis=1).reshape(W.shape[0],1)
        W_dat = DataFrame(W_prop,columns=var_names)

        # Map to data
        self.data['id_'] = [i+1 for i in range(self.data.shape[0])]
        W_dat['id_'] = [i+1 for i in range(self.data.shape[0])]

        # Return original data with the topic assignment weights.
        return self.data.merge(W_dat,on="id_").drop(columns="id_")


class lda_model(topicModel):
    '''
    Class for processing and preparing data to run an Latent Latent Dirichlet Allocation model on.
    '''

    def __init__(self,data=None,text_var = "",no_features=5000):
        topicModel.__init__(self,data,text_var,no_features)

    def vectorize(self):
        '''
        Vectorize into document term matrix for processing.
        '''
        self.vectorizer = CountVectorizer(max_df = 0.95,
                                          min_df = 2,
                                          max_features = self.no_features,
                                          stop_words = "english",
                                          tokenizer=self.tokenize)
        self.vector_fit = self.vectorizer.fit_transform(self.text)
        self.feature_names = self.vectorizer.get_feature_names()
        self.vec_type = "word_count"
        self.is_vectorized = True

    def fit(self,n_topics = 5,random_state=1988):
        '''
        Fit an LDA model.
        '''
        if self.is_vectorized == False:
            print('Vectorizing text.')
            self.vectorize()
        print('Fitting model.')
        self.model_output = LatentDirichletAllocation(n_components=n_topics,
                                                 max_iter=5,
                                                 learning_method='online',
                                                 learning_offset=50.,
                                                 random_state=random_state).fit(self.vector_fit)
        print("Model fit.")


class nmf_model(topicModel):
    '''
    Non-negative Matrix Factorization Topic Model. Uses the Inverse Document Term
    Frequency (TFIDF) rather than term frequencies. This results in a topic model
    that emphasizes the differences _between_ documents rather than the commonalities.
    '''

    def __init__(self,data=None,text_var = "",no_features=5000):
        topicModel.__init__(self,data,text_var,no_features)

    def vectorize(self):
        '''
        Vectorize into document term matrix for processing.
        '''
        self.vectorizer = TfidfVectorizer(max_df=0.95,
                                           min_df=2,
                                           max_features = self.no_features,
                                           stop_words = "english",
                                           tokenizer=self.tokenize)
        self.vector_fit = self.vectorizer.fit_transform(self.text)
        self.feature_names = self.vectorizer.get_feature_names()
        self.vec_type = "tf_idtf"
        self.is_vectorized = True

    def fit(self,n_topics = 5,random_state=1988):
        '''
        Fit an Non-Negative Matrix Factorization model.
        '''
        if self.is_vectorized == False:
            print('Vectorizing text.')
            self.vectorize()
        print('Fitting model.')
        self.model_output = NMF(n_components=n_topics,
                           random_state=1988,
                           alpha=.1,
                           l1_ratio=.5,
                           init='nndsvd').fit(self.vector_fit)
        print("Model fit.")
