'''
Build Text Mining Methods (for personal use)

TODOs:
    - Quick plotting methods
    - Method to incorporate N-grams
    - Build tidytext emulator (to weild text data in that way)
'''

from pandas import DataFrame
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
        self.is_vectorized = False
        self.stop_words = ["a","a's","able","about","above","according","accordingly","across","actually","after","afterwards","again","against","ain't","all","allow","allows","almost","alone","along","already","also","although","always","am","among","amongst","an","and","another","any","anybody","anyhow","anyone","anything","anyway","anyways","anywhere","apart","appear","appreciate","appropriate","are","aren't","around","as","aside","ask","asking","associated","at","available","away","awfully","b","be","became","because","become","becomes","becoming","been","before","beforehand","behind","being","believe","below","beside","besides","best","better","between","beyond","both","brief","but","by","c","c'mon","c's","came","can","can't","cannot","cant","cause","causes","certain","certainly","changes","clearly","co","com","come","comes","concerning","consequently","consider","considering","contain","containing","contains","corresponding","could","couldn't","course","currently","d","definitely","described","despite","did","didn't","different","do","does","doesn't","doing","don't","done","down","downwards","during","e","each","edu","eg","eight","either","else","elsewhere","enough","entirely","especially","et","etc","even","ever","every","everybody","everyone","everything","everywhere","ex","exactly","example","except","f","far","few","fifth","first","five","followed","following","follows","for","former","formerly","forth","four","from","further","furthermore","g","get","gets","getting","given","gives","go","goes","going","gone","got","gotten","greetings","h","had","hadn't","happens","hardly","has","hasn't","have","haven't","having","he","he's","hello","help","hence","her","here","here's","hereafter","hereby","herein","hereupon","hers","herself","hi","him","himself","his","hither","hopefully","how","howbeit","however","i","i'd","i'll","i'm","i've","ie","if","ignored","immediate","in","inasmuch","inc","indeed","indicate","indicated","indicates","inner","insofar","instead","into","inward","is","isn't","it","it'd","it'll","it's","its","itself","j","just","k","keep","keeps","kept","know","knows","known","l","last","lately","later","latter","latterly","least","less","lest","let","let's","like","liked","likely","little","look","looking","looks","ltd","m","mainly","many","may","maybe","me","mean","meanwhile","merely","might","more","moreover","most","mostly","much","must","my","myself","n","name","namely","nd","near","nearly","necessary","need","needs","neither","never","nevertheless","new","next","nine","no","nobody","non","none","noone","nor","normally","not","nothing","novel","now","nowhere","o","obviously","of","off","often","oh","ok","okay","old","on","once","one","ones","only","onto","or","other","others","otherwise","ought","our","ours","ourselves","out","outside","over","overall","own","p","particular","particularly","per","perhaps","placed","please","plus","possible","presumably","probably","provides","q","que","quite","qv","r","rather","rd","re","really","reasonably","regarding","regardless","regards","relatively","respectively","right","s","said","same","saw","say","saying","says","second","secondly","see","seeing","seem","seemed","seeming","seems","seen","self","selves","sensible","sent","serious","seriously","seven","several","shall","she","should","shouldn't","since","six","so","some","somebody","somehow","someone","something","sometime","sometimes","somewhat","somewhere","soon","sorry","specified","specify","specifying","still","sub","such","sup","sure","t","t's","take","taken","tell","tends","th","than","thank","thanks","thanx","that","that's","thats","the","their","theirs","them","themselves","then","thence","there","there's","thereafter","thereby","therefore","therein","theres","thereupon","these","they","they'd","they'll","they're","they've","think","third","this","thorough","thoroughly","those","though","three","through","throughout","thru","thus","to","together","too","took","toward","towards","tried","tries","truly","try","trying","twice","two","u","un","under","unfortunately","unless","unlikely","until","unto","up","upon","us","use","used","useful","uses","using","usually","uucp","v","value","various","very","via","viz","vs","w","want","wants","was","wasn't","way","we","we'd","we'll","we're","we've","welcome","well","went","were","weren't","what","what's","whatever","when","whence","whenever","where","where's","whereafter","whereas","whereby","wherein","whereupon","wherever","whether","which","while","whither","who","who's","whoever","whole","whom","whose","why","will","willing","wish","with","within","without","won't","wonder","would","wouldn't","x","y","yes","yet","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves","z","zero","she's","he'd","she'd","he'll","she'll","shan't","mustn't","when's","why's","how's","area","areas","asked","asks","back","backed","backing","backs","began","beings","big","case","cases","clear","differ","differently","downed","downing","downs","early","end","ended","ending","ends","evenly","face","faces","fact","facts","felt","find","finds","full","fully","furthered","furthering","furthers","gave","general","generally","give","good","goods","great","greater","greatest","group","grouped","grouping","groups","high","higher","highest","important","interest","interested","interesting","interests","kind","knew","large","largely","latest","lets","long","longer","longest","made","make","making","man","member","members","men","mr","mrs","needed","needing","newer","newest","number","numbers","older","oldest","open","opened","opening","opens","order","ordered","ordering","orders","part","parted","parting","parts","place","places","point","pointed","pointing","points","present","presented","presenting","presents","problem","problems","put","puts","room","rooms","seconds","sees","show","showed","showing","shows","side","sides","small","smaller","smallest","state","states","thing","things","thinks","thought","thoughts","today","turn","turned","turning","turns","wanted","wanting","ways","wells","work","worked","working","works","year","years","young","younger","youngest"]


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
