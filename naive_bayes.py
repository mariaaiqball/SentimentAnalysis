import re 
import pandas as pd 
import numpy as np 
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import math


class naive_bayes: 

    def __init__(self, df):
        self.df = df
        self.train_df = None 
        self.test_df = None
        self.train_positive = None
        self.train_negative = None 
        self.train_neutral = None
        self.positive_tokens_list = []
        self.positive_word_counts = 0 
        self.negative_tokens_list = []
        self.negative_word_counts = 0 
        self.neutral_tokens_list = []
        self.neutral_word_counts = 0 
        self.P_positive = 0
        self.P_negative = 0
        self.P_neutral = 0
        self.vocab = []
        self.y_true = []
        self.y_pred = []
        self.run_naive_bayes()

    
    def split(self):
        # split df into training and testing 
        #train_df, test_df = train_test_split(self.df, test_size=0.2, random_state=42)
        train_df, test_df = train_test_split(self.df, test_size=0.3, random_state=42, stratify=self.df['label'])
        self.train_df = train_df.reset_index(drop=True)
        self.test_df = test_df.reset_index(drop=True)
        

    def split_train(self):
        # split train data into positive, negative, and neutral 
        df = self.train_df
        self.train_positive = df[df['label']=='positive']
        self.train_negative = df[df['label']=='negative']
        self.train_neutral = df[df['label']=='neutral']


    def preprocess_text(self, text, remove_stopwords=True):
        # lowercase all words 
        lowercase = text.lower()
        # tokenize both words and emojis 
        emoji_pattern = re.compile("[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
                            "\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF"
                            "\U00002700-\U000027BF\U0001F900-\U0001F9FF"
                            "\U00002600-\U000026FF\u200d]+", flags=re.UNICODE)

        text = re.sub(r"[^a-z0-9\s" + emoji_pattern.pattern + "]", "", lowercase)
        tokens = text.split()
        #remove stopwords 
        stop_words = stopwords.words('english')
        if remove_stopwords:
            tokens = [w for w in tokens if w not in stop_words]
        return tokens 
    

    def tokenize(self, _print=False):
        # build the Counters
        pos_counts = Counter()
        neg_counts = Counter()
        neu_counts = Counter()

        for text in self.train_positive['text']:
            toks = self.preprocess_text(text)
            pos_counts.update(toks)
        for text in self.train_negative['text']:
            toks = self.preprocess_text(text)
            neg_counts.update(toks)
        for text in self.train_neutral['text']:
            toks = self.preprocess_text(text)
            neu_counts.update(toks)

        # attach to self
        self.positive_tokens_list = [list(pos_counts)]  # if you need the lists
        self.negative_tokens_list = [list(neg_counts)]
        self.neutral_tokens_list  = [list(neu_counts)]

        self.positive_word_counts = pos_counts
        self.negative_word_counts = neg_counts
        self.neutral_word_counts  = neu_counts

        # total tokens per class
        self.total_positive_tokens = sum(pos_counts.values())
        self.total_negative_tokens = sum(neg_counts.values())
        self.total_neutral_tokens  = sum(neu_counts.values())

        self.vocab = set(pos_counts) | set(neg_counts) | set(neu_counts)

        # build vocab as union
        if _print==True:
            print(f"Vocabulary size: {len(self.vocab)}")
            print(f"Total positive tokens: {self.total_positive_tokens}")
    

    def conditional_probability_smoothing(self, word, class_label):
        V = len(self.vocab)
        if class_label == 'positive':
            count = self.positive_word_counts.get(word, 0)
            return (count + 1) / (self.total_positive_tokens + V)
        elif class_label == 'negative':
            count = self.negative_word_counts.get(word, 0)
            return (count + 1) / (self.total_negative_tokens + V)
        elif class_label == 'neutral':
            count = self.neutral_word_counts.get(word, 0)
            return (count + 1) / (self.total_neutral_tokens + V)
        else:
            raise ValueError("Invalid class")


    def prior_prob(self):
            # Number of messages in positive, negative, neutral, and total. 
            num_positive = len(self.train_positive)
            num_negative = len(self.train_negative)
            num_neutral = len(self.train_neutral)
            num_total = len(self.train_df)

            # Compute the prior probabilities: P(positive), P(negative), P(neutral)
            self.P_positive = num_positive/num_total
            self.P_negative = num_negative/num_total
            self.P_neutral = num_neutral/num_total


    def _naive(self, text):
        tokens = self.preprocess_text(text, remove_stopwords=True)

        # Initialize prob with prior prob
        log_prob_positive = math.log(self.P_positive)
        log_prob_negative = math.log(self.P_negative)
        log_prob_neutral = math.log(self.P_neutral)

        # Add log conditional prob to each word 
        for word in tokens:
            if word in self.vocab:
                log_prob_positive += math.log(self.conditional_probability_smoothing(word, 'positive'))
                log_prob_negative += math.log(self.conditional_probability_smoothing(word, 'negative'))
                log_prob_neutral += math.log(self.conditional_probability_smoothing(word, 'neutral'))
        
        if log_prob_positive > log_prob_negative and log_prob_positive > log_prob_neutral:
            return 'positive' 
        elif log_prob_negative > log_prob_positive and log_prob_negative > log_prob_neutral:
            return 'negative'
        else:
            return 'neutral'
        

    def test_naive_bayes(self, _print=False, df=None):
        if df == None:
            df = self.test_df
        df = self.test_df
        self.y_true = list(df['label'])
        self.y_pred = []

        for text in df['text']:
            pred_label = self._naive(text)
            self.y_pred.append(pred_label)

        df["predicted"] = self.y_pred 

        for i in range(10):
            actual = df['label'][i]
            predicted = df['predicted'][i]
            if _print==True:
                print(f'message: {df['text'][i]}')
                print(f'{actual} -> {predicted}\n\n')


    def analyze(self):
        pred = self.y_pred
        true = self.y_true
        a = accuracy_score(true, pred)
        print("Accuracy:", a)

        precision = precision_score(true, pred, average="weighted")
        print(f'Precision: {precision}')

        recall = recall_score(true, pred, average="weighted")
        print(f'Recall: {recall}')

        f1 = f1_score(true, pred, average="weighted")
        print(f'F1 Score: {f1}')


        print("Classification Report:")
        print(classification_report(true, pred))
        

    def run_naive_bayes(self):
        self.split()
        self.split_train()
        self.tokenize()
        self.prior_prob()
        self.test_naive_bayes()
        self.analyze()