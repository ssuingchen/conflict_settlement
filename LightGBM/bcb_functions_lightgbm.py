import pandas as pd
import numpy as np
import nltk
import re
import os
from nltk.corpus import stopwords
nltk.download('stopwords')
stopwords1 = set(stopwords.words('english'))
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import lightgbm as lgbm
from lightgbm import LGBMClassifier


class LightGBM_Classifier:
    def __init__(self, model_file="lightgbm_sasb_model.txt", train_data="sasb_train_eval_dataset.csv", 
                 label_data="Sierra_Leone_top_companies_wurls.csv"):
        '''
        Load the fine-tuned lightgbm model and use it to label gdelt
        Vectorized the training data before labeling new data
        '''
        self.model = lgbm.Booster(model_file=model_file)
        self.sasb_data = pd.read_csv(train_data)
        self.gdelt_data = pd.read_csv(label_data)
        self.tfidf_vec = TfidfVectorizer(dtype=np.float32, sublinear_tf=True, use_idf=True, smooth_idf=True)
        self.data_tfidf = self.tfidf_vec.fit_transform(self.sasb_data['train_text'])

    @staticmethod    
    def clean_gdelt_url(text):
        '''
        get rid of ".html" and digital numbers before extracting
        the text part from the url

        parameter: string (gdelt url)

        return: string (the text part of a url)
        '''
        text = re.sub("\-[a-z][a-z]?\.html", "", text)
        text = re.sub("\.html", "", text)
        text = re.sub("\d+", "", text)
        text = re.findall("\/\w+\-.*$", text)
        
        if len(text) == 0:
            return None
        
        url_text = text[0]
        url_text = re.sub("\-+", " ", url_text)
        url_text = re.sub("\/", "", url_text)
        
        if re.search("%", url_text):
            return None

        return url_text

    
    @staticmethod
    def clean_gdelt_theme(text):
        '''
        get rid of "TAX", "FNCACT", "ECON", "ENV", "SOC", 
        these are the abbreviations frequently used

        parameter: string (gdelt Themes)

        return: sting (cleaned Themes)
        '''
        if type(text) != float:
            text = re.sub("TAX\_", "", text)
            text = re.sub("ENV\_", "", text)
            text = re.sub("ECON\_", "", text)
            text = re.sub("SOC\_", "", text)
            text = re.sub("ETH\_", "", text)
            text = re.sub("WB\_", "", text)
            text = re.sub("EPU\_", "", text)
            text = re.sub("UNGP\_", "", text)
            text = re.sub("SECT\_", "", text)
            text = re.sub("MSM\_", "", text)
            text = text.replace("FNCACT", "")
            text = text.replace("POINTSOFINTEREST", "")
            text = text.replace("WORLDLANGUAGES", "")
            text = text.replace("WORLDMAMMALS", "")
            text = text.replace("FOREIGNBANKS", "FOREIGN BANKS")
            text = text.replace("FOREIGNINVEST", "FOREIGN INVEST")
            text = text.replace("EMERGINGTECH", "EMERGING TECH")
            
            text = re.sub("\;+", " ", text)
            text = re.sub("\d+", "", text)
            text = re.sub("\_+", " ", text)

        else:
            text = None
        
        return text
    
    @staticmethod
    def transfer_label_value(label):
    '''
    transfer a value to it's corresponding key 
    
    parameter: int
    
    return: str
    '''

    label_dict = {
        'consumer goods': 0,
        'extractives & minerals processing': 1,
        'financials': 2,
        'food & beverage': 3,
        'health care': 4,
        'infrastructure': 5,
        'renewable resources & alternative energy': 6,
        'resource transformation': 7,
        'services': 8,
        'technology & communications': 9,
        'transportation': 10}
    
    inverse_dict = {value: key for key, value in label_dict.items()}
    
    return inverse_dict[label]


    def clean_gdelt(self):
        '''
        an aggregated functions which use the "clean_gdelt_url" and 
        "clean_gdelt_theme" functions to clean gdelt dataframe and 
        return the dataframe
        '''
        self.gdelt_text_df = self.gdelt_data[["DocumentIdentifier", "Themes"]]
        self.gdelt_text_df["cleaned_url"] = self.gdelt_text_df["DocumentIdentifier"].apply(self.clean_gdelt_url)
        self.gdelt_text_df["cleaned_themes"] = self.gdelt_text_df["Themes"].apply(self.clean_gdelt_theme)
        self.gdelt_url_df = self.gdelt_text_df[["DocumentIdentifier", "cleaned_url"]].dropna()
        self.gdelt_theme_df = self.gdelt_text_df[["Themes", "cleaned_themes"]].dropna()
    
    
    def label_gdelt(self):
        '''
        transform GDELT data and get the labels
        
        return: two dataframes, one for gdelt url labels, one for gdelt theme labels
        '''
        self.gdelt_url = self.tfidf_vec.transform(self.gdelt_url_df["cleaned_url"])
        self.gdelt_themes = self.tfidf_vec.transform(self.gdelt_theme_df["cleaned_themes"])
        self.pred_url = self.model.predict(self.gdelt_url)
        self.pred_themes = self.model.predict(self.gdelt_themes)
        self.max_url = np.argmax(self.pred_url, axis=1)
        self.max_themes = np.argmax(self.pred_themes, axis=1)
        
        self.max_url = map(self.transfer_label_value, self.max_url)
        self.max_themes = map(self.transfer_label_value, self.max_themes)
        
        self.gdelt_url_df["SASB_Tag_based_on_URL_lgbm"] = list(self.max_url)
        self.gdelt_theme_df["SASB_Tag_based_on_Themes_lgbm"] = list(self.max_themes)
         
        outname_url = "gdelt_url_industry_tag_lightgbm.csv"
        outname_theme = "gdelt_theme_industry_tag_lightgbm.csv"
        outdir = './gdelt_outputs_lightgbm'

        if not os.path.exists(outdir):
            os.mkdir(outdir)

        fullname_url = os.path.join(outdir, outname_url)
        fullname_theme = os.path.join(outdir, outname_theme)
        self.gdelt_url_df.to_csv(fullname_url)
        self.gdelt_theme_df.to_csv(fullname_theme)
        