import nltk
import re
import os
import torch
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
# from transformers import BertForSequenceClassification

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

def clean_gdelt(gdelt_file="Sierra_Leone_top_companies_wurls.csv"):
    '''
    an aggregated functions which use the "clean_gdelt_url" and 
    "clean_gdelt_theme" functions to clean gdelt dataframe and 
    return two dataframes for url and themes
    '''
    gdelt_data = pd.read_csv(gdelt_file)
    gdelt_data["cleaned_url"] = gdelt_data["DocumentIdentifier"].apply(clean_gdelt_url)
    gdelt_data["cleaned_themes"] = gdelt_data["Themes"].apply(clean_gdelt_theme)
    gdelt_url_df = gdelt_data[gdelt_data["cleaned_url"].notna()].dropna()
    gdelt_theme_df = gdelt_data[gdelt_data["cleaned_themes"].notna()].dropna()
    
    gdelt_url_df.drop("cleaned_themes", axis=1, inplace=True)
    gdelt_theme_df.drop("cleaned_url", axis=1, inplace=True)
    
    return gdelt_url_df, gdelt_theme_df

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


def label_gdelt(gdelt_url_df, gdelt_theme_df, model, tokenizer):
    '''
    transform GDELT data and get the labels

    return: two dataframes, one for gdelt url labels, one for gdelt theme labels
    '''
    batch_size = 3

    # label with url                                          
    encoded_data_url = tokenizer.batch_encode_plus(
        gdelt_url_df["cleaned_url"], 
        add_special_tokens=True, 
        return_attention_mask=True, 
        padding=True, 
        max_length=384,
        truncation=True,
        return_tensors='pt'
    )
    input_ids_url = encoded_data_url['input_ids']
    dataloader_url = DataLoader(input_ids_url,batch_size=batch_size)
    total_predicted_label_url = []
    for batch_url in dataloader_url:
        output_url = model(batch_url)
        _, predicted_url = torch.max(output_url[0], 1)
        total_predicted_label_url += predicted_url.tolist()

    total_predicted_label_url = map(transfer_label_value, total_predicted_label_url)
    gdelt_url_df["SASB_tag_based_on_url_bert"] = list(total_predicted_label_url)

    outdir = './gdelt_outputs_BERT'
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    outname_url = "gdelt_url_industry_tag_bert.csv"
    fullname_url = os.path.join(outdir, outname_url)
    gdelt_url_df.to_csv(fullname_url)

    # label with themes
    encoded_data_themes = tokenizer.batch_encode_plus(
        gdelt_theme_df["cleaned_themes"], 
        add_special_tokens=True, 
        return_attention_mask=True, 
        padding=True, 
        max_length=384,
        truncation=True,
        return_tensors='pt'
    )
    input_ids_themes = encoded_data_themes['input_ids']
    dataloader_themes = DataLoader(input_ids_themes,batch_size=batch_size)
    total_predicted_label_themes = []
    for batch_theme in dataloader_themes:
        output_themes = model(batch_theme)
        _, predicted_themes = torch.max(output_themes[0], 1)
        total_predicted_label_themes += predicted_themes.tolist()
    
    total_predicted_label_url = map(transfer_label_value, total_predicted_label_themes)
    gdelt_theme_df["SASB_Tag_based_on_Themes_bert"] = list(total_predicted_label_themes)

    outname_theme = "gdelt_theme_industry_tag_bert.csv"
    fullname_theme = os.path.join(outdir, outname_theme)
    gdelt_theme_df.to_csv(fullname_theme)
        