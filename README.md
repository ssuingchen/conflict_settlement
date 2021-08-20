# conflict_settlement
I worked for the Africa Center of Dispute Settlement at the University of Stellenbosch Business School as an intern in summer 2021 and implemented Natural Language Processing models to help textual analysis which will be used to analyze conflicts between private sectors and general public.

# This repository contains:

The specification for how a standard README should look.
1. A README file
2. An Allen NLP models application folder
3. A sequence classification folder using LightGBM
4. A sequence classification folder using BERT
5. A conda requirements text file
6. A pip requirements text file
7. A bash file which install all the required libraries

# Install

The libraries used are provided in the one pip requirement file and one conda requirement file which can be installed through the bash file using this command: **__bash install_env.sh__**

# How to run the code?
Please follow the markdown instructions in the example file in each folder. Each of the example files import a function python file to run the code. 

# Where do the models come from?

* Allen NLP models
  * Obtained the models by following Allen NLP documentation.
* LightGBM model
  * I fine-tuned a lightgbm.LGBMClassifier and it is stored in the LightGBM folder named **__lightgbm_sasb_model.txt__**
  * The training code is located in the LightGBM folder named **__lightgbm_classifier_model.ipynb__**
* BERT model
  * I fine-tuned a BertForSequenceClassification model, but I didn't upload this model due to it's size. (this is a 439.6 MB file)
  * The training code is located in the BERT folder named **__BERT_classifier_model.ipynb__**
