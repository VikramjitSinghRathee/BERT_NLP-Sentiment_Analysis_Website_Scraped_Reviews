__author__='Vikramjit Singh Rathee'

'''1. IMPORT NECESSARY PACKAGES'''
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import torch.nn.functional as F
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np

def main():
     ''' ### MAIN ###'''
     '''2. INSTANTIATE MODEL'''
     model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
     tokenizer = AutoTokenizer.from_pretrained(model_name)
     model = AutoModelForSequenceClassification.from_pretrained(model_name)
     ## Define a function that returns sentiment label
     def sentiment_value(batch):
          print("**************** INSIDE SENTIMENT VALUE FUNCTION ****************")
          with torch.no_grad():
               outputs = model(**batch) # ** means unpack the dictonary to get the tensors, no need to do this with tensorflow
               # labels=torch.tensor([0,0,0]), labels argument is special to AutoModelForSequenceClassification and gets the loss value
               print("## OUTPUTS")
               print(outputs)
               predictions = F.softmax(outputs.logits, dim=1) # logits are the scores from the last layer of the NN
               print("## PREDICTIONS")
               print(predictions)
               labelIDs = torch.argmax(predictions, dim=1)
               print("## LABEL IDS")
               print(labelIDs)
               labels = [model.config.id2label[label_id] for label_id in labelIDs.tolist()] # id2label is only available for AutoModelForSequenceClassification
               print("## LABELS")
               print(labels)
               return labels

     '''3. CHECKING AN EXAMPLE OF TOKENIZATION USING THE SENTIMENT CLASSIFICATION MODEL'''
     ExTokens = tokenizer.tokenize("We are very happy to show you the Transformers library")
     token_ids = tokenizer.convert_tokens_to_ids(ExTokens)
     input_ids = tokenizer("We are very happy to show you the Transformers library") # dictionary with keys 'input_ids', 'token_type_ids', 'attention_mask'
     print(f' Example 1, Tokens: {ExTokens}')
     print(f'Token IDs: {token_ids}')
     print(f'Input IDs: {input_ids}') # 101 and 102 tokens represent the beginning and end of the string
     ## To decode tokens
     decodedTokens = tokenizer.decode(input_ids['input_ids']) # needs numerical ids to decode

     '''4. TEST THE SENTIMENT CLASSIFICATION MODEL ON FEW SENTENCES'''
     Xfew = ["I loved this, absolutely loved this",
               "We are not happy to visit this place.",
               "Oh no! I did not like this."]
     # create a batch of samples with same length by applying padding and truncation operation
     batch = tokenizer(Xfew, padding=True, truncation=True, max_length=512, return_tensors="pt") # pt is to return pytorch tensor, only pick first 512 tokens
     print(f"Batch of few sentences:{batch}")
     print(f"Sentiment labels of few sentences: {sentiment_value(batch)}")

     '''5. COLLECT REVIEWS'''
     r = requests.get('https://www.yelp.com/biz/cafe-sabarsky-new-york-4') # website to scrap
     soup = BeautifulSoup(r.text, 'html.parser')
     pattern = re.compile('.*comment.*')
     Extracted = soup.find_all('p',{'class':pattern})
     reviews = [e.text for e in Extracted]
     print(f"Example scraped review : {reviews[0]}")

     '''6. LOAD REVIEWS INTO DATAFRAME AND SCORE SENTIMENT VALUE'''
     df = pd.DataFrame(np.array(reviews), columns=['reviews'])
     df['Tokens'] = df['reviews'].apply(lambda x: tokenizer.tokenize(x))
     batch_reviews = tokenizer(reviews, padding=True, truncation=True, max_length=512, return_tensors="pt")
     df['Calc Sentiment Rating'] = sentiment_value(batch_reviews)
     print("COMPARE WITH THE ACTUAL RATINGS GIVEN THE USERS WHO LEFT THE REVIEWS ON THE WEBSITE, AND IF THEY MATCH!!!")

     '''7. TO SAVE AND LOAD THE MODEL AND TOKENIZER LOCALLY'''
     # save_dictionary = "SavedReviewModel"
     # tokenizer.save_pretrained(save_dictionary)
     # model.save_pretrained(save_dictionary)
     ### FOR LOADING ###
     # tokenizer = AutoTokenizer.from_pretrained(save_dictionary)
     # model = AutoModelForSequenceClassification.from_pretrained(save_dictionary)


### IMPORT PROOFING
if __name__ == '__main__':
     main()
else:
     print("ERROR, NOT AN IMPORTABLE MODULE")