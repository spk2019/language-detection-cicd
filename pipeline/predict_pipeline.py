import sys
import pickle
import os
import re



class PredictPipeline:

    def __init__(self):
        pass

    def rawToclean(self,sentence):
        clean_text = []
        for text in sentence:
            text = re.sub(r'[!@#$(),\n"%^*?\:;~`0-9]', '', text)
            text = re.sub(r'[[]]', ' ', text)
            text = text.lower()
            clean_text.append(text)
        return clean_text

        
    def predict(self,raw_text):
        print("Before Loading")
        model=pickle.load(open("model.pkl", 'rb'))
        preprocessor=pickle.load(open("preprocessor.pkl", 'rb'))
        print("After Loading")
        clean_text = self.rawToclean([raw_text])
        text_vector=preprocessor.transform(clean_text)
        preds=model.predict(text_vector)
        return preds



