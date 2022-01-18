import os
import numpy as np
import pickle
import tensorflow as tf
from transformers import TFDistilBertForSequenceClassification,DistilBertTokenizerFast

def cls():
    os.system('cls' if os.name=='nt' else 'clear')
cls()
print('Initialising the tokenizer and the model...')
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
cls()
model = TFDistilBertForSequenceClassification.from_pretrained('Movie_prediction_model')
cls()

file=open('model_metrics','rb')
history=pickle.load(file)
file.close()
loss,accuracy,ROC_AUC_SCORE=history.values()

def analysis(text, model, tokenizer):
    encodings=tokenizer(text, truncation=True, padding=True)
    data=tf.data.Dataset.from_tensor_slices((dict(encodings),[1]))
    return model.predict(data)

def input_fn(model,tokenizer):
    print('Enter the short movie description to be predicted for public response:')
    text=[input()]
    result=analysis(text,model,tokenizer)
    ans=np.argmax(result[0], axis = - 1)
    if ans:
        print(f'\nMovie is predicted with great public response of {accuracy} accuracy, {ROC_AUC_SCORE} ROC AUC Score.')
    else:
        print(f'\nMovie is predicted with bad public response of {accuracy} accuracy, {ROC_AUC_SCORE} ROC AUC Score.')

def main():
    input_fn(model,tokenizer)

if __name__ == '__main__':
    main()

