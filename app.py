import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
ps = PorterStemmer()

def transform_text(text):
    text=text.lower()
    text = tokenizer.tokenize(text)   
    
    filtered_tokens = [token for token in text if token.isalnum()]
    
    text=filtered_tokens[:]
    filtered_tokens.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            filtered_tokens.append(i)

    text=filtered_tokens[:]
    filtered_tokens.clear()
    
    for i in text:
        filtered_tokens.append(ps.stem(i))
    
    return " ".join(filtered_tokens)

tfidf=pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))
st.title("Email Spam Detector")
input_email=st.text_area("Enter the Message")
if st.button('Predict'):

    # 1.preprocess
    transform_email=transform_text(input_email)
    
    # 2.vectorize
    vector_input=tfidf.transform([transform_email])
    
    # 3.predict
    result=model.predict(vector_input)[0]
    
    # 4.display
    if result==1:
        st.header("Spam")
    else:
        st.header("Not a Spam")