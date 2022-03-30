#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st 
import pandas as pd
import numpy as np
from pickle import dump
from pickle import load
from PIL import Image
import sklearn
import base64
from sklearn.feature_extraction.text import CountVectorizer
import pickle

CV = CountVectorizer(stop_words="english",max_features=5000,lowercase=False)

from load_css import local_css

local_css("style.css")

st.markdown(
        f"""
<style>
    .reportview-container .main .block-container{{
        padding-top: 0rem;
    }}
   
</style>
""",
        unsafe_allow_html=True,
    )

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    width: auto;
    height: auto;
    }
  }
    
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)
                                      
set_background('pic2.jpg')                          


col1, col2 = st.beta_columns(2)

names = "<div><span class='black_heading'>Topic Modeling</span></div>"
col1.markdown(names, unsafe_allow_html=True)

image = Image.open('excelr.png')
col2.image(image, caption=None, width=200, use_column_width=None, clamp=False, channels='RGB', output_format='auto')

if st.button("About Project"):
         st.text("Topic Modeling using Gensim")

if st.checkbox("Team Member"):
    
 
 st.text("Priyanka")
 st.text("Tejaswini")               
 st.text("Abhijit" )
 st.text ("Sanu") 
 st.text ("Bhasker")
 st.text("Maheshwari")



sentence = st.text_area("Input your sentence here:",height=200)
sentence1 = [sentence]   
arr_2d = np.reshape(sentence1, (1, -1))
print(arr_2d)

#loaded_model = load(open("clf.pickle", 'rb'))
loaded_model = load(open('lda_pickle', 'rb'))
#loaded_model = load(open('nby_model', 'rb'))

def Convert(string):
    li = list(string.split(" "))
    return li

#Convert('sentence1')
#vect= CV.fit_transform(sentence1).toarray()
#arr_2 = np.reshape(vect, (1, -1))


# Testing
#t_st = "applied statistics comprises descriptive statistics. and the application of inferential statistics. "

#sentence1 = [t_st]
##sentence = Convert(sentence1)
t1 = CV.fit_transform(sentence1)
test = np.array(t1,dtype=object).reshape(1,-1)



#if st.button("Predict"): 
    
if(len(sentence)!=0):
    prediction = loaded_model.predict(t1)
    if(prediction==0): col1.markdown("<div><span class='purple'>TOPIC:Democracy</span></div>",unsafe_allow_html=True)
    if(prediction==1): col1.markdown("<div><span class='yellow'>TOPIC: Politics</span></div>",unsafe_allow_html=True)
    if(prediction==2): col1.markdown("<div><span class='red'>TOPIC: Reseraches</span></div>",unsafe_allow_html=True)
    if(prediction==3): col1.markdown("<div><span class='blue'>TOPIC: Social_medium</span></div>",unsafe_allow_html=True)
    if(prediction==4): col1.markdown("<div><span class='green'>TOPIC :Statistic</span></div>",unsafe_allow_html=True)
         

st.write("\n")
st.write("\n")
