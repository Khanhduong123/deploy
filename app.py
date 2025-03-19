import streamlit as st
import os
import boto3
from transformers import pipeline
import torch


bucket_name = 'test-deploy-241218' 
local_path = 's3_download_1'
s3_prefix = 'ml-models/tinybert-sentiment-analysis/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


s3 = boto3.client('s3')
def download_dir(local_path, s3_perfix):
    os.makedirs(local_path, exist_ok=True)
    paginator= s3.get_paginator('list_objects_v2')
    for result in paginator.paginate(Bucket=bucket_name,Prefix=s3_perfix ):
        if 'Contents' in result:
            for key in result['Contents']:
                s3_key = key['Key']
                local_file = os.path.join(local_path,os.path.relpath(s3_key,s3_perfix))           
                s3.download_file(bucket_name,s3_key, local_file)

st.title('Machine Learning Model Deployment at the Server')
button = st.button("Download Model")
if button:
    with st.spinner("Downloading...Please wait"):
        download_dir(local_path, s3_prefix)

text  = st.text_area("Enter your reviews")
predict_button = st.button("Predict")
if predict_button:
    with st.spinner("Predicting..."):
        classifier = pipeline('text-classification', model=local_path, device=device)
        predicted = classifier(text)
        st.info(predicted[0]['label'])