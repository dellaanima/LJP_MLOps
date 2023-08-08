import subprocess

# required packages 
required_packages = [
    "pandas",
    "tqdm",
    "boto3",
]

# package installation function
def install_packages(package_list):
    for package in package_list:
        try:
            subprocess.check_call(["pip", "install", package])
        except subprocess.CalledProcessError as e:
            print(f"Failed to install package {package}: {e}")
            raise


install_packages(required_packages)

import pandas as pd
import xml.etree.ElementTree as ET
from urllib.request import urlopen
from tqdm import trange
import re
import os
import boto3
from io import StringIO, BytesIO
from datetime import datetime 

# XML data 
url = "https://www.law.go.kr/DRF/lawSearch.do?OC=rnqhgml12&target=prec&type=XML"
response = urlopen(url).read()
xtree = ET.fromstring(response)
totalCnt = int(xtree.find('totalCnt').text)

rows = []
for page in trange(1, totalCnt // 20 + 1):
    url = "https://www.law.go.kr/DRF/lawSearch.do?OC=rnqhgml12&target=prec&type=XML&page={}".format(page)
    response = urlopen(url).read()
    xtree = ET.fromstring(response)

    try:
        items = xtree[5:]
    except:
        break

    for node in items:
        판례일련번호 = node.find('판례일련번호').text

        rows.append({'판례일련번호': 판례일련번호})

    if len(rows) >= 50:
        break

case_list = pd.DataFrame(rows)

# 앞서 만든 case_list를 아래 데이터를 불러오는데 사용
contents = ['참조조문', '판례내용']

def remove_tag(content):
    if content is None:
        return ''
    cleaned_text = re.sub('<.*?>', '', content)
    return cleaned_text

# Create a DataFrame to store the results
results = pd.DataFrame(columns=['판례일련번호'] + contents)

# 판례상세링크에 접속해서 필요한 데이터 내려받기
for i in trange(len(case_list)):
    url = "https://www.law.go.kr/DRF/lawService.do?OC=rnqhgml12&target=prec&ID="
    id = case_list.loc[i]['판례일련번호']
    url += str(id)
    end = "&type=XML&mobileYn="
    url += end
    response = urlopen(url).read()
    xtree = ET.fromstring(response)

    case_data = {'판례일련번호': ''}
    case_info = xtree.find('판례일련번호')
    if case_info is not None:
        case_data['판례일련번호'] = case_info.text

    # 불필요한 문자데이터 정리 및 데이터 받아오기
    for content in contents:
        content_element = xtree.find(content)
        if content_element is not None:
            text = content_element.text
            text = remove_tag(text)

            # '판례내용'의 길이가 1000자를 넘으면 1000자로 잘라냄
            if content == '판례내용' and len(text) > 1000:
                text = text[:1000]
        else:
            text = ''
        case_data[content] = text

    # Append the case data to the results DataFrame
    results = results.append(case_data, ignore_index=True)

#S3에 저장
s3_bucket = "sagemaker-us-east-1-353411055907"  # S3 버킷 이름으로 바꿔주세요
s3_client = boto3.client('s3')

# S3에서 labels.csv 가져오기
labels_obj = s3_client.get_object(Bucket=s3_bucket, Key='GP-LJP-mlops/labels.csv')
labels_data = labels_obj['Body'].read().decode('cp949')
labels_df = pd.read_csv(StringIO(labels_data))
# Compare and filter the rows
filtered_df2 = results[results['참조조문'].str.contains('|'.join(labels_df['laws_service']))]

# Function to find matched 'laws_service' and 'laws_service_id'
def find_matched_laws_service_id(row):
    matched_laws_service = [law for law in labels_df['laws_service'] if law in row['참조조문']]
    matched_laws_service_id = [str(labels_df.loc[labels_df['laws_service'] == law, 'laws_service_id'].values[0]) for law in matched_laws_service]
    return '|'.join(matched_laws_service_id)

# Apply the function to create the 'matched_laws_service_id' column
filtered_df2['matched_laws_service_id'] = filtered_df2.apply(find_matched_laws_service_id, axis=1)

# Concatenate matched 'laws_service' values in 'filtered_df2'
filtered_df2['matched_laws_service'] = filtered_df2.apply(lambda row: '|'.join([law for law in labels_df['laws_service'] if law in row['참조조문']]), axis=1)

# Extract the specified columns and rename them
new_df = filtered_df2[['matched_laws_service_id', '판례내용', 'matched_laws_service']].copy()
new_df.columns = ['laws_service_id', 'fact', 'laws_service']


current_date_str = f"data_{datetime.now().strftime('%Y-%m-%d %H')}"
#new_df를 CSV로 변환하여 S3에 저장
new_df_csv = new_df.to_csv(index=False)
file_name = f'GP-LJP-mlops/data/collected_data/{current_date_str}.csv'
s3_client.put_object(Bucket=s3_bucket, Key=file_name, Body=new_df_csv)


