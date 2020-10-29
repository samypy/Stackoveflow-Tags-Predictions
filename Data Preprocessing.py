
import pandas as pd
import re

quest_data = pd.read_csv('/content/drive/My Drive/Stack Overflow/Questions.csv',encoding='latin-1')

quest_data.head(3)

tag_data = pd.read_csv('/content/drive/My Drive/Stack Overflow/Tags.csv',encoding='latin-1')

tag_data.head(3)

data = pd.merge(quest_data,tag_data, on='Id')

del quest_data

del tag_data

n = 10
filter_list = data['Tag'].value_counts()[:n].index.tolist()

data = data[data.Tag.isin(filter_list)]

data.reset_index(inplace=True)

data = data[['Id', 'Title', 'Body','Tag']]


def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
  cleantext = re.sub(cleanr, '', raw_html)
  cleantext = re.sub('\n', '', cleantext)
  return cleantext

data['Body']= data['Body'].apply(lambda x: cleanhtml(x))

data['Title']= data['Title'].apply(lambda x: cleanhtml(x))

data.head(5)

data['Title & Body'] = data['Title'].str.cat(data['Body'],sep = " ")

data = data[['Id', 'Title & Body', 'Tag']]

df = data.pivot_table(index =['Title & Body'], columns = ['Tag'],
                       values =['Tag'], 
                       aggfunc ='count')

df.fillna(0,inplace=True)

flattened = pd.DataFrame(df.to_records())

flattened.rename(columns = {"('Id', 'android')":'android',"('Id', 'c#')":'c#',"('Id', 'c++')":'c++',"('Id', 'html')":'html',"('Id', 'ios')":'ios',"('Id', 'java')":'java',"('Id', 'javascript')":'javascript',"('Id', 'jquery')":'jquery',"('Id', 'php')":'php',
                 "('Id', 'python')":'python'},
                 inplace = True)

col_list = ['android', 'c#', 'c++', 'html', 'ios', 'java',
       'javascript', 'jquery', 'php', 'python']
for col in col_list:
    flattened[col] = flattened[col].astype(int)

flattened.head(2)

flattened.to_pickle('flattened.pkl')

