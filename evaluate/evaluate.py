import sys
sys.path.append("..")
import os
from client.client import Client
from docarray import Document, DocumentArray
import pandas as pd
import numpy as np
import math

#clicking module and weight function
def weight(index, row=0, column=0):
    row = index / 5
    column = index % 5
    return 40 - row - column

def clean():
    for dic in counter.keys():
        for attr_name in attributes:
            counter[dic][attr_name]["output"] = 0
            counter[dic][attr_name]["interface"] = 0

#the method of quantified bias
def calculate_bias1(counter):
    inline_bias = math.sqrt(np.mean(np.square(counter["input"])))
    content_diff = counter["input"] - counter["output"]
    content_bias = math.sqrt(np.mean(np.square(content_diff)))
    interface_diff = counter["input"] - counter["interface"]
    interface_bias = math.sqrt(np.mean(np.square(interface_diff)))
    return content_bias/inline_bias, interface_bias/inline_bias
            
def calculate_bias2(counter):
    content_diff = counter["output"] / counter["input"]
    content_bias = math.sqrt(np.mean(np.square(content_diff)))
    interface_diff = counter["interface"]/ counter["input"]
    interface_bias = math.sqrt(np.mean(np.square(interface_diff)))
    return content_bias, interface_bias

def calculate_bias3(counter):
    content_bias = sum(abs(counter["output"] - counter["input"]))
    interface_bias = sum(abs(counter["interface"] - counter["input"]))
    return content_bias, interface_bias

bias = {}
#computing bias
def get_bias(key):
    dics = []
    if key in counter.keys():
        dics = [key,]
    else:
        dics = counter.keys()
    for dic in dics:
        for attr_name in attributes:
            sum_val = counter[dic][attr_name]["output"].sum()
            if sum_val == 0:
                idxmin = counter[dic][attr_name]["input"].idxmin()
                counter[dic][attr_name].loc[idxmin,"output"] = 1
                counter[dic][attr_name].loc[idxmin,"interface"] = 1
                sum_val = 1
            
            counter[dic][attr_name]["output"] /= sum_val
            sum_val = counter[dic][attr_name]["interface"].sum()
            counter[dic][attr_name]["interface"] /= sum_val
            
                
            content_bias, interface_bias = calculate_bias1(counter[dic][attr_name])
    
            if dic not in bias[key].keys():
                bias[key][dic] = {}
            bias[key][dic][attr_name] = [content_bias, interface_bias]
            print(key,dic,attr_name,content_bias,interface_bias)

attributes = ["century","country","medium"]
req_key = ["vase","bottle","bowl","chair","cup","plate","pot","table"]
counter = {}
#read file and data
csv_file="../db/data.csv"
data_path = csv_file
dir_path = os.path.dirname(data_path)
file_name = os.path.basename(data_path)
df = pd.read_csv(csv_file, dtype=str, index_col=0)
docs = DocumentArray.from_dataframe(df)

objs = df["category"].value_counts().to_frame()

for obj_name in objs.index:
    counter[obj_name] = {}
    sub_obj = df[df["category"] == obj_name]
    for attr_name in attributes:
        attr_counter = sub_obj[attr_name].value_counts().to_frame()
        attr = pd.DataFrame()
        attr["input"] = attr_counter["count"]/len(sub_obj)
        attr.insert(1,"output",0)
        attr.insert(2,"interface",0)
        counter[obj_name][attr_name]=attr
res_item=objs
res_item['count'] = 0
#setting the pathway according to the file name and id we set
for doc in docs:
    doc.uri = dir_path + "/"+str(doc.tags['category'])+"/" + str(doc.tags["objid"]) + ".jpg"
#build the link to the index module
client = Client('http://0.0.0.0:51200')
client._client.post(on='/clear')
client.index(docs, show_progress=True, return_results=True)

for key in req_key:
    #key words searching
    ret = client.search([key,],limit=50)
    res = ret['@m']
    bias[key] = {}
    index = 0
    clean()
    res_item['count'] = 0
    for doc in res:
        res_item.loc[doc.tags["category"],"count"] += 1
        for attr_name in attributes:
            counter[doc.tags["category"]][attr_name].loc[doc.tags[attr_name],"output"] += 1
            counter[doc.tags["category"]][attr_name].loc[doc.tags[attr_name],"interface"] += weight(index)
        index += 1
    print(key)
    get_bias(key)
    #print(res_item)
    #print(bias)
