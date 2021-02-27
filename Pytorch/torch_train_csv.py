import pandas as pd
df = pd.read_csv("E:\\Aquib\\MCA\\Python\\Dance Form Detection\\dataset\\train.csv")

Class_map={'manipuri':0, 'bharatanatyam':1, 'odissi':2 ,'kathakali':3, 'kathak':4, 'sattriya':5,
 'kuchipudi':6, 'mohiniyattam':7}

df.target = df.target.map(Class_map)

df.to_csv ('torch_train.csv', index = False)

