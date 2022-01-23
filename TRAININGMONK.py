import numpy as np
import pandas as pd
sampleName="no"
sampleLabel=3
file_location=pd.read_csv("before_model/"+sampleName+"trainingdata.csv")
sd1={"sampleLabel_"+str(i):[] for i in range(len(file_location.columns)-1)}
for j in range(0,len(file_location),21):
    sd=file_location[j:j+21]
    for i in range(len(file_location.columns)-1):
        sd1["sampleLabel_"+str(i)].append(sum(sd["sampleLabel_"+str(i)])/21)
df = pd.DataFrame(sd1)
df['value']=[sampleLabel for i in range(len(file_location)//21)]
df.to_csv('data_folder/'+sampleName+'trainingdataavg.csv')