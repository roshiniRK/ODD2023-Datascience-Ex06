# EX-06 FEATURE TRANSFORMATION :
## AIM :
To read the given data and perform Feature Transformation process and save the data to a file.
## EXPLANATION :
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.
## ALGORITHM :
### STEP 1 :
Read the given Data.
### STEP 2 :
Clean the Data Set using Data Cleaning Process.
### STEP 3 :
Apply Feature Transformation techniques to all the features of the data set.
### STEP 4 :
Print the transformed features.
## PROGRAM :
```
NAME : ROSHINI R K
REG NO : 212222230123
```


### Importing libraries and reading csv file:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
df=pd.read_csv("Data_to_Transform.csv")
```
```
df.head()
print(df.info())
df.tail()
```
### Before Transformation:
```
sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.title("Highly Positive Skew")
plt.show()

sm.qqplot(df['Highly Negative Skew'],fit=True,line='45')
plt.title("Highly Negative Skew")
plt.show()


sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.title("Moderate Positive Skew")
plt.show()

sm.qqplot(df['Moderate Negative Skew'],fit=True,line='45')
plt.title("Moderate Negative Skew")
plt.show()
```
### Log Transformation:
```

df['Highly Positive Skew'] = np.log(df['Highly Positive Skew'])
sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.title("Highly Positive Skew")
plt.show()

df['Moderate Positive Skew'] = np.log(df['Moderate Positive Skew'])
sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.title("Moderate Positive Skew")
plt.show()
```
### Reciprocal Transformation:
```
df['Highly Positive Skew'] = 1/df['Highly Positive Skew']
sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.title("Highly Positive Skew")
plt.show()
```
### SquareRoot Transformation:
```
df['Highly Positive Skew'] = df['Highly Positive Skew']**(1/1.2)
sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.title("Highly Positive Skew")
plt.show()
```
### Power Transformation:
```


df['Moderate Positive Skew_1'], parameters=stats.yeojohnson(df['Moderate Positive Skew'])
sm.qqplot(df['Moderate Positive Skew_1'],fit=True,line='45')
plt.title("Moderate Positive Skew")
plt.show()

transformer=PowerTransformer("yeo-johnson")
df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.title("Moderate Negative Skew")
plt.show()
```
### Quantile Transformation:
```
qt = QuantileTransformer(output_distribution = 'normal')
df['ModerateNegativeSkew_2'] = pd.DataFrame(qt.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.title("Moderate  Negative Skew")
plt.show()
```
## OUTPUT :
### INFORMATION :

![Screenshot 2023-10-14 090757](https://github.com/Mamthaiyappaprabu/ODD2023-Datascience-Ex06/assets/119393563/93d7a13d-fa4e-4c46-9f53-256ca5bb4d52)

![Screenshot 2023-10-14 090804](https://github.com/Mamthaiyappaprabu/ODD2023-Datascience-Ex06/assets/119393563/5bbd678a-55ad-4a40-8888-fd7a2505e9a9)


![Screenshot 2023-10-14 090819](https://github.com/Mamthaiyappaprabu/ODD2023-Datascience-Ex06/assets/119393563/d5c25851-d98f-4b5c-8f3e-f1bcb9d4e79a)

### BEFORE TRANSFORMATION :

![Screenshot 2023-10-14 091226](https://github.com/Mamthaiyappaprabu/ODD2023-Datascience-Ex06/assets/119393563/c3394437-379d-40eb-bd16-27e78daa3404)

![Screenshot 2023-10-14 091234](https://github.com/Mamthaiyappaprabu/ODD2023-Datascience-Ex06/assets/119393563/601b49e0-912a-4dd1-97bb-2a2f3dd8546e)

![Screenshot 2023-10-14 091242](https://github.com/Mamthaiyappaprabu/ODD2023-Datascience-Ex06/assets/119393563/65924b76-174e-4877-ade2-5147be48d7f3)


![Screenshot 2023-10-14 091252](https://github.com/Mamthaiyappaprabu/ODD2023-Datascience-Ex06/assets/119393563/48789a82-3c9f-47ae-abc5-fea3898a4363)


### LOG INFORMATION :
![image](https://github.com/Mamthaiyappaprabu/ODD2023-Datascience-Ex06/assets/119393563/1b07fb42-828c-4f65-8a55-57cd2c6d9a99)


![Screenshot 2023-10-14 091505](https://github.com/Mamthaiyappaprabu/ODD2023-Datascience-Ex06/assets/119393563/08da3265-1ac3-4acf-9093-80c95a908959)

### RECIPROCAL TRANSFORMATION :

![Screenshot 2023-10-14 091725](https://github.com/Mamthaiyappaprabu/ODD2023-Datascience-Ex06/assets/119393563/8f1b6878-8612-41d4-b0f4-5d480752ce08)

### SQUARE ROOT TRANSFORMATION :
![image](https://github.com/Mamthaiyappaprabu/ODD2023-Datascience-Ex06/assets/119393563/644b1e8a-19bb-4676-bd31-b31176ae1ee3)

### POWER TRANSFORMATION :


![Screenshot 2023-10-14 092027](https://github.com/Mamthaiyappaprabu/ODD2023-Datascience-Ex06/assets/119393563/b91215b6-1f47-47cd-b313-5114f8ee7f4b)


![Screenshot 2023-10-14 092050](https://github.com/Mamthaiyappaprabu/ODD2023-Datascience-Ex06/assets/119393563/c9c346e2-732d-477c-baf3-d4ee02dbf390)

### QUANTILE TRANSFORMATION :

![image](https://github.com/Mamthaiyappaprabu/ODD2023-Datascience-Ex06/assets/119393563/35b6208e-b3fd-4f3e-bc45-a6c39422288c)

## RESULT :

Thus feature transformation is done for the given dataset.

 
