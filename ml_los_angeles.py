from los_angeles_data_analysis import clear_data
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler 
from sklearn.tree import DecisionTreeClassifier,plot_tree

df=pd.read_csv("dataset/Crime_Data_from_2020_to_2024.csv")
df=clear_data(df)


d_sex={"Male":0,
       "Female":1,
       "Non-Binary / Unspecified":2,
       "Intersex / Hermaphrodite":3,
       "-":4}
df["Vict Sex"]=df["Vict Sex"].map(d_sex)
df=df[df["Vict Sex"]<2]

d_descent={"Hispanic/Latin/Mexican":0,
           "White":1,
           "Black":2,
           "Unknown":3,
           "Other":5,
           "Other Asian":4,
           "Filipino":6,
           "Chinese":7,
           "Japanese":8,
           "Vietnamese":9,
           "American Indian/Alaskan Native":10,
           "Asian Indian":11,
           "Pacific Islander":12,
           "Hawaiian":13,
           "Cambodian":14,
           "Laotian":15,
           "Guamanian":16,
           "Samoan":17,
           "-":18}
df["Vict Descent"]=df["Vict Descent"].map(d_descent)
df=df[df["Vict Descent"]<19]


"""
 #   Column          Non-Null Count   Dtype  
---  ------          --------------   -----  

 1   AREA            849323 non-null  int64 
 4   Part 1-2        849323 non-null  int64  
 5   Crm Cd          849323 non-null  int64  
 8   Vict Age        728927 non-null  float64
 9   Vict Sex        849323 non-null  str    
 10  Vict Descent    849323 non-null  str   
 13  Weapon Used Cd  325282 non-null  float64
 18  LAT             849323 non-null  float64
 19  LON             849323 non-null  float64
 
 15  Status          849323 non-null  str  """

scaler=StandardScaler()
print(df["AREA"].value_counts())
print(df["Part 1-2"].value_counts())
print(df["Crm Cd"].value_counts())
print(df["Vict Age"].value_counts())
print(df["Vict Sex"].value_counts())
print(df["Vict Descent"].value_counts())
print(df["Weapon Used Cd"].value_counts().to_string())
print(df["Status"].value_counts())
"""
Crm Cd
624    74002
330    60649
354    58541
230    52199
740    48758
       ...  
904        2
445        2
882        2
926        1
453        1
Name: count, Length: 138, dtype: int64
Vict Age
 30.0    21831
 35.0    21293
 31.0    20946
 29.0    20877
 28.0    20516
         ...  
 98.0       63
-1.0        25
-2.0        10
-3.0         2
-4.0         1
Name: count, Length: 102, dtype: int64

Vict Descent
0.0     293243
1.0     199064
2.0     134453
4.0      76690
5.0      21177
3.0       9452
7.0       4702
8.0       4576
9.0       1565
10.0      1185
11.0      1002
12.0       567
13.0       284
14.0       186
15.0        90
16.0        76
17.0        71
18.0        53

Name: count, dtype: int64
Weapon Used Cd
400.0    167683
500.0     32242
511.0     22387
102.0     18149
200.0      6530
          ...  
120.0         3
121.0         2
119.0         2
124.0         2
123.0         1
Name: count, Length: 79, dtype: int64
Status
IC    572666
AO    101299
AA     70205
JA      2648
JO      1614
CC         4
Name: count, dtype: int64
"""
df=df[df["Vict Descent"]<3]
df=df[df["Status"]!="CC"]

features=["AREA","Part 1-2","Vict Age","Vict Sex","Vict Descent"]

x=df[features]
y = df["Status"]

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)

dtree=DecisionTreeClassifier()
dtree=dtree.fit(x_train,y_train)

y_predict=dtree.predict(x_test)

plt.figure(figsize=(15,9))
plot_tree(dtree,feature_names=features)
plt.savefig("images/decision_tree_plot.png")

c_matrix=metrics.confusion_matrix(y_test,y_predict)

cm=metrics.ConfusionMatrixDisplay(confusion_matrix=c_matrix)
cm.plot()
plt.title("Confusion Matrix")
plt.savefig("images/confusion_matrix_plot.png")
plt.show()

# ACCURACY SCORE
#-----------------------------------
print(f"Accuracy: {metrics.accuracy_score(y_test, y_predict)}")
# Accuracy: 0.7454073004743246

""""
AREA","Part 1-2","Vict Age","Vict Sex","Vict Descent","Weapon Used Cd
Accuracy: 0.7352359574581797

["AREA","Part 1-2","Vict Age","Vict Sex","Vict Descent"]
Accuracy: 0.753130288836298
"""