import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import os

data="dataset/Crime_Data_from_2020_to_2024.csv"
OUTPUT="images"
os.makedirs(OUTPUT,exist_ok=True)

# LOAD DATA
#-------------------------------
def load_dataset(data):
    df=pd.read_csv(data)   
    return df 

# INSPECT DATA
#-------------------------------
def inspect_data(df):
    print(df.columns)
    print(df.head(5).to_string())

    print(df.isnull().sum())
    print(df.duplicated().sum()) # 7324
    print(df.info()) 
    """
     #   Column          Non-Null Count    Dtype  
---  ------          --------------    -----  
 0   TIME OCC        1004894 non-null  int64  
 1   AREA            1004894 non-null  int64  
 2   AREA NAME       1004894 non-null  str    
 3   Rpt Dist No     1004894 non-null  int64  
 4   Part 1-2        1004894 non-null  int64  
 5   Crm Cd          1004894 non-null  int64  
 6   Crm Cd Desc     1004894 non-null  str    
 7   Mocodes         853296 non-null   str    
 8   Vict Age        1004894 non-null  int64  
 9   Vict Sex        860263 non-null   str    
 10  Vict Descent    860251 non-null   str    
 11  Premis Cd       1004878 non-null  float64
 12  Premis Desc     1004306 non-null  str    
 13  Weapon Used Cd  327216 non-null   float64
 14  Weapon Desc     327216 non-null   str    
 15  Status          1004893 non-null  str    
 16  Status Desc     1004894 non-null  str    
 17  Crm Cd 1        1004883 non-null  float64
 18  Crm Cd 2        69154 non-null    float64
 19  Crm Cd 3        2314 non-null     float64
 20  Crm Cd 4        64 non-null       float64
 21  LOCATION        1004894 non-null  str    
 22  Cross Street    154228 non-null   str    
 23  LAT             1004894 non-null  float64
 24  LON             1004894 non-null  float64
dtypes: float64(8), int64(6), str(11)
memory usage: 191.7 MB"""
    print(df.describe())
    print(df.corr(numeric_only=True))

# CLEAR DATA
#-------------------------------
def clear_data(df):
    df.drop(columns=["DR_NO","Date Rptd","DATE OCC","Crm Cd 1","Crm Cd 2","Crm Cd 3","Crm Cd 4","Cross Street"],inplace=True)
    df.dropna(subset=["Vict Sex","Vict Descent","Mocodes","Premis Cd","Premis Desc","Status"],inplace=True)
    df.fillna(value='N/A')
    df.drop_duplicates(inplace=True)
    df['TIME OCC']=df['TIME OCC'].astype(str)
    df['TIME OCC'] = df['TIME OCC'].astype(str).str.zfill(4).apply(lambda x: f"{x[:2]}:{x[2:]}")
    df["Vict Sex"]=df["Vict Sex"].replace("F","Female")
    df["Vict Sex"]=df["Vict Sex"].replace("M","Male")
    df["Vict Sex"]=df["Vict Sex"].replace("X","Non-Binary / Unspecified")
    df["Vict Sex"]=df["Vict Sex"].replace("H","Intersex / Hermaphrodite")

    df["Vict Descent"]=df["Vict Descent"].replace("A","Other Asian")
    df["Vict Descent"]=df["Vict Descent"].replace("B","Black")
    df["Vict Descent"]=df["Vict Descent"].replace("C","Chinese")
    df["Vict Descent"]=df["Vict Descent"].replace("D","Cambodian")
    df["Vict Descent"]=df["Vict Descent"].replace("F","Filipino")
    df["Vict Descent"]=df["Vict Descent"].replace("G","Guamanian")
    df["Vict Descent"]=df["Vict Descent"].replace("H","Hispanic/Latin/Mexican")
    df["Vict Descent"]=df["Vict Descent"].replace("I","American Indian/Alaskan Native")
    df["Vict Descent"]=df["Vict Descent"].replace("J","Japanese")
    df["Vict Descent"]=df["Vict Descent"].replace("K","Korean ")
    df["Vict Descent"]=df["Vict Descent"].replace("L","Laotian")
    df["Vict Descent"]=df["Vict Descent"].replace("O","Other")
    df["Vict Descent"]=df["Vict Descent"].replace("P","Pacific Islander")
    df["Vict Descent"]=df["Vict Descent"].replace("S","Samoan")
    df["Vict Descent"]=df["Vict Descent"].replace("U","Hawaiian")
    df["Vict Descent"]=df["Vict Descent"].replace("V","Vietnamese")
    df["Vict Descent"]=df["Vict Descent"].replace("W","White")
    df["Vict Descent"]=df["Vict Descent"].replace("X","Unknown")
    df["Vict Descent"]=df["Vict Descent"].replace("Z","Asian Indian")

    df["Vict Age"] = df["Vict Age"].replace(0, np.nan)
    """
    The unusually high number of victims with age 0 is most likely due to missing or unknown age values rather than actual infant victims. According to common U.S. crime reporting standards (NIBRS/LIBRS), age value "00" is used to indicate an unknown victim age. Therefore, records with Vict Age = 0 were treated as missing values (or analyzed separately) to avoid misleading conclusions."""
    df=df[df["Vict Age"]>0]

    return df

# ANALYSIS OF SEX RATIO IN CRIME CASES IN LOS ANGELES 
#-------------------------------
def sex_ratio_analysis(df):
    sex_ratio=df["Vict Sex"].value_counts().sort_values(ascending=False).head(3)
    plt.figure(figsize=(6,6))
    sb.barplot(x=sex_ratio.index,y=sex_ratio.values)
    plt.title("Sex Ratio in Crimes Committed in Los Angeles")
    plt.xlabel("Sex")
    plt.ylabel("Number of Victims")
    plt.xticks(rotation=45,ha='right')
    plt.tight_layout()
    plt.grid(axis="y")
    plt.savefig(f"{OUTPUT}/sex_ratio_analysis.png")
    plt.show()

# ANALYSIS OF AGES OF VICTIMS
#-------------------------------
def analysis_of_ages_of_victims(df):
    age_graph=df["Vict Age"].value_counts()
    sb.lineplot(x=age_graph.index,y=age_graph.values)
    plt.title("Age Chart")
    plt.xlabel("Age")
    plt.ylabel("Number of Victims")
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"{OUTPUT}/analysis_of_ages_of_victims.png")
    plt.show()

# ANALYSIS OF DESCENTS OF VICTIMS
#-------------------------------
def analysis_of_descent_of_victims(df):
    descent_of_victims=df['Vict Descent'].value_counts().sort_values(ascending=False).head(10)
    plt.figure(figsize=(8,6))
    sb.barplot(y=descent_of_victims.index,x=descent_of_victims.values)
    plt.title("Top 10 Most Victimized Races")
    plt.xlabel("Number of Victims")
    plt.ylabel("Races")
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"{OUTPUT}/analysis_of_descent_of_victims_top_10.png")
    plt.show()

# ANALYSIS OF THE 20 MOST USED WEAPONS
#-------------------------------
def analysis_of_weapon_used_top_20(df):
    weapon_used_top_20=df["Weapon Desc"].value_counts().sort_values(ascending=False).head(20)
    plt.figure(figsize=(11,7))
    sb.barplot(x=weapon_used_top_20.values,y=weapon_used_top_20.index)
    plt.title("The 20 Weapons Most Frequently Used in Crimes")
    plt.xlabel("Number of Incidents Involving the Use of the Weapon")
    plt.ylabel("Weapons")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT}/analysis_of_weapon_used_top_20.png")
    plt.show()

# ANALYSIS OF AGE COMPARISON OF GENDERS
#-------------------------------
def analysis_of_age_comparison_of_genders(df):
    agevssex = df[(df['Vict Sex'] == 'Male') | (df['Vict Sex'] == 'Female')].groupby('Vict Age')['Vict Sex'].value_counts().reset_index(name='count')
    plt.figure(figsize=(12,6))
    sb.lineplot(data=agevssex,x="Vict Age",y="count",hue="Vict Sex")
    plt.title("Age Comparison of Male and Female Victims")
    plt.xlabel("Victim Age")
    plt.ylabel("Number of Victims")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT}/analysis_of_age_comparison_of_genders.png")
    plt.show()

# ANALYSIS OF STATUS DESCRIPTIONS
#-------------------------------
def analysis_of_status_description(df):
    status_description=df["Status Desc"].value_counts().sort_values(ascending=False)
    plt.figure(figsize=(7,4))
    sb.barplot(x=status_description.values,y=status_description.index)
    plt.title("Bar Plot of Status Description")
    plt.ylabel("Status Descriptions")
    plt.xlabel("Number of status of crimes")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT}/analysis_of_status_description.png")
    plt.show()

# ANALYSIS OF HOURS THAT CRIMES COMMITED
#-------------------------------
def analysis_of_crime_hours(df):
    time=df["TIME OCC"].value_counts().head(50).sort_index(ascending=True)
    plt.figure(figsize=(7,9))
    sb.barplot(y=time.index,x=time.values)
    plt.title("The Top 50 Times When Crimes Are Most Frequently Committed")
    plt.ylabel("Hours")
    plt.xlabel("Number of Crimes Commited")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT}/analysis_of_crime_hours.png")
    plt.show()

# ANALYSIS OF STATUS DESCRIPTION
#-------------------------------
def analysis_of_status_description(df):
    status=df["Status Desc"].value_counts()
    sb.barplot(x=status.index,y=status.values)
    plt.title("Bar Plot of Status Descriptions")
    plt.ylabel("Number of Crime Cases")
    plt.xlabel("Status")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT}/analysis_of_status_description.png")
    plt.show()

# ANALYSIS OF AREAS
#-------------------------------
def anaylsis_of_area(df):
    area=df["AREA NAME"].value_counts()
    sb.barplot(y=area.index,x=area.values)
    plt.title("Areas and Numbers of Crimes Committed")
    plt.ylabel("Areas")
    plt.xlabel("Number of Crimes Commited")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT}/analysis_of_areas.png")
    plt.show()


# MAIN PIPELINE
#-------------------------------
def main():
    df=load_dataset(data)
    inspect_data(df)
    df= clear_data(df)
    inspect_data(df)
    sex_ratio_analysis(df)
    analysis_of_ages_of_victims(df)
    analysis_of_descent_of_victims(df)
    analysis_of_weapon_used_top_20(df)
    analysis_of_age_comparison_of_genders(df)
    analysis_of_status_description(df)

# RUN
#-------------------------------
if __name__=="__main__":
    main()
