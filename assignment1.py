# This is the file you will need to edit in order to complete assignment 1
# You may create additional functions, but all code must be contained within this file


# Some starting imports are provided, these will be accessible by all functions.
# You may need to import additional items
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# My imported libraries
import json
import csv
import os
import re
import seaborn as sns
import nltk
from collections import defaultdict as dd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



# You should use these two variable to refer the location of the JSON data file and the folder containing the news articles.
# Under no circumstances should you hardcode a path to the folder on your computer (e.g. C:\Chris\Assignment\data\data.json) as this path will not exist on any machine but yours.
datafilepath = 'data/data.json'
articlespath = 'data/football'

def task1():
    pl_data = open(datafilepath, "r")
    y = json.load(pl_data)

    team_codes = y["teams_codes"]
    team_codes.sort()
    return team_codes
    
def task2():
    new_file = open("task2.csv", "x")
    writer = csv.writer(new_file)

    # retrieve each clubs data and insert into dict
    pl_data = open(datafilepath, "r")
    y = json.load(pl_data)

    lst = []
    i = 0
    x = y["clubs"]
    while i < y["number_of_participating_clubs"]:
        team_code = (x[i])["club_code"]
        goals_scored = (x[i])["goals_scored"]
        goals_against = (x[i])["goals_conceded"]

        lst.append([team_code, goals_scored, goals_against])
        i += 1

    sorted_lst = sorted(lst, key=lambda f: f[0])

    header = ["team_code", "goals_scored_by_team", "goals_scored_against_team"]
    writer.writerow(header)
    
    for data in sorted_lst:
        writer.writerow(data)

    return
      
def task3():
    lst = []
    for filename in os.listdir(articlespath):
        f = open(os.path.join(articlespath, filename), "r")
        score = re.findall(r"\s(\d{1,2})-(\d{1,2})[^\d]", f.read())

        total = 0
        if score == None:
            lst.append([filename, total])
        else:
            max_total = 0
            for goals in score:
                total = int(goals[0])+int(goals[1])
                if total > max_total:
                    max_total = total
            lst.append([filename, max_total])

    lst.sort(key=lambda x: x[0])
    new_file = open("task3.csv", "x")
    writer = csv.writer(new_file)
    
    header = ["filename", "total_goals"]
    writer.writerow(header)
    
    for data in lst:
        writer.writerow(data)

    return

def task4():
    total_goals = pd.read_csv("task3.csv")

    goals = total_goals['total_goals']

    plt.boxplot(goals)
    plt.title("Total goals scored per article")
    plt.ylabel("Total goals scored")
    plt.savefig("task4.png")
    return
    
def task5():
    new_file = open("task5.csv", "x")
    writer = csv.writer(new_file)

    header = ["club_name", "number_of_mentions"]
    writer.writerow(header)


    pl_data = open(datafilepath, "r")
    y = json.load(pl_data)

    for team in y["participating_clubs"]:
        total = 0
        for filename in os.listdir(articlespath):
            f = open(os.path.join(articlespath, filename), "r")
            score = re.search(team, f.read())

            
            if score != None:
                total += 1

        writer.writerow([team, total])
        
    new_file.close()
    df = pd.read_csv("task5.csv")

    plt.bar(np.arange(len(df['number_of_mentions'])), df["number_of_mentions"])
    plt.xticks(np.arange(len(df["club_name"])), df["club_name"],
                 fontsize='xx-small', rotation=90)
    plt.ylabel("Number of Articles")
    plt.title("Number of Articles Mentioning each Club")
    plt.savefig("task5.png")
    
    return
    
def task6():
    pl_data = open(datafilepath, "r")
    y = json.load(pl_data)

    task5_df = pd.read_csv("task5.csv")
    task5_df.set_index("club_name", inplace=True)

    index = columns = y["participating_clubs"]
    lst = []
    for team1 in index:
        sub_lst = []
        for team2 in index:
            value = sim(team1, team2, task5_df["number_of_mentions"][team1],
             task5_df["number_of_mentions"][team2])
            sub_lst.append(value)
        lst.append(sub_lst)

    df = pd.DataFrame(lst, index=index, columns=columns)
    
    hm = sns.heatmap(df, xticklabels=True)
    figure = hm.get_figure()
    plt.title("Article mention similarity for pairs of Clubs")
    plt.xticks(fontsize="xx-small")
    plt.yticks(fontsize="xx-small")
    figure.savefig("task6.png")
    return
 
def task7():

    df_2 = pd.read_csv("task2.csv")
    df_5 = pd.read_csv("task5.csv")
    
    f = df_2.iloc[:,1]

    ff = df_5.join(f.rename("sums"))
    ff.set_index("club_name", inplace=True)
    
    plt.scatter(ff.iloc[:,0], ff.iloc[:,1])
    plt.title("Article mentions compared to Goals scored")
    plt.xlabel("Article mentions")
    plt.ylabel("Goals scored")
    plt.savefig("task7.png")
    return
    
def task8(filename):
    f = open(filename, "r")
    nltk.download('punkt')
    nltk.download('stopwords')

    out_string = ""
    for letter in f.read():
        if (letter == " " or letter == "\n") and (out_string[-1] != " "):
            out_string += " "
        elif letter.isupper():
            out_string += letter.lower()
        elif not letter.isalpha():
            out_string += " "
        else:
            out_string += letter

    wordlist = nltk.word_tokenize(out_string) 
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))

    token_wordlist = [w for w in wordlist if not w in stop_words]

    final_list = [w for w in token_wordlist if len(w)>1]
    return final_list
    
def task9():
    
    lst = []
    filenames = []
    for filename in os.listdir(articlespath):
        filenames.append(filename)
        lst.append(task8(os.path.join(articlespath, filename)))
        
    tfidf = TfidfVectorizer(tokenizer=lambda i:i, lowercase=False)
    result = tfidf.fit_transform(lst)
    
    out_list = []
    i = 0
    while i < (result.shape[0]-1):
        j = i+1
        while j < (result.shape[0]-1):
            
            app_list = [(filenames[i]), (filenames[j]), np.dot(result[i],
                         result[j].T)[0,0]]
            
            out_list.append(app_list)
            j+=1
        i+=1
    out_list.sort(key=lambda x: x[2], reverse=True)

    new_file = open("task9.csv", "x")
    writer = csv.writer(new_file)
    
    writer.writerows(out_list[:10])
    return

# similarity function for task6
def sim(team1, team2, team1_mentions, team2_mentions):
    value = 1

    if team1 == team2:
        return value
    
    mention_both = 0
    for filename in os.listdir(articlespath):
            f = open(os.path.join(articlespath, filename), "r")
            string = f.read()
            match1 = re.search(team1, string)
            match2 = re.search(team2, string)

            if (match1 != None) and (match2 != None):
                mention_both += 1

    value = (2*mention_both) / (team1_mentions + team2_mentions)
    return value

