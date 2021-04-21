#%%#######################################################################
#                                SETUP                                   #
##########################################################################
import pandas as pd
from gensim.models import LdaModel
from progressbar import progressbar as pb
import pickle
from glob import glob
import numpy as np
import selenium
import requests
import base64
import time
import re

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import itertools

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm
tqdm.pandas()

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
matplotlib.rcParams['font.weight'] = 'normal'

# All sets
path = os.getcwd()+'/'
ldapath = path+'rq1/top_13/'
sovuln = pd.read_parquet(path+'data/generated/sovuln.parquet')
ssevuln = pd.read_parquet(path+'data/generated/ssevuln.parquet')
allvuln = pd.read_parquet(path+'data/generated/allvuln.parquet')
cols = ['postid','year','tags','title','question','answers']
sets = [('so', sovuln[cols].copy()), ('sse', ssevuln[cols].copy()), ('all', allvuln[cols].copy())]

# Make directory if not present
def makedir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# Topic Name dictionary
topic_names = pd.read_csv(ldapath+'clean_topics.txt',header=None)
topic_names = topic_names[topic_names[0].str.contains("Label")].reset_index(drop=True)
topic_names[0] = topic_names[0].str.replace('Label: ', '')
topic_names.to_csv(path+'data/generated/topic_names.csv',index=False)
topic_names = topic_names.to_dict()[0]

# Globals
n_topics = 13
t_threshold = 0.1

# Calculate topics for each set

if len(sys.argv) == 1:

    # %%######################################################################
    #                              Load Texts                                #
    ##########################################################################

    print('Load texts')

    ## Calculate texts
    texts = dict()
    for setname, s in sets:
        texts[setname] = []
        for row in pb(s.itertuples()):
            t = "{} {} {} {}".format(row.tags, row.title, row.question, row.answers).split()
            texts[setname] += [[w for w in t if len(w) > 1 and len(w) < 12]]

    # %%######################################################################
    #                             Infer topics                               #
    ##########################################################################

    print('Infer Topics')

    lda_texts = dict()
    for setname, s in sets:

        # Load model and data
        lda = LdaModel.load(ldapath+'lda.model')
        corpus = [lda.id2word.doc2bow(text) for text in pb(texts[setname])]
        lda_texts[setname] = [lda[i] for i in pb(corpus)]

        ## Create topic probability table
        df = s[['postid','year','tags']].copy()
        df['topics'] = lda_texts[setname]
        dfr = []
        for i in pb(df.itertuples()):
            for j in i.topics:
                dfr.append((i.postid, i.year, i.tags, j[0], j[1]))
        dfr = pd.DataFrame(dfr, columns=['postid','year','tags','topic','score'])
        makedir(path+'data/generated')
        dfr.to_csv(path+'data/generated/{}_posttopics.csv'.format(setname),index=False)

#%%#######################################################################
#                            Topic Share                                 #
##########################################################################

print('Topic Share')

for setname, s in sets:

    ## Get share value
    topicdf = pd.read_csv(path+'data/generated/{}_posttopics.csv'.format(setname))
    topicdf = topicdf[topicdf.score > t_threshold]
    tshare = topicdf.groupby('topic').sum().divide(len(s)).drop(columns=['postid']).reset_index()
    tshare['topic'] = tshare.reset_index().topic.apply(lambda x: topic_names[x])

    # %% Topic share RQ1
    f, ax = plt.subplots(figsize=(10, 10))
    sns.set_color_codes("colorblind")
    sns.set(style="whitegrid")
    sns_plot = sns.barplot(x="score", y="topic", data=tshare.sort_values('score',ascending=0))
    sns_plot.set_title('Topic Share RQ1 ({})'.format(setname))
    savedir = path+'/outputs/rq1/{}/'.format(setname)
    makedir(savedir)

    tshare.sort_values('score', ascending=0).to_csv(savedir + 'topic_share_{}.csv'.format(setname), index=False)

    sns_plot.figure.savefig(savedir+'1_topic_share.png',bbox_inches='tight')

#%%#######################################################################
#                        Topic Trends Over Time                          #
##########################################################################

print('Topic Trends over Time')

## Replace nth occurance of string
def nth_repl(s, sub, repl, nth):
    find = s.find(sub)
    # if find is not p1 we have found at least one match for the substring
    i = find != -1
    # loop util we find the nth or we find no match
    while find != -1 and i != nth:
        # find + 1 means we start at the last match start index + 1
        find = s.find(sub, find + 1)
        i += 1
    # if i  is equal to nth we found nth matches so replace
    if i == nth:
        return s[:find]+repl+s[find + len(sub):]
    return s

## Get number of posts in year interval
def num_post_ran(year, ran, dict_times, col='postid'):
    if year >= ran[0][0] and year <= ran[0][1]:
        return dict_times.loc[(dict_times['year']>=ran[0][0]) & (dict_times['year']<=ran[0][1])].sum()[col]
    if year >= ran[1][0] and year <= ran[1][1]:
        return dict_times.loc[(dict_times['year']>=ran[1][0]) & (dict_times['year']<=ran[1][1])].sum()[col]
    if year >= ran[2][0] and year <= ran[2][1]:
        return dict_times.loc[(dict_times['year']>=ran[2][0]) & (dict_times['year']<=ran[2][1])].sum()[col]
    return dict_times.loc[(dict_times['year']<=ran[0][0])].sum()[col]

## Year groups
def year_group(year, ran):
    if year >= ran[0][0] and year <= ran[0][1]:
        return "{}-{}".format(ran[0][0],ran[0][1])
    if year >= ran[1][0] and year <= ran[1][1]:
        return "{}-{}".format(ran[1][0],ran[1][1])
    if year >= ran[2][0] and year <= ran[2][1]:
        return "{}-{}".format(ran[2][0],ran[2][1])
    return "<{}".format(ran[0][0])

for setname, s in sets:

    ## Sparklines CSV
    topicdf = pd.read_csv(path+'data/generated/{}_posttopics.csv'.format(setname))
    topicdf.topic = topicdf.topic.apply(lambda x: topic_names[x])
    topicdf = topicdf.groupby(['topic','year']).sum()[['score']].reset_index()
    dict_times = s.groupby('year').count()[['postid']]
    dict_times.columns=['post_count']
    dict_times.post_count = dict_times.post_count.astype('int')
    dict_times.index = dict_times.index.astype('int')
    topicdf = topicdf.set_index('year').join(dict_times).reset_index()
    topicdf.score /= topicdf.post_count
    topicdf.score *= 100
    topicdf = topicdf.pivot(index='topic',columns='year',values='score')
    topicdf.to_csv(path+"/outputs/rq1/topics_sparkline_{}.csv".format(setname))

    ## Cumalative
    for ran_type, ran in [('groups',[(2010,2013),(2014,2016),(2017,2019)]), ('cumulative',[(2010,2013),(2010,2016),(2010,2019)])]:

        ## Get topic trends over time
        topicdf = pd.read_csv(path+'data/generated/{}_posttopics.csv'.format(setname))
        topicdf.topic = topicdf.topic.apply(lambda x: topic_names[x])
        topicdf = topicdf.groupby(['topic','year']).sum().reset_index()
        tdf = []
        for t in topicdf.topic.drop_duplicates():
            for yr in topicdf.year.drop_duplicates():
                tdf.append([t, yr, num_post_ran(yr, ran, topicdf[topicdf.topic==t], 'score')])
        tdf = pd.DataFrame(tdf,columns=['topic','year','score']).sort_values('score').drop_duplicates()
        tdf.year = tdf.year.apply(lambda x: year_group(x, ran))
        topicdf = tdf.copy().reset_index(drop=1).drop_duplicates()

        ## Get post numbers
        dict_times = s.groupby('year').count()[['postid']]
        dict_times.index = dict_times.index.astype('int')
        dict_times = dict_times.reset_index()
        dict_times = pd.DataFrame([(i, num_post_ran(i,ran,dict_times)) for i in dict_times.year], columns=['year','counts'])
        dict_times.year = dict_times.year.apply(lambda x: year_group(x, ran))
        dict_times = dict_times.drop_duplicates().set_index('year')

        ## Join Topicdf and dict_times
        topicdf = topicdf.set_index('year').join(dict_times)
        topicdf.score = topicdf.score / topicdf.counts
        tovertime = topicdf.copy()
        tovertime = tovertime.groupby(['topic','year']).sum()[['score']].reset_index()
        tovertime = tovertime[tovertime.year.str.contains('-')]


# %%#######################################################################
#                       Sparkline visualization                           #
###########################################################################
so_sparkline = pd.read_csv(path + "outputs/rq1/topics_sparkline_so.csv")
sse_sparkline = pd.read_csv(path + "outputs/rq1/topics_sparkline_sse.csv")

savedir = path+'/outputs/rq1/sparkline_plots/'
makedir(savedir)

so_sparkline.topic = so_sparkline.topic.astype('str')
sse_sparkline.topic = sse_sparkline.topic.astype('str')

def extract_topic_number(topic):
    return re.findall(r'\d+', topic)[0]

so_sparkline['topic_no'] = so_sparkline.topic.map(extract_topic_number).astype(np.int64)
sse_sparkline['topic_no'] = sse_sparkline.topic.map(extract_topic_number).astype(np.int64)

so_sparkline.sort_values(by=['topic_no'], inplace=True)
sse_sparkline.sort_values(by=['topic_no'], inplace=True)

so_sparkline.drop(columns=["topic", "topic_no"], inplace=True)
sse_sparkline.drop(columns=["topic", "topic_no"], inplace=True)

def plot_sparkline(source, data, ax):

    if source == 'sse': linestyle = '--'
    else: linestyle = '-'

    ax.plot(data, linestyle=linestyle, linewidth=10, color='k')

    # remove all the axes
    for k, v in ax.spines.items():
        v.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

for row in range(len(so_sparkline)):

    cur_so_sparkline = so_sparkline.iloc[row].values
    cur_sse_sparkline = sse_sparkline.iloc[row].values

    fig, ax_array = plt.subplots(2, 1)
    fig.tight_layout(w_pad=0)
    ax_ravel = np.ravel(ax_array)
    plot_sparkline('so', cur_so_sparkline, ax_ravel[0])
    plot_sparkline('sse', cur_sse_sparkline, ax_ravel[1])
    plt.savefig(savedir + 'Sparklines_T{}.png'.format(row + 1), pad_inches=0, bbox_inches="tight", dpi=300)


#%%#######################################################################
#                             Tags in topics                             #
##########################################################################

print('Tags in Topics')

for setname, s in sets:

    ## Get Tag/Topic analysis
    topicdf = pd.read_csv(path+'data/generated/{}_posttopics.csv'.format(setname))
    topicdf['tags'] = topicdf.tags.apply(lambda x: x.split())
    topicdf = topicdf.explode('tags')
    topicdf = topicdf.drop(columns=['postid','year'])
    topicdf = topicdf.groupby(['topic','tags']).sum().divide(len(s)).reset_index()
    topicdf.topic = topicdf.topic.apply(lambda x: topic_names[x])
    topics_tags = topicdf.sort_values('score',ascending=0).groupby(['topic']).head(5).sort_values(['topic','score'],ascending=False)

    ## Save output as CSV
    savedir = path+'/outputs/rq1/{}/'.format(setname)
    makedir(savedir)
    topics_tags.to_csv(savedir+'4_topics_tags.csv',index=False)
