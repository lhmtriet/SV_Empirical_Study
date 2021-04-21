#%%#######################################################################
#                                SETUP                                   #
##########################################################################

import pandas as pd
from glob import glob
from progressbar import progressbar as pb
import numpy as np
import itertools
from collections import Counter
from datetime import datetime
import os
from scipy.stats.mstats import gmean
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn import preprocessing
import itertools
from textwrap import wrap
matplotlib.rcParams['font.family'] = ['Times New Roman']
matplotlib.rcParams['font.weight'] = 'normal'
pd.set_option('display.max_colwidth', None)

# Load data
path = os.getcwd()+'/'
sovuln = pd.read_parquet(path+'data/generated/sovuln.parquet')
ssevuln = pd.read_parquet(path+'data/generated/ssevuln.parquet')
allvuln = pd.read_parquet(path+'data/generated/allvuln.parquet')
sovuln = sovuln[['postid','year','creation_date']].set_index('postid')
ssevuln = ssevuln[['postid','year','creation_date']].set_index('postid')
topic_names = pd.read_csv(path+'data/generated/topic_names.csv').to_dict()['0']

# Helper Functions
def read_data(p):
    return pd.concat([pd.read_csv(i) for i in glob(path+'rq2/data/generated/'+p)])

def checktag(row, num):
    try:
        if num==1:
            if row.toptags1 in row.post_tags: return 1
        if num==3:
            for j in row.toptags3:
                if j in row.post_tags: return 1
        if num==5:
            for j in row.toptags5:
                if j in row.post_tags: return 1
    except:
        return 0
    return 0

def hour_diff(t1, t2):
    try:
        t1 = datetime.strptime(t1, '%Y-%m-%d %H:%M:%S')
        t2 = datetime.strptime(t2, '%Y-%m-%d %H:%M:%S')
        return (t1-t2).total_seconds() / 3600
    except:
        return 1

# Make directory if not present
def makedir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# From RQ4
def genexpertise(posts, files):
    df = posts.join(read_data('accepted_answer_owner/{}'.format(files)).set_index('id'))
    df.index.names = ['postid']
    df = df.reset_index()
    df = df.set_index('owneruserid').join(read_data('user_creation_date/{}'.format(files)).set_index('id'))
    df = df.set_index('postid')
    pts = pd.concat([pd.read_csv(i) for i in glob(path+'data/generated/{}_posttopics.csv'.format(files))])
    pts.topic = pts.topic.apply(lambda x: topic_names[x])
    pts = pts[pts.score > 0.1]
    df = df.drop(columns=['year'])
    pts = pts.set_index('postid')
    df = pts.loc[pts.index.isin(df.index)].join(df)
    df = df.dropna()
    df.a_reputation *= df.score
    df = df.groupby('topic').mean()
    df = df[['a_reputation']].reset_index()
    df = df[['topic','a_reputation']]
    df.columns = ['group','ans_rep_{}'.format(files[:-1])]
    return df.set_index('group')

#%%#######################################################################
#                           Community Owned Posts                        #
##########################################################################

print('Community Owned Posts')

def community(posts, files, soft=False):
    joined = posts.join(read_data('community_owned_date/{}'.format(files)).set_index('id')).fillna(0)
    makedir(path+'outputs/rq2/community_posts/')
    softcaptopics = pd.concat([pd.read_csv(i) for i in glob(path+'/data/generated/{}_posttopics*'.format(files))])
    softcaptopics.columns = ['postid','year_t','tags','topic','topic_score']
    if not soft:
        softcaptopics = softcaptopics.sort_values('topic_score',ascending=0).groupby('postid').head(1)
    df = softcaptopics.set_index('postid').join(joined)
    df.topic = df.topic.apply(lambda x: topic_names[x])
    df[df.community_owned_date!=0].to_csv(path+'outputs/rq2/community_posts/{}.csv'.format(files[:-1]))
# community(sovuln, 'so*')
# community(ssevuln, 'sse*')

#%%#######################################################################
#                                Popularity                              #
##########################################################################

def popularity(posts, files, nanswers=False, soft=False):
    makedir(path+'outputs/rq2/popularity/')
    joined = posts.join(read_data('favorite/{}'.format(files)).set_index('id')).fillna(0)
    joined = joined.join(read_data('view/{}'.format(files)).set_index('id')).fillna(0)
    joined = joined.join(read_data('score/{}'.format(files)).set_index('id')).fillna(0)
    joined = joined.join(read_data('ncomments/{}'.format(files)).set_index('id')).fillna(0)
    joined = joined.join(read_data('nanswers/{}'.format(files)).set_index('id')).fillna(0)

    ## Join topics
    softcaptopics = pd.concat([pd.read_csv(i) for i in glob(path+'/data/generated/{}_posttopics*'.format(files))])
    softcaptopics.columns = ['postid','year_t','tags','topic','topic_score']
    if not soft:
        softcaptopics = softcaptopics.sort_values('topic_score',ascending=0).groupby('postid').head(1)
        softcaptopics.topic_score = 1
    df = softcaptopics.set_index('postid').join(joined)
    df = df.groupby('topic').sum().reset_index()
    df.topic = df.topic.apply(lambda x: topic_names[x])

    ## Popularity is a list of SUM(P1), SUM(P2), SUM(P3), SUM(P4)
    pop_use = ['favorite','view_count','score','ncomments']
    if nanswers: pop_use += ['nanswers']
    df['popularity'] = df[pop_use].values.tolist()
    df.popularity = df.popularity.apply(np.product)
    df.popularity = df.popularity.apply(lambda x: x ** (1 / len(pop_use)))
    df['actual'] = df.popularity / df.topic_score
    if nanswers:
        df = df[['topic','topic_score','favorite','view_count','score','ncomments','nanswers','popularity','actual']].copy()
        df.columns = ['topic','|topic_i|','sum_favorite','sum_views','sum_score','sum_ncomments','sum_nanswers','(f*v*s*nc*na)^(1/{})'.format(len(pop_use)),'popularity']
    else:
        df = df[['topic','topic_score','favorite','view_count','score','ncomments','popularity','actual']].copy()
        df.columns = ['topic','|topic_i|','sum_favorite','sum_views','sum_score','sum_ncomments','(f*v*s*n)^(1/{})'.format(len(pop_use)),'popularity']
    df.to_csv(path+'outputs/rq2/popularity/{}_popularity_nanswers={}_soft={}.csv'.format(files[:-1],str(nanswers),str(soft)),index=False)

    ## Rename topics
    return df[['topic','popularity']].copy().set_index('topic')

#%%#######################################################################
#                                Difficulty                              #
##########################################################################

def difficulty_p1(posts, files, soft=False):
    makedir(path+'outputs/rq2/hours_accepted_answer/')
    joined = posts.join(read_data('answer_date/{}'.format(files)).set_index('id'))
    joined = joined.join(read_data('nanswers/{}'.format(files)).set_index('id')).fillna(0)
    joined = joined.join(read_data('view/{}'.format(files)).set_index('id')).fillna(0)
    joined.creation_date = joined.creation_date.str.replace('\..*','').str.replace(' UTC','')
    joined.answer_date = joined.answer_date.str.replace('\..*','')
    joined = joined.dropna()
    joined['answerhours'] = joined.apply(lambda row: hour_diff(row.answer_date, row.creation_date), axis=1)
    joined['answersviews'] = joined.nanswers / joined.view_count
    joined[['answerhours','nanswers','view_count','answersviews']].sort_values('answerhours',ascending=0).rename(columns={'answerhours':'hours_for_accepted_answer'}).to_csv(path+'outputs/rq2/hours_accepted_answer/{}_posts.csv'.format(files[:-1]))
    joined = joined[['answerhours','answersviews']]
    joined[['answerhours']].rename(columns={'answerhours':'hours_for_accepted_answer'}).describe().to_csv(path+'outputs/rq2/hours_accepted_answer/{}_summary.csv'.format(files[:-1]))

    ## Join topics
    softcaptopics = pd.concat([pd.read_csv(i) for i in glob(path+'/data/generated/{}_posttopics*'.format(files))])
    softcaptopics.columns = ['postid','year_t','tags','topic','topic_score']

    if not soft:
        softcaptopics = softcaptopics.sort_values('topic_score',ascending=0).groupby('postid').head(1)
        softcaptopics.topic_score = 1

    df = softcaptopics.set_index('postid').join(joined)
    # df = df.dropna().groupby('topic').sum().reset_index()
	
    df = df.dropna().groupby('topic').agg({'topic_score':'sum','answerhours':'median','answersviews':'sum'}).reset_index()
	
    #df_tmp1 = df.dropna().groupby('topic').median()['answerhours'].to_frame()
	
    #print(df_tmp1.head())
	
    #df_tmp2 = df.dropna().groupby('topic').sum()[['topic_score', 'answersviews']]
	
    #print(df_tmp2.head())
	
    #df_combined = df_tmp1.join(df_tmp2).reset_index()

    ## Rename topics
    df.topic = df.topic.apply(lambda x: topic_names[x])
	
    # print(df.head())

    return df[['topic','topic_score','answerhours','answersviews']].copy().set_index('topic')

def difficulty_p2(posts, files, soft=False):
    joined = posts.join(read_data('accepted_answer_owner/{}'.format(files)).set_index('id')).fillna(0)
    joined.acceptedanswerid = joined.acceptedanswerid.apply(np.clip, a_min=0, a_max=1)

    ## Join topics
    softcaptopics = pd.concat([pd.read_csv(i) for i in glob(path+'/data/generated/{}_posttopics*'.format(files))])
    softcaptopics.columns = ['postid','year_t','tags','topic','topic_score']
    if not soft:
        softcaptopics = softcaptopics.sort_values('topic_score',ascending=0).groupby('postid').head(1)
        softcaptopics.topic_score = 1
    df = softcaptopics.set_index('postid').join(joined)
    df = df[['topic','topic_score','acceptedanswerid']]
    df = df.groupby('topic').agg({'topic_score':'sum', 'acceptedanswerid':'sum'}).reset_index()

    ## Rename topics
    df.topic = df.topic.apply(lambda x: topic_names[x])

    return df.set_index('topic')[['topic_score','acceptedanswerid']].copy()

def difficulty(posts, files, soft=False):
    df = difficulty_p1(posts,files,soft).join(difficulty_p2(posts,files,soft), lsuffix='_drop', rsuffix='')
    df = df[['topic_score','topic_score_drop','acceptedanswerid','answerhours','answersviews']]
    df.columns = ['|topic_i|','|topic_i|_drop','sum_D1','sum_D2','sum_D3']
    df['|topic_i|/sum_D1'] = df['|topic_i|'] / df.sum_D1
    # df['sum_D2/|topic_i|_drop'] = df.sum_D2 / df['|topic_i|_drop']
    df['sum_D2/|topic_i|_drop'] = df.sum_D2
    df['|topic_i|_drop/sum_D3'] = df['|topic_i|_drop'] / df.sum_D3
    df['difficulty'] = df[['|topic_i|/sum_D1','sum_D2/|topic_i|_drop','|topic_i|_drop/sum_D3']].values.tolist()
    df.difficulty = df.difficulty.apply(np.product)
    df.difficulty = df.difficulty.apply(lambda x: x ** (1/3))
    makedir(path+'outputs/rq2/difficulty/')
    df.to_csv(path+'outputs/rq2/difficulty/{}_difficulty_soft={}.csv'.format(files[:-1], str(soft)))
    return df[['difficulty']]

#%%#######################################################################
#                    Plot difficulty and popularity                      #
##########################################################################

print('Plot Popularity and Difficulty')

# Convenience function
def rq2(posts, files, includes_answers, soft):
    d = difficulty(posts, files,soft)
    p = popularity(posts,files,includes_answers,soft)

    # print(d)
    # print(p)

    ret = d.join(p)
    ret = ret.iloc[::-1]
    ret.columns = ['Difficulty', 'Popularity']
    return ret

# Font size
import matplotlib
font = {'weight': 'normal', 'size': 16}
matplotlib.rc('font', **font)

## Generate Plots
for includes_answers in [False]:
    for soft in [False]:

        rq2plot = rq2(sovuln, 'so*',includes_answers,soft).join(rq2(ssevuln, 'sse*',includes_answers, soft), lsuffix=' (SO)', rsuffix=' (SSE)')
        # rq2plot = rq2plot.join(genexpertise(sovuln, 'so*')).join(genexpertise(ssevuln, 'sse*'))

        # Manually set the values for 'Resource Leaks (T8)'
        rq2plot.at['Resource Leaks (T8)', 'Popularity (SSE)'] = 6.654581176
        rq2plot.at['Resource Leaks (T8)', 'Difficulty (SSE)'] = np.min(rq2plot['Difficulty (SSE)'])

        rq2barnorm = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(rq2plot), columns=rq2plot.columns, index=rq2plot.index)
        rq2barnorm = rq2barnorm[rq2barnorm.columns[rq2barnorm.columns.str.contains('SO')].tolist() + rq2barnorm.columns[rq2barnorm.columns.str.contains('SSE')].tolist()]

        ## Plot bar
        def mybar(df, w=0.3, pltxlabel='xlabel', pltylabel='ylabel', plttitle='title', dp=0):

            ## Setup
            N = len(df.variable.unique())
            ind = np.arange(len(df.group.unique()))
            width = w
            clrs = sns.color_palette('husl', n_colors=N)  # a list of RGB tuples
            plt.rcParams['axes.ymargin'] = 0.01
            plt.rcParams['axes.xmargin'] = 0.13
            fig, ax = plt.subplots(figsize=(10, 15))

            # Bar plots
            rects = []
            for count, row in enumerate(df.variable.unique()):
                rects.append(ax.barh(ind - width*(0 if count < 2 else 1), df[df.variable==row].value.tolist(), width, color=clrs[count], label=row, edgecolor='black'))

            # Labels
            ax.set_ylabel('Scores')
            ax.set_yticks(ind-(width/2))
            ax.set_yticklabels( df.group.unique().tolist() )
            ax.set_xlabel("", fontsize=18)
            ax.set_ylabel("", fontsize=18)
            ax.set_title("", fontsize=18)
            plt.xticks([])

            ax.yaxis.set_tick_params(labelsize=30)

            ax.set_axisbelow(True)
            ax.yaxis.grid(color='gray', linestyle='dashed')

            cnt = 0

            # Plot labels atop bars
            for rectsall in rects:
                for rect in rectsall:
                    h = rect.get_width()

                    cnt += 1

                    if cnt == 32:
                        ax.text(h + (0.09 if h >= 0 else -0.09), rect.get_y() + (w / 6), '-', ha='center', va='bottom', size=20)
                    else:
                        ax.text(h + (0.09 if h >= 0 else -0.09), rect.get_y()+(w/6), '%.{}f'.format(dp)%abs(h), ha='center', va='bottom', size=20)


            # Plot
            plt.legend(bbox_to_anchor=(-0.74, -0.06, 1.74, .102), loc=3, ncol=4, mode="expand", borderaxespad=0, prop={'size': 23})
            return fig

        for comp in list(itertools.combinations(['Difficulty','Popularity'], 2)):
            toplot = rq2barnorm[rq2barnorm.columns[rq2barnorm.columns.str.contains('|'.join(comp))]].copy()
            toplot.loc[:, toplot.columns[toplot.columns.str.contains(comp[0])]] += 0.0001
            toplot.loc[:, toplot.columns[toplot.columns.str.contains(comp[1])]] *= -1
            toplot.loc[:, toplot.columns[toplot.columns.str.contains(comp[1])]] -= 0.0001
            toplot = toplot.reset_index().melt(id_vars=['topic'])
            toplot.columns = ['group','variable','value']
            fig = mybar(toplot, w=0.4, dp=2, pltxlabel='Score Normalised by Category', pltylabel='Topics', plttitle='Normalised {} and {} of Topics'.format(comp[0],comp[1]))
            # fig.savefig(path+'outputs/rq2/{}{}_nanswers={}_soft={}.png'.format(comp[0],comp[1],str(includes_answers),str(soft)),bbox_inches="tight", dpi=300)
            fig.savefig(path + 'outputs/rq2/RQ2.png', bbox_inches="tight", dpi=300)
			
            # save_path = path + 'outputs/rq2/RQ2_new.pdf'
            #
            # fig.savefig(save_path, bbox_inches="tight", edgecolor='black')

#%%#######################################################################
#                      Plot Closed/Duplicate Posts                       #
##########################################################################

print('Plot Closed/Duplicate Posts')

def closeddupes(posts, files):
    lenposts = len(posts)
    joined = posts.join(read_data('moreposthistory/{}'.format(files)).set_index('postid'))
    joined = joined[joined.posthistorytypeid.isin([10,12,14,17,18,35,36,37,38])].reset_index()
    joined = joined.set_index('comment').join(pd.read_csv(path+'rq2/data/closereasontypes.csv').astype(str).set_index('comment')).reset_index()
    joined.loc[joined.posthistorytypeid==10,'Name'] = 'closed: ' + joined[joined.closereasontype.notnull()].closereasontype
    joined.Name = joined.Name.str.lower()
    edits = joined.groupby('postid')['Name'].apply(set)
    edits = pd.DataFrame(Counter([i for j in edits for i in j]).items(), columns=['edit','count']).sort_values('count')
    edits['ratio'] = edits['count'] / lenposts
    return edits

def plotcloseddupes():
    socd = closeddupes(sovuln,'so*').set_index('edit')
    ssecd = closeddupes(ssevuln,'sse*').set_index('edit')
    plotcd = socd.join(ssecd,lsuffix='_so',rsuffix='_sse')
    plotcd['count_all'] = plotcd.count_so + plotcd.count_sse
    plotcd['ratio_all'] = plotcd.count_all / len(allvuln)
    plotcd = plotcd.sort_values('count_all')
    plotcd = plotcd[plotcd.columns[plotcd.columns.str.contains('count')]]

    plt = plotcd.plot.barh(figsize=(10,10))
    plt.set_xlabel('Raw number of posts')
    plt.set_ylabel('Post Types')
    plt.set_title('Close/Duplicate Posts')
    plt.legend(loc='upper left',bbox_to_anchor=(1.0, 1))
    plt.get_figure().savefig(path+'outputs/rq2/closed_dupes.png',bbox_inches="tight")

plotcloseddupes()
