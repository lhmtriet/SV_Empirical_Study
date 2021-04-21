#%%#######################################################################
#                                SETUP                                   #
##########################################################################
import pandas as pd
from glob import glob
import so_textprocessing as stp
from nltk.corpus import stopwords
from gensim.models import LdaModel
from progressbar import progressbar as pb
import os
from sklearn import preprocessing
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from natsort import natsorted
matplotlib.rcParams['font.family'] = ['Times New Roman']
matplotlib.rcParams['font.weight'] = 'normal'
pd.set_option('display.max_colwidth', None)

## Load data
path = os.getcwd()+'/'
ldapath = path+'rq1/top_13/'
sovuln = pd.read_parquet(path+'data/generated/sovuln.parquet')
ssevuln = pd.read_parquet(path+'data/generated/ssevuln.parquet')
sovuln = sovuln[['postid','year','creation_date']].set_index('postid')
ssevuln = ssevuln[['postid','year','creation_date']].set_index('postid')

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

#%%#######################################################################
#                      Accepted Answers Preprocessing                    #
##########################################################################

print('Process Accepted Answers')

## Preprocess df Function
tp = stp.TextPreprocess()

for site in ['so','sse']:

    ## Create topic probability table for accepted answers
    aanswers = pd.concat([pd.read_csv(i) for i in glob(path+'rq2/data/generated/acceptedanswerbody/{}-*'.format(site))]).dropna()
    aanswers = tp.transform_df(aanswers, columns=['body'], reformat='stemprocessstoponly')
    aanswers = aanswers.rename(columns={'body':'answer'})

    ## Use LDA model to predict answer topics
    texts = [[j for j in i.split() if len(j) > 1 and len(j) < 12] for i in aanswers.answer]
    lda = LdaModel.load(ldapath+'lda.model')
    corpus = [lda.id2word.doc2bow(text) for text in pb(texts)]
    lda_texts = [lda[i] for i in pb(corpus)]

    ## Dictionary for topic names
    topic_names = pd.read_csv(ldapath+'clean_topics.txt',header=None)
    topic_names = topic_names[topic_names[0].str.contains("Label")].reset_index(drop=True)
    topic_names[0] = topic_names[0].str.replace('Label: ', '')
    topic_names[1] = topic_names[0].str.extract(pat = "\((.*?)\)")
    tdict = topic_names.set_index(1).to_dict()[0]

    # Topic Name dictionary
    topic_names = pd.read_csv(ldapath+'clean_topics_shrt.txt',header=None)
    topic_names = topic_names[topic_names[0].str.contains("Label")].reset_index(drop=True)
    topic_names[0] = topic_names[0].str.replace('Label: ', '')
    topic_names = topic_names.to_dict()[0]

    ## Create topic probability table
    aanswers['topics'] = lda_texts
    dfr = []
    for i in pb(aanswers.itertuples()):
        for j in i.topics:
            dfr.append((i.id, i.acceptedanswerid, i.answer_owner, i.accountid, j[0], j[1]))
    dfr= pd.DataFrame(dfr, columns=['postid','acceptedanswerid','answer_owner', 'accountid', 'topic','score'])
    dfr['topic'] = dfr.reset_index().topic.apply(lambda x: topic_names[x])
    makedir(path+'rq3/data/generated/')
    dfr.to_csv(path+'rq3/data/generated/{}-aanswertopics.csv'.format(site),index=False)

    ## Combined answers by accountid
    answers_by_aid = pd.DataFrame(aanswers.groupby('accountid')['answer'].apply(list)).reset_index()
    answers_by_aid.answer = answers_by_aid.answer.apply(lambda x: x[0])
    texts = [[j for j in i.split() if len(j) > 1 and len(j) < 12] for i in answers_by_aid.answer]
    lda = LdaModel.load(ldapath+'lda.model')
    corpus = [lda.id2word.doc2bow(text) for text in pb(texts)]
    lda_texts = [lda[i] for i in pb(corpus)]
    answers_by_aid['topics'] = lda_texts
    dfr = []
    for i in pb(answers_by_aid.itertuples()):
        for j in i.topics:
            dfr.append((i.accountid, i.answer, j[0], j[1]))
    dfr= pd.DataFrame(dfr, columns=['accountid','answer','topic','score'])
    dfr.topic = dfr.reset_index().topic.apply(lambda x: topic_names[x])
    dfr.to_csv(path+'rq3/data/generated/{}-aanswertopics-combined.csv'.format(site),index=False)

#%%#######################################################################
#                          General Expertise                             #
##########################################################################

def account_reputation(site=None):

    ## Get SO reputation by account id
    so_df = sovuln.join(read_data('accepted_answer_owner/{}'.format('so*')).set_index('id'))
    so_df = so_df.reset_index().copy()
    so_df = so_df.set_index('owneruserid').join(read_data('user_creation_date/{}'.format('so*')).set_index('id'))
    so_accountid = pd.read_csv(path+'rq3/data/generated/so-aanswertopics.csv').set_index('answer_owner')[['accountid']].drop_duplicates()
    so_df = so_df.dropna().join(so_accountid).reset_index().rename(columns={'index':'userid'}).set_index('accountid')[['userid','a_reputation']]

    ## Get SSE reputation by account id
    sse_df = ssevuln.join(read_data('accepted_answer_owner/{}'.format('sse*')).set_index('id'))
    sse_df.index.names = ['postid']
    sse_df = sse_df.reset_index().copy()
    sse_df = sse_df.set_index('owneruserid').join(read_data('user_creation_date/{}'.format('sse*')).set_index('id'))
    sse_accountid = pd.read_csv(path+'rq3/data/generated/sse-aanswertopics.csv').set_index('answer_owner')[['accountid']].drop_duplicates()
    sse_df = sse_df.dropna().join(sse_accountid).reset_index().rename(columns={'index':'userid'}).set_index('accountid')[['userid','a_reputation']]

    ## Join
    joined_df = so_df.join(sse_df, lsuffix='_so', rsuffix='_sse', how='outer').reset_index().fillna(0)
    joined_df['so_sse_reputation'] = joined_df.a_reputation_so + joined_df.a_reputation_sse
    if site == None:
        return joined_df.drop_duplicates()

    return joined_df[["userid_{}".format(site), 'so_sse_reputation']].drop_duplicates().set_index("userid_{}".format(site))


def genexpertise(posts, files, soft=True, accountwide=False):
    df = posts.join(read_data('accepted_answer_owner/{}'.format(files)).set_index('id'))
    df.index.names = ['postid']
    df = df.reset_index()
    df = df.dropna()
    df = df.set_index('owneruserid').join(read_data('user_creation_date/{}'.format(files)).set_index('id'))
    if accountwide:
        df = df.join(account_reputation(files[:-1]))
        df.a_reputation = df.so_sse_reputation
    df = df.set_index('postid')
    pts = pd.concat([pd.read_csv(i) for i in glob(path+'data/generated/{}_posttopics*'.format(files))])
    pts.topic = pts.topic.apply(lambda x: topic_names[x])
    if not soft:
        pts = pts.sort_values('score',ascending=0).groupby('postid').head(1)
        pts.score = 1
    df = df.drop(columns=['year'])
    pts = pts.set_index('postid')
    df = pts.loc[pts.index.isin(df.index)].join(df)
    df = df.groupby('topic').sum()
    df['a_reputation_raw'] = df.a_reputation
    df.a_reputation /= df.score
    # df.a_reputation = df.apply(lambda x: 0 if x.score <= 1 else x.a_reputation, axis=1)
    df[['score', 'a_reputation_raw', 'a_reputation']].rename(
        columns={'score': 'num_posts', 'a_reputation_raw': 'a_reputation',
                 'a_reputation': 'a_reputation/num_posts'}).loc[natsorted(df.index)].to_csv(
        path + 'outputs/rq3/general/{}_gen_expertise_calc_soft={}_accountwide={}.csv'.format(files[:-1],
                                                                                                   str(soft),
                                                                                                   str(accountwide)))
    df = df[['a_reputation']].reset_index()
    return df

#%%#######################################################################
#                            Overlapping Users                           #
##########################################################################

def topic_post_count(files, soft, one):
    pts = pd.concat([pd.read_csv(i) for i in glob(path+'data/generated/{}_posttopics*'.format(files))])
    pts.topic = pts.topic.apply(lambda x: topic_names[x])
    if not soft:
        pts = pts.sort_values('score',ascending=0).groupby('postid').head(1)
    if one:
        pts.score = 1
    return pts.groupby('topic').sum()[['score']].rename(columns={'score':'count'})

def posts_by_topic(posts, files, soft=False, one=False):
    df = posts.join(read_data('accepted_answer_owner/{}'.format(files)).set_index('id'))
    df.index.names = ['postid']
    df = df.dropna()
    df = df.join(read_data('owner_user_id/{}'.format(files)).set_index('id'))
    df.index.names = ['postid']
    df = df.reset_index()
    pts = pd.concat([pd.read_csv(i) for i in glob(path+'data/generated/{}_posttopics*'.format(files))])
    pts.topic = pts.topic.apply(lambda x: topic_names[x])
    if not soft:
        pts = pts.sort_values('score',ascending=0).groupby('postid').head(1)
    if one:
        pts.score = 1
    df = df.set_index('postid')[['owneruserid','owner_user_id']].join(pts.set_index('postid')[['topic','score']])
    df.columns = ['answer_owner_id','question_owner_id','topic','score']
    df = df.reset_index().melt(id_vars=['topic','score','postid']).sort_values('postid')
    df.columns= ['topic','score','postid','user_id_type','userid_{}'.format(files[:-1])]
    return df.dropna()

def overlapping(source, soft=False, one=False):
    df = account_reputation()
    df = df[(df.userid_sse > 0) & (df.userid_so > 0)]
    df = df.sort_values('so_sse_reputation')[['accountid','userid_so','userid_sse']]

    so_df = df.set_index('userid_so').join(posts_by_topic(sovuln,'so*',soft,one).set_index('userid_so')).reset_index()
    sse_df = df.set_index('userid_sse').join(posts_by_topic(ssevuln,'sse*',soft,one).set_index('userid_sse')).reset_index()
    so_df['source'] = 'so'
    sse_df['source'] = 'sse'
    all_df = pd.concat([so_df,sse_df])
    all_df = all_df[all_df.user_id_type.str.contains('answer')]

    makedir(path+'outputs/rq3/overlapping/{}/'.format(source))
    all_df = all_df[all_df.source==source]
    all_df.to_csv(path+'outputs/rq3/overlapping/{}/raw_soft={}_roundup={}.csv'.format(source, str(soft),str(one)))
    all_df = all_df.groupby(['topic']).sum()[['score']]

    all_df = all_df.join(topic_post_count(source+'*', soft, one))
    all_df['score/count'] = all_df.score / all_df['count']
    all_df.to_csv(path+'outputs/rq3/overlapping/{}/groupbytopic_soft={}_roundup={}.csv'.format(source, str(soft),str(one)))
    return all_df

for soft in [False]:
    for one in [True, False]:
        for source in ['so','sse']:
            overlapping(source, soft, one)

#%%#######################################################################
#                     Naive Expertise using Tags                         #
##########################################################################

def naive_expertise(posts, files):

    ## Get expertise
    expertise = read_data('expertise_up/{}'.format(files)).set_index(['user_id','tag']).join(read_data('expertise_down/so-*').set_index(['user_id','tag']), lsuffix='_up', rsuffix='_down').reset_index()
    expertise['score'] = expertise.UpVotes - expertise.DownVotes
    expertise_1 = expertise.sort_values('score', ascending=0).groupby(['user_id']).head(1)[['user_id','tag']].rename(columns={'tag':'toptags1'}).dropna().set_index('user_id')
    expertise_3 = expertise.sort_values('score', ascending=0).groupby(['user_id']).head(3)[['user_id','tag']].rename(columns={'tag':'toptags3'}).dropna()
    expertise_5 = expertise.sort_values('score', ascending=0).groupby(['user_id']).head(5)[['user_id','tag']].rename(columns={'tag':'toptags5'}).dropna()
    expertise_3 = pd.DataFrame(expertise_3.groupby('user_id').toptags3.apply(list))
    expertise_5 = pd.DataFrame(expertise_5.groupby('user_id').toptags5.apply(list))
    joined_e = posts.join(read_data('accepted_answer_owner/{}'.format(files)).set_index('id'))
    joined_e = joined_e.join(read_data('tags/{}'.format(files)).set_index('id')).rename(columns={'tag':'post_tags'})
    joined_e = joined_e.reset_index().set_index('owneruserid').join(expertise_1).join(expertise_3).join(expertise_5)
    joined_e['toptags1'] = joined_e.apply(lambda row: checktag(row, 1), axis=1)
    joined_e['toptags3'] = joined_e.apply(lambda row: checktag(row, 3), axis=1)
    joined_e['toptags5'] = joined_e.apply(lambda row: checktag(row, 5), axis=1)
    joined_e = joined_e[['postid','toptags1','toptags3','toptags5']]
    pts = pd.concat([pd.read_csv(i) for i in glob(path+'data/generated/{}_posttopics.csv'.format(files))])
    pts.topic = pts.topic.apply(lambda x: topic_names[x])
    pts = pts[pts.score > 0.1]
    joined_e = joined_e.set_index('postid').join(pts.set_index('postid'))
    final = joined_e.groupby('topic').mean()[['toptags1','toptags3','toptags5']].reset_index()
    final.to_csv(path+'outputs/rq3/naive_general/naive_expertise_{}.csv'.format(files[:-1]))
    return final

#%%#######################################################################
#                      Question-user correlation                         #
##########################################################################

def account_expertise(combined_answers=False):
    # Reads CSV files containing topic predictions for accepted answers
    if combined_answers:
        aanswers_so = pd.read_csv(path+'rq3/data/generated/so-aanswertopics-combined.csv')
        aanswers_sse = pd.read_csv(path+'rq3/data/generated/sse-aanswertopics-combined.csv')
    else:
        aanswers_so = pd.read_csv(path+'rq3/data/generated/so-aanswertopics.csv')
        aanswers_sse = pd.read_csv(path+'rq3/data/generated/sse-aanswertopics.csv')
    df = pd.concat([aanswers_so, aanswers_sse])[['accountid','topic','score']]
    df = df.groupby(['accountid','topic']).sum()
    return df.reset_index().set_index('accountid')

def specexpertise(posts, files, combined_answers):
    df = read_data('acceptedanswerbody/{}'.format(files)).set_index('id')
    pts = pd.concat([pd.read_csv(i) for i in glob(path+'data/generated/{}_posttopics.csv'.format(files))])
    df = df.join(pts.set_index('postid')).reset_index().rename(columns={'index':'postid'})
    df = df[['postid','body','accountid','topic','score']].dropna().set_index('accountid')
    df.topic = df.topic.apply(lambda x: topic_names[x])
    df = df.join(account_expertise(combined_answers), lsuffix='_q', rsuffix='_a').reset_index()
    df['score'] = df.score_q * df.score_a
    df = df.drop(columns=['accountid','body','score_q','score_a'])
    df = df.groupby(['topic_q','topic_a']).sum()
    df = df.reset_index().drop(columns=['postid'])
    df = df.set_index(['topic_q','topic_a'])
    df = df.loc[natsorted(df.index)].reset_index()
    makedir(path+'outputs/rq3/specexpertise/')
    df.to_csv(path+'outputs/rq3/specexpertise/{}_combine={}.csv'.format(files[:-1],str(combined_answers)), index=False)
    return df

#%%#######################################################################
#                                Plotting                                #
##########################################################################

print('Plot all expertise')

def mybar(df, w=0.2, pltxlabel='xlabel', pltylabel='ylabel', plttitle='title', dp=0):

    ## Setup
    N = len(df.variable.unique())
    ind = np.arange(len(df.group.unique()))
    width = w
    clrs = sns.color_palette('husl', n_colors=N)  # a list of RGB tuples
    plt.rcParams['axes.ymargin'] = 0.01
    plt.rcParams['axes.xmargin'] = 0.07
    fig, ax = plt.subplots(figsize=(13,20))

    # Bar plots
    rects = []
    for count, row in enumerate(df.variable.unique()):
        rects.append(ax.barh(ind - width*count, df[df.variable==row].value.tolist(), width, color=clrs[count], label=row, edgecolor='black'))

    # Labels
    ax.set_ylabel('Scores')
    ax.set_yticks(ind-(width*((N-1)/2)))
    ax.set_yticklabels( df.group.unique().tolist() )
    ax.set_xlabel('', fontsize=18)
    ax.set_ylabel('', fontsize=18)
    ax.set_title('', fontsize=18)
    plt.xticks(fontsize=21)
    plt.yticks(fontsize=21)

    # Plot labels atop bars
    for rectsall in rects:
        for rect in rectsall:
            h = rect.get_width()
            ax.text(h+0.035, rect.get_y()-0.03, '%.{}f'.format(dp)%h, ha='center', va='bottom', size=21)

    # Plot
    # plt.legend(bbox_to_anchor=(-0.40, -0.08, 1.40, .102), loc=3, ncol=4, mode="expand", borderaxespad=0, prop={'size': 25})
    plt.legend(prop={'size': 25})
    return fig

## Gen vs Naive Normalised
makedir(path+'outputs/rq3/general/')
topic_names = pd.read_csv(ldapath+'clean_topics_shrt.txt',header=None)
topic_names = topic_names[topic_names[0].str.contains("Label")].reset_index(drop=True)
topic_names[0] = topic_names[0].str.replace('Label: ', '')
topic_names = topic_names.to_dict()[0]
for setname in ['so','sse']:
    for soft in [False]:
        ngvuln = sovuln.copy() if setname == 'so' else ssevuln.copy()
        # naive_gen = naive_expertise(ngvuln, '{}*'.format(setname)).set_index('topic').join(genexpertise(ngvuln, '{}*'.format(setname),soft).set_index('topic'))

        naive_gen = genexpertise(ngvuln, '{}*'.format(setname),soft).set_index('topic')

        ## Convoluted way to sort the yticks in correct order
        naive_gen = naive_gen.loc[natsorted(naive_gen.index)]
        naive_gen = naive_gen.iloc[::-1].reset_index()
        naive_gen.topic = naive_gen.topic.apply(lambda x: tdict[x])
        naive_gen = naive_gen.set_index('topic')

        ## Remove T8 if SSE
        if setname == 'sse':
            naive_gen = naive_gen.loc[~naive_gen.index.isin(['Resource Leaks (T8)'])]

        ## Combine and normalise all data and plot
        # all_naive_gen = naive_gen.toptags1.tolist() + naive_gen.toptags3.tolist() + naive_gen.toptags5.tolist()
        # naive_gen_max = np.max(all_naive_gen)
        # naive_gen_min = np.min(all_naive_gen)
        # naive_gen.toptags1 = naive_gen.toptags1.apply(lambda x: (x - naive_gen_min) / (naive_gen_max - naive_gen_min))
        # naive_gen.toptags3 = naive_gen.toptags3.apply(lambda x: (x - naive_gen_min) / (naive_gen_max - naive_gen_min))
        # naive_gen.toptags5 = naive_gen.toptags5.apply(lambda x: (x - naive_gen_min) / (naive_gen_max - naive_gen_min))
        naive_gen.a_reputation = naive_gen.a_reputation.apply(lambda x: (x - naive_gen.a_reputation.min())/(naive_gen.a_reputation.max()-naive_gen.a_reputation.min()))
        # naive_gen.columns=["Top Tags 1", "Top Tags 2", "Top Tags 3", "Reputation"]
        naive_gen.columns = ["Reputation"]
        naive_gen.to_csv(path + 'outputs/rq3/general/general_expertise_{}_soft={}_normalised.csv'.format(setname, str(soft)))
        naive_gen = naive_gen.reset_index().melt(id_vars='topic').rename(columns={'topic':'group'})
        # ngfig = mybar(naive_gen, w=0.2, dp=2, pltxlabel='Normalised Scores', pltylabel='Topic', plttitle='Normalised General Expertise (reputation) vs Naive Expertise by Topic ({})'.format(setname))
        ngfig = mybar(naive_gen, w=0.2, dp=2, pltxlabel='Normalised Scores', pltylabel='Topic',plttitle='Normalised General Expertise (reputation) by Topic ({})'.format(setname))
        ngfig.savefig(path+'outputs/rq3/general/{}_general_expertise_soft={}.png'.format(setname,str(soft)),bbox_inches="tight",dpi=300)

## Heat map for Spec Expertise
def plot_heatmap(df, files, comb_answers, ax, cbar_ax, xlabel, ylabel):
    df = specexpertise(df, files, comb_answers).pivot(index='topic_q', columns='topic_a', values='score')
    df = df[natsorted(df.columns)].copy()
    df = df.loc[natsorted(df.index)]
    df = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(df.T), columns=df.T.columns, index=df.T.index).T

    makedir(path+'outputs/rq3/specexpertise')
    df.to_csv(path+'outputs/rq3/specexpertise/heatmap_{}.csv'.format(files[:-1]))

    sns.axes_style("darkgrid")
    plt.rcParams['axes.ymargin'] = 0
    plt.rcParams['axes.xmargin'] = 0
    sns.heatmap(df, square=True, cmap="YlGnBu", vmin=0, ax=ax, cbar_ax=cbar_ax)

    ax.set_xlabel(xlabel, fontsize=31, fontweight='bold')
    ax.xaxis.labelpad = 20
    ax.set_ylabel(ylabel, fontsize=21)
    ax.set_title('', fontsize=18)
    ax.tick_params(labelsize=27, rotation=0)
    cbar_ax.tick_params(labelsize=25, rotation=0)

    cbar_ax.set_position([0.93, 0.07, 0.02, 0.875])

    ax.spines['right'].set_visible(True)
    ax.spines['right'].set_color('0')
    ax.spines['top'].set_visible(True)
    ax.spines['top'].set_color('0')
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_color('0')
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_color('0')

## Plot Heatmap as Combined Plot
def combined_heatmap():
    fig, ax_array = plt.subplots(1, 3, figsize=(21,10), gridspec_kw={'width_ratios': [24, 24, 1]})
    fig.tight_layout(w_pad=6)
    ax_ravel = np.ravel(ax_array)
    plot_heatmap(sovuln,'so*',True,ax_ravel[0], ax_ravel[2], xlabel='(a) Specific Expertise: Question topics (y-axis)\n vs. Answerers\' topics (x-axis) (SO)',ylabel=None)
    plot_heatmap(ssevuln,'sse*',True,ax_ravel[1], ax_ravel[2], xlabel='(b) Specific Expertise: Question topics (y-axis)\n vs. Answerers\' topics (x-axis) (SSE)',ylabel=None)
    
    plt.savefig(path + 'outputs/rq3/specexpertise/RQ3.png', bbox_inches="tight", dpi=300)
	
    #save_path = path + 'outputs/rq3/specexpertise/RQ3_new.pdf'
    #fig.savefig(save_path, bbox_inches="tight", edgecolor='black')

combined_heatmap()
