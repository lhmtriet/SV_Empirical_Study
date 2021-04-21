import os
import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

sns.set()

sns.set_style("whitegrid")
# sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman', 'font.color': 'black'})

matplotlib.rcParams['font.family'] = ['Times New Roman']
matplotlib.rcParams['text.color'] = '#000000'
matplotlib.rcParams['font.weight'] = 'normal'

path = os.getcwd()+'/'

# Make directory if not present
def makedir(directory):
	if not os.path.exists(directory):
		os.makedirs(directory)

# Graph the percentage of question/answer labels
def source_comparison(label_type):
	df_so = pd.read_csv(path + "rq4/data/rq4_sample_so.csv", encoding = "ISO-8859-1")
	df_sse = pd.read_csv(path + "rq4/data/rq4_sample_sse.csv", encoding = "ISO-8859-1")

	no_sample = 385

	df_so = df_so.sample(n=no_sample, random_state=42)
	df_sse = df_sse.sample(n=no_sample, random_state=42)

	df = pd.concat([df_so, df_sse])

	# Get counts of labels
	so_ans_count, so_ques_count, sse_ans_count, sse_ques_count = 0, 0, 0, 0
	for index, row in df_so.iterrows():
		so_ans_count += len(str(row['answer_label']).split(','))
		so_ques_count += len(str(row['question_label']).split(','))
	for index, row in df_sse.iterrows():
		sse_ans_count += len(str(row['answer_label']).split(','))
		sse_ques_count += len(str(row['question_label']).split(','))

	# Compile statistics
	question_cats = ['HT', 'DC', 'Er', 'DH', 'Co', 'Re', 'Nv']
	answer_cats = ['DC', 'Co', 'Ex', 'Er', 'AT', 'ES', 'CS', 'SA']
	question_df = pd.DataFrame(columns=['Question Type', 'count', 'Percentage (%)', 'source'])
	answer_df = pd.DataFrame(columns=['Answer Type', 'count', 'Percentage (%)', 'source'])
	for q in question_cats:
		question_df.loc[len(question_df)] = [q, 0, 0, "SO"]
		question_df.loc[len(question_df)] = [q, 0, 0, "SSE"]
		for index, row in df.iterrows():
			if q in str(row['question_label']):
				question_df.loc[(question_df['Question Type'] == q) & (question_df['source'] == str(row['source']).upper()), 'count'] += 1
		question_df.loc[(question_df['Question Type'] == q) & (question_df['source'] == 'SSE'), 'Percentage (%)'] = 100* question_df.loc[(question_df['Question Type'] == q) & (question_df['source'] == 'SSE'), 'count'] / sse_ques_count
		question_df.loc[(question_df['Question Type'] == q) & (question_df['source'] == 'SO'), 'Percentage (%)'] = 100* question_df.loc[(question_df['Question Type'] == q) & (question_df['source'] == 'SO'), 'count'] / so_ques_count
	for a in answer_cats:
		answer_df.loc[len(answer_df)] = [a, 0, 0, "SO"]
		answer_df.loc[len(answer_df)] = [a, 0, 0, "SSE"]
		for index, row in df.iterrows():
			if a in str(row['answer_label']):
				answer_df.loc[(answer_df['Answer Type'] == a) & (answer_df['source'] == str(row['source']).upper()), 'count'] += 1
		answer_df.loc[(answer_df['Answer Type'] == a) & (answer_df['source'] == 'SSE'), 'Percentage (%)'] = 100* answer_df.loc[(answer_df['Answer Type'] == a) & (answer_df['source'] == 'SSE'), 'count'] / sse_ans_count
		answer_df.loc[(answer_df['Answer Type'] == a) & (answer_df['source'] == 'SO'), 'Percentage (%)'] = 100* answer_df.loc[(answer_df['Answer Type'] == a) & (answer_df['source'] == 'SO'), 'count'] / so_ans_count
	# Merge confirmation and disconfirmation
	answer_df.loc[(answer_df['Answer Type'] == 'Co') & (answer_df['source'] == 'SSE'), 'Percentage (%)'] += float(answer_df.loc[(answer_df['Answer Type'] == 'DC') & (answer_df['source'] == 'SSE'), 'Percentage (%)'])
	answer_df.loc[(answer_df['Answer Type'] == 'Co') & (answer_df['source'] == 'SSE'), 'Answer Type'] = 'DC/Co'
	answer_df.loc[(answer_df['Answer Type'] == 'Co') & (answer_df['source'] == 'SO'), 'Percentage (%)'] += float(answer_df.loc[(answer_df['Answer Type'] == 'DC') & (answer_df['source'] == 'SO'), 'Percentage (%)'])
	answer_df.loc[(answer_df['Answer Type'] == 'Co') & (answer_df['source'] == 'SO'), 'Answer Type'] = 'DC/Co'
	answer_df = answer_df[answer_df['Answer Type'] != 'DC']

	makedir(path+'outputs/rq4/')
	if label_type == 'question':
		print("Plotting question types")
		ax = sns.barplot(x="Question Type", y="Percentage (%)", hue="source", palette='husl', data=question_df)
		ax.legend().set_title('')
		ax.set_ylim([0,50.1])
		ax.set_xlabel("Question Type", fontweight='bold')
		ax.set_ylabel("Percentage (%)", fontweight='bold')
		fig = ax.get_figure()
		fig.savefig(path+'outputs/rq4/question_source_comparison.png', bbox_inches="tight", dpi=1200)
	elif label_type == 'answer':
		print("Plotting answer types")
		# ax = sns.barplot(x="Answer Type", y="Percentage (%)", hue="source", palette='husl', data=answer_df)
		# ax.legend().set_title('')
		# ax.set_xlabel("Answer Type", fontweight='bold')
		# ax.set_ylabel("Percentage (%)", fontweight='bold')
		# ax.set_ylim([0,30.1])

		answer_map = {'DC/Co': "(Dis-)Confirmation (DC/Co)", 'Ex': 'Explanation (Ex)', 'Er': 'Error (Er)',
					  'AT': 'Action to Take (AT)',
					  'ES': 'External Source (ES)', 'CS': 'Code Sample (CS)', 'SA': 'Self-Answer (SA)'}

		answer_df['Answer Type'] = answer_df['Answer Type'].map(answer_map)

		plt.rcParams['axes.ymargin'] = 0.01
		plt.rcParams['axes.xmargin'] = 0.01

		fig = plt.figure(linewidth=1, edgecolor='black')

		ax = sns.barplot(y="Answer Type", x="Percentage (%)", hue="source", palette='husl', data=answer_df)
		ax.legend().set_title('')

		ax.set_ylabel("Answer Type", fontweight='bold', labelpad=-5, fontsize=16)

		ax.set_xlabel("Percentage (%)", fontweight='bold', fontsize=16)

		ax.xaxis.labelpad = 8
		ax.yaxis.labelpad = 2
		
		#for i, v in enumerate(answer_df["Percentage (%)"]):
		#		ax.text(v, i, " "+str(v), color='blue', va='center', fontweight='bold')		

		ax.tick_params(labelsize=15, rotation=0)

		ax.set_xlim([0, 30.1])

		#plt.legend(prop={'size': 15})

		fig.savefig(path+'outputs/rq4/RQ4.png', bbox_inches="tight", dpi=1200)
		
		#save_path = path + 'outputs/rq4/RQ4_new.pdf'
            #
		#fig.savefig(save_path, bbox_inches="tight", edgecolor='black')

if __name__ == '__main__':
	source_comparison('question')
	source_comparison('answer')
