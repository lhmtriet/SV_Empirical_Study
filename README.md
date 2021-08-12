# SV_Empirical_Study
Reproduction package for the paper "A Large-scale Study of Security Vulnerability Support on Developer Q&A Websites", accepted for publication at the [Evaluation and Assessment in Software Engineering (EASE) 2021](https://conf.researchr.org/details/ease-2021/ease-2021-papers/2/A-Large-scale-Study-of-Security-Vulnerability-Support-on-Developer-Q-A-Websites). Preprint: https://arxiv.org/abs/2008.04176

It is noted that our reproduction package has been tested in Python 3.6.4. If you cannot run .sh file in Windows operating system, you can copy and run each command specified in .sh files in cmd.

Because of the large size of the data, please refer to this link to download the data in the `data` folder: https://figshare.com/s/ddfc16eb65262a73dbba

## Steps to run code

Run `run.sh` in the root directory. The requirements at the top of the file must be run before any other code.

From the second run onwards, you can also run each RQ individually (see `rqs.sh`). See each folder for more details. It is also noted that RQ2, RQ3 and RQ4 require the data generated by RQ1, so RQ1 must be run before RQ2, RQ3 and RQ4. To ensure correct filepaths, run all code from this root directory.

# Results
+ The tags and keywords used in the work are located in:
	+ Vulnerability keywords: data/vuln_keywords.txt
	+ SO tags: data/so_tags.txt
	+ SSE tags: data/sse_tags.txt

+ The 13 topics are located in: rq2/top_13/, in which:
	+ The trained LDA model is in: lda.model
	+ The topic names are in: clean_topics.txt
	+ The top documents of each topic are in: top_docs/

+ The mapping between the 13 identified topics and CWEs is located in: outputs/Topics_2_CWE_mapping.xlsx

# Visualization mentioned in the paper:
+ RQ1: outputs/rq1/sparkline_plots/
+ RQ2: outputs/rq2/RQ2.png
+ RQ3: outputs/rq3/specexpertise/RQ3.png
+ RQ4: outputs/rq4/RQ4.png
