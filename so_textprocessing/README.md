# Python module for string matching
ONLY TESTED WITH PYTHON 3.5+.

# Installation
```
pip install so-processing
```

# Usage
```
df = pd.read_parquet('sse.parquet')
```
Text is stemmed and preprocessed

| postid | tags             | title                                      | question                                                     | answers                                                      |
| ------ | ---------------- | ------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 12041  | javascript\|html | are “man in the middle” attack extrem rare | i am wonder if there is any cold hard real world data to back up that assert -- are man in the middl attack actual rare in the real world base on gather data from actual intrus or secur incid | my favorit current resourc for cold hard real world data is the verizon 2011 data breach investig report an excerpt from page 69 of the report action the top three threat action categori were hack malwar and social the most common type of hack action use were the use of stolen login credenti exploit backdoor and man-in-the-middl attack from read that i infer that its a secondari action use onc somebodi has a foothold in the system but the dutch high tech crime unit data say its quit credibl for concern of the 32 data breach that made up their statist 15 involv mitm action |

```
import so_textprocessing as stp
tp = stp.TextPreprocess(strlist='targetwords.txt')
df = tp.transform_df(df)
```
Adds new columns. transform_df() uses reformat='summary' by default.

| postid | tags             | title                                      | question                                                     | answers                                                      | uniq | raw  | words                                                   | len  | ratio  |
| ------ | ---------------- | ------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ---- | ---- | ------------------------------------------------------- | ---- | ------ |
| 12041  | javascript\|html | are “man in the middle” attack extrem rare | i am wonder if there is any cold hard real world data to back up that assert -- are man in the middl attack.... | my favorit current resourc for cold hard real world data is the verizon 2011 data... | 4    | 7    | man in the middle\| attack\|malware \|man-in-the-middle | 151  | 0.0463 |

# Reformat Options
```df = tp.transform_df(df, reformat='raw')```
returns raw aho-corasick output

```df = tp.transform_df(df, reformat='index')```
returns raw string matches with start/end indexes

```df = tp.transform_df(df, reformat='words')```
returns list of raw words (as orignal, unstemmed words)

```df = tp.transform_df(df, reformat='summary')```
(default): returns unique_count, raw_count, words, len and ratio. Only keeps longest string if a substring exists.  e.g. 'time bomb' will only count 'time bomb' and not 'bomb'.

```df = tp.transform_df(df, reformat='full')```
same as summary, but when used in transform_df, it also processes+stems the text of the dataframe. Slowest option.
