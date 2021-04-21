from nltk.stem import PorterStemmer
import pandas as pd
import ahocorasick
import re
from collections import Counter
from progressbar import progressbar as pb
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from tqdm import tqdm
tqdm.pandas()

## Import importlib resources
try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources
from . import data

## Import StringIO
try:
    from StringIO import StringIO ## for Python 2
except ImportError:
    from io import StringIO ## for Python 3

class TextPreprocess():

    def __init__(self, strlist=None):
        """Text preprocessing module
        Args:
            strlist (list): list of target strings. If unspecified, it uses
            a default list of vulnerability-related target phrases

        Use strmatch() to obtain all strings that were matched by the
        aho-corasick algorithm
        """

        # Stemmer
        self.ps = PorterStemmer()
        self.origdict = dict()

        # Read target strings from input file
        try:
            self.strlist = pd.read_csv(strlist, header=None)[0].str.strip().to_list()
        except:
            strlist = StringIO(pkg_resources.read_text(data, 'sec.txt'))
            self.strlist = pd.read_csv(strlist, header=None)[0].str.strip().to_list()

        # Read UK/US spelling dictionary
        uk_us_csv = pkg_resources.read_text(data, 'uk_us.csv')
        self.ukus = pd.read_csv(StringIO(uk_us_csv)).set_index("us")['uk'].to_dict()
        self.ukus.update(pd.read_csv(StringIO(uk_us_csv)).set_index("uk")['us'].to_dict())

        # Add UK/US spelling variations and create dict for future reference
        for i in list(self.strlist):
            altspelling = self.reverse_ukus(i)
            self.strlist.append(altspelling)
            self.origdict[self.stem(i)] = i
            self.origdict[self.stem(altspelling)] = i

        # Stem words in target string list
        self.strlist = [self.stem(i) for i in self.strlist]

        # Remove duplicate target strings
        self.strlist = set(self.strlist)

        # Create aho-corasick automaton using target strings
        self.A = self.makeaho(self.strlist)

        # Small words
        self.smallwords = [i for i in self.origdict.keys() if len(i) < 4]


    def makeaho(self, alist):
        """Make automaton for aho-corasick fast multi-string search
        Args:
            a (list): list of target strings
        Returns:
            object: ahocorasick automaton object
        """
        A = ahocorasick.Automaton()
        for idx, key in enumerate(alist):
            A.add_word(key, (idx, key))
        A.make_automaton()
        return A

    def preprocess(self, s):
        """Preprocess text - Removes code blocks and special char
        Args:
            s (str): string to be preprocessed
        Returns:
            str: preprocessed text joined by ' ' delimiter
        """

        # Remove code blocks
        s = re.sub('<pre>(.|\n)*?<\/pre>', ' ', s)
        s = re.sub('<code>(.|\n)*?<\/code>', ' ', s)
        s = re.sub('<samp>(.|\n)*?<\/samp>', ' ', s)
        s = re.sub('<.*?>|\'', ' ', s)

        # Remove certain special characters
        s = s.replace('\"', "''").replace('\"', "''").replace('\'', '').replace(': ', ' ').replace(', ', ' ').\
            replace('; ', ' ').replace('. ', ' ').replace('(', '').replace(')', '').replace('\n', '').strip().lower()

        # Check empty string
        if s == "": return ""

        # Remove last char if letter or number
        if not s[-1].isalnum(): s = s[:-1]
        return s

    def stem(self, t):
        """Convenience function for stemming multiple words delimited by ' '.
           Only stems words that are length > 3
        Args:
            t (str): string of words (tokens defined with delimiter ' ')
        Returns:
            str: string of stemmed words
        """
        return ' '.join([self.ps.stem(i) if len(i) > 3 else i for i in t.split() ])

    def reverse_ukus(self, phrase, delim=' '):
        """Reverse UK US spelling of a phrase
        Args:
            phrase (str): string of words where spelling is to be reversed
            delim (str): phrase delimiter. Could be ' ', '_', etc.
        Returns:
            str: string of words joined by given delimiter with the spelling
            reversed from UK to US or US to UK
        """

        ret = []
        for i in phrase.split():
            if i in self.ukus: ret.append(self.ukus[i])
            else: ret.append(i)
        return delim.join(ret)

    def strmatch(self, text, process=False, stem=False, reformat='summary'):
        """Match words (WARNING: Enabling stemming decreases speed)
        Args:
            text (str): search target words in text
            process (bool): perform preprocessing on text
            stem (bool): perform stemming on text
            reformat (str): how to format output.
                'raw' returns raw aho-corasick output
                'index': returns raw string matches with start/end indexes
                'words': returns raw words (as orignal, unstemmed words)
                'summary': unique_count, raw_count, words, len, and ratio.
                           Only keeps longest string if a substring exists.
                           e.g. 'time bomb' will only count 'time bomb' and not
                          'bomb'.
                'full': same as summary
        Returns:
            list: aho-corasick results based on reformat mode
        """
        t = text
        if process: t = self.preprocess(t)
        if stem: t = self.stem(t)
        ret = list(self.A.iter(t))

        # Only keep smallword if does not occur inside another word
        for sw in self.smallwords:
            if re.subn('[^a-zA-Z0-9]{}[^a-zA-Z0-9]'.format(sw), '', t)[1] == 0:
                ret = [i for i in ret if i[1][1] != sw]

        if reformat == 'raw':
            return ret

        if reformat == 'index':
            return [(i[0]-len(i[1][1])+1,i[0],i[1][1]) for i in ret]

        if reformat == 'words':
            return [self.origdict[i[1][1]] for i in ret]

        if reformat == 'summary' or reformat == 'full':

            # Count all string matches
            c = Counter([i[1][1] for i in ret])

            # If string match (x) is a substring of another string match (y),
            # decrease the string count of x by the string count of y
            string_list = sorted(list(c.items()), key=lambda s: len(s[0]), reverse=True)
            out = []
            for s in string_list:
                substr = False
                for o in out:
                    if s[0] in o[0]:
                        c[s[0]] -= c[o[0]]
                        if c[s[0]] == 0: del c[s[0]]
                        substr = True
                if not substr: out.append(s)

            # Set original strings (unstemmed)
            c_n = dict()
            for i in c.items():
                c_n[self.origdict[i[0]]] = i[1]

            if reformat == 'summary' or reformat == 'full':
                r1 = len(c_n)
                r2 = sum(c_n.values())
                r3 = c_n.keys()
                r4 = len(text.split())
                r5 = r2 / r4
                return (r1,r2,r3,r4,r5)

        raise ValueError('reformat should be set to "raw", "index", "words", "summary" or "full"')

    def transform_df(self, df, process=False, stop=False, stem=False, tags=False, reformat='summary', columns=['title','question','answers']):
        """Convenience function using dataframe with columns:
            tags (str), title (str), question (str), answers (str)
           See strmatch() for additional argument details
           If reformat = 'full', then text is preprocessed and stemmed in-place.
        """
        if tags or "tags" in reformat or reformat=='full':
            df['tags'] = df.tags.progress_apply(lambda x: ' '.join(x.split('|')))
        if process or "process" in reformat or reformat=='full':
            for col in columns:
                df.loc[:,col] = df[col].progress_apply(self.preprocess)
        if stop or "stop" in reformat or reformat=='full':
            en_stop = set(stopwords.words('english'))
            en_stop.update(['use', 'like', 'tri', 'get', 'set', 'way', 'may', 'would', 'could', 'might', 'also'])
            for col in columns:
                df[col] = df[col].progress_apply(lambda x: ' '.join([i for i in x.split() if not i in en_stop]))
        if stem or "stem" in reformat or reformat=='full':
            for col in columns:
                df.loc[:,col] = df[col].progress_apply(self.stem)
        if 'only' in reformat: return df
        process=False
        stem=False

        column = []
        for i in pb(df.itertuples()):
            text = "{} {} {} {}".format(' '.join(i.tags.split('|')), i.title, i.question, i.answers)
            column.append(self.strmatch(text, process=process, stem=stem, reformat=reformat))
        if reformat == 'summary' or reformat == 'full':
            df['uniq'] = [i[0] for i in column]
            df['raw'] = [i[1] for i in column]
            df['words'] = ['|'.join(i[2]) for i in column]
            df['len'] = [i[3] for i in column]
            df['ratio'] = [i[4] for i in column]
        else:
            df['strmatch'] = column
        return df
