"""
Foundations of Natural Language Processing

Assignment 1

Please complete functions, based on their doc_string description
and instructions of the assignment. 

To test your code run:

  [hostname]s1234567 python3 s1234567.py
or
  [hostname]s1234567 python3 -i s1234567.py

The latter is useful for debugging, as it will allow you to access many
 useful global variables from the python command line

*Important*: Before submission be sure your code works _on a DICE machine_
with the --answers flag:

  [hostname]s1234567 python3 s1234567.py --answers

Also use this to generate the answers.py file you need for the interim
checker.

Best of Luck!
"""
from collections import defaultdict, Counter
from typing import Tuple, List, Any, Set, Dict, Callable

import numpy as np  # for np.mean() and np.std()
import nltk, sys, inspect
import nltk.corpus.util
from nltk import MaxentClassifier
from nltk.corpus import brown, ppattach  # import corpora

# Import LgramModel
from nltk_model import *

# Import the Twitter corpus
from twitter.twitter import *

twitter_file_ids = "20100128.txt"
assert twitter_file_ids in xtwc.fileids()

# Some helper functions

import matplotlib.pyplot as plt


def hist(hh: List[float], title: str, align: str = 'mid',
         log: bool = False, block: bool = False):
    """
  Show a histgram with bars showing mean and standard deviations
  :param hh: the data to plot
  :param title: the plot title
  :param align: passed to pyplot.hist, q.v.
  :param log: passed to pyplot.hist, q.v.  If present will be added to title
  """
    hax = plt.subplots()[1]  # Thanks to https://stackoverflow.com/a/7769497
    sdax = hax.twiny()
    hax.hist(hh, bins=30, color='lightblue', align=align, log=log)
    hax.set_title(title + (' (log plot)' if log else ''))
    ylim = hax.get_ylim()
    xlim = hax.get_xlim()
    m = np.mean(hh)
    sd = np.std(hh)
    sdd = [(i, m + (i * sd)) for i in range(int(xlim[0] - (m + 1)), int(xlim[1] - (m - 3)))]
    for s, v in sdd:
        sdax.plot([v, v], [0, ylim[0] + ylim[1]], 'r' if v == m else 'pink')
    sdax.set_xlim(xlim)
    sdax.set_ylim(ylim)
    sdax.set_xticks([v for s, v in sdd])
    sdax.set_xticklabels([str(s) for s, v in sdd])
    plt.show(block=block)


def compute_accuracy(classifier, data: List[Tuple[List, str]]) -> float:
    """
    Computes accuracy (range 0 - 1) of a classifier.
    :type classifier: e.g. NltkClassifierWrapper or NaiveBayes
    :param classifier: the classifier whose accuracy we compute.
    :param data: A list with tuples of the form (list with features, label)
    :return accuracy (range 0 - 1).
    """
    correct = 0
    for d, gold in data:
        predicted = classifier.classify(d)
        correct += predicted == gold
    return correct / len(data)


def apply_extractor(extractor_f: Callable[[str, str, str, str, str], List[Any]], data: List[Tuple[Tuple[str], str]]) \
        -> List[Tuple[List[Any], str]]:
    """
    Helper function:
    Apply a feature extraction method to a labeled dataset.
    :param extractor_f: the feature extractor, that takes as input V, N1, P, N2 (all strings) and returns a list of features
    :param data: a list with tuples of the form (id, V, N1, P, N2, label)

    :return a list with tuples of the form (list with features, label)
    """
    r = []
    for d in data:
        r.append((extractor_f(*d[1:-1]), d[-1]))
    return r


def get_annotated_tweets():
    """
    :rtype list(tuple(list(str), bool))
    :return: a list of tuples (tweet, a) where tweet is a tweet preprocessed by us,
    and a is True, if the tweet is in English, and False otherwise.
    """
    import ast
    with open("twitter/annotated_dev_tweets.txt") as f:
        return [ast.literal_eval(line) for line in f.readlines()]


class NltkClassifierWrapper:
    """
    This is a little wrapper around the nltk classifiers so that we can interact with them
    in the same way as the Naive Bayes classifier.
    """

    def __init__(self, classifier_class: nltk.classify.api.ClassifierI, train_features: List[Tuple[List[Any], str]],
                 **kwargs):
        """

        :param classifier_class: the kind of classifier we want to create an instance of.
        :param train_features: A list with tuples of the form (list with features, label)
        :param kwargs: additional keyword arguments for the classifier, e.g. number of training iterations.
        :return None
        """
        self.classifier_obj = classifier_class.train(
            [(NltkClassifierWrapper.list_to_freq_dict(d), c) for d, c in train_features], **kwargs)

    @staticmethod
    def list_to_freq_dict(d: List[Any]) -> Dict[Any, int]:
        """
        :param d: list of features

        :return: dictionary with feature counts.
        """
        return Counter(d)

    def classify(self, d: List[Any]) -> str:
        """
        :param d: list of features

        :return: most likely class
        """
        return self.classifier_obj.classify(NltkClassifierWrapper.list_to_freq_dict(d))

    def show_most_informative_features(self, n=10):
        self.classifier_obj.show_most_informative_features(n)


# End helper functions

# ==============================================
# Section I: Language Identification [60 marks]
# ==============================================

# Question 1.1 [7.5 marks]
def train_LM(corpus: nltk.corpus.CorpusReader) -> LgramModel:
    """
    Build a bigram letter language model using LgramModel
    based on the lower-cased all-alpha subset of the entire corpus

    :param corpus: An NLTK corpus

    :return: A padded letter bigram model based on nltk.model.NgramModel
    """
    # raise NotImplementedError  # remove when you finish defining this function

    # subset the corpus to only include all-alpha tokens,
    # converted to lower-case (_after_ the all-alpha check)

    # Extract the alpha tokens from the corpus and convert them into lowercase.
    corpus_tokens = [word.lower() for word in corpus.words() if word.isalpha()]

    # Put tokens into one string.
    # corpus_string = ''.join(corpus_tokens)

    # Train the LgramModel (bigram) with left and right padding on.
    lm = LgramModel(2, corpus_tokens, pad_left=True, pad_right=True)

    # Return the tokens and a smoothed (using the default estimator)
    #   padded bigram letter language model
    return lm


# Question 1.2 [7.5 marks]
def tweet_ent(file_name: str, bigram_model: LgramModel) -> List[Tuple[float, List[str]]]:
    """
    Using a character bigram model, compute sentence entropies
    for a subset of the tweet corpus, removing all non-alpha tokens and
    tweets with less than 5 all-alpha tokens, then converted to lowercase

    :param file_name: twitter file to process

    :return: ordered list of average entropies and tweets"""

    # raise NotImplementedError # remove when you finish defining this function

    # Clean up the tweet corpus to remove all non-alpha
    # tokens and tweets with less than 5 (remaining) tokens, converted
    # to lowercase
    list_of_tweets = xtwc.sents(file_name)
    tweet_entropies = []

    for tweet in list_of_tweets:

        # Clean up the Twitter corpus.
        cleaned_tweet = [word.lower() for word in tweet if word.isalpha()]
        if len(cleaned_tweet) < 5:
            continue

        # Compute the entropy for each tweet
        total_ent = 0
        for token in cleaned_tweet:
            total_ent += bigram_model.entropy(token, pad_left=True, pad_right=True, verbose=False, perItem=True)
        average_entropy = total_ent / len(cleaned_tweet)
        tweet_entropies.append((average_entropy, cleaned_tweet))

    # Sort the tweets by entropy
    tweet_entropies.sort(key=lambda x: x[0])

    # Return a list of tuples of the form: (entropy,tweet)
    #  for each tweet in the cleaned corpus, where entropy is the
    #  average per_item bigram entropy of the tokens in the tweet.
    #  The list should be sorted by entropy.
    return tweet_entropies


# Question 1.3 [3 marks]
def short_answer_1_3() -> str:
    """
    Briefly explain what left and right padding accomplish and why
    they are a good idea. Assuming you have a bigram model trained on
    a large enough sample of English that all the relevant bigrams
    have reliable probability estimates, give an example of a string
    whose average letter entropy you would expect to be (correctly)
    greater with padding than without and explain why.
   
    :return: your answer
    """
    return inspect.cleandoc("Left and right padding, represented by special symbols '<s>' and '</s>', indicate a string's beginning and end, respectively. This practice improves the accuracy of probability estimates for N-gram and L-gram models. (<s>xq</s>) includes (<s>x) and (q</s>) with padding. 'x' and 'q' are rare as initial and final letters in English. Thus, the bigram with padding will have a lower probability and lead to a higher average letter entropy.")


# Question 1.4 [3 marks]
def short_answer_1_4() -> str:
    """
    Explain the output of lm.entropy('bbq',verbose=True,perItem=True)
    See the Coursework 1 instructions for details.

    :return: your answer
    """
    return inspect.cleandoc(
        "p(b|('<s>',)) = [2-gram] 0.046511 # bigram probability of 'b' following '<s>' "
        "p(b|('b',)) = [2-gram] 0.007750 # bigram probability of 'b' following 'b'"
        "backing off for ('b', 'q') # Use a lower-order model to calculate the probability of 'q' following 'b'-> the bigram ('b', 'q') is not found in the training data"
        "p(q|()) = [1-gram] 0.000892 # unigram probability of 'q' occurring in any context"
        "p(q|('b',)) = [2-gram] 0.000092 # bigram probability of 'q' following 'b' after backing off"
        "p(</s>|('q',)) = [2-gram] 0.010636 # bigram probability of '</s>' following 'q'"
        "7.85102054894183 # the entropy of 'bbq'")


# Question 1.5 [3 marks]
def short_answer_1_5() -> str:
    """
    Inspect the distribution of tweet entropies and discuss.
    See the Coursework 1 instructions for details.

    :return: your answer
    """
    global ents
    # Uncomment the following lines when you are ready to work on this.
    # Please comment them out again or delete them before submitting.
    # Note that you will have to close the two plot windows to allow this
    #  function to return.
    # just_e = [e for (e, tw) in ents]
    # hist(just_e, "Bi-char entropies from cleaned twitter data")
    # hist(just_e, "Bi-char entropies from cleaned twitter data",
    #     log=True, block=True)
    return inspect.cleandoc("The entropy values are clustered around the mean with a significant drop-off as the entropy value increases which indicates that most tweets have a similar level of predictability in character combinations. Figure 2 reveals a wide range of higher entropy values in the distribution's tail. To classify tweets, lower entropy tweets can represent text that aligns with the character patterns in Brown corpus (English). Higher entropy tweets can represent non-English text.")


# Question 1.6 [10 marks]
def is_English(bigram_model: LgramModel, tweet: List[str]) -> bool:
    """
    Classify if the given tweet is written in English or not.

    :param bigram_model: the bigram letter model trained on the Brown corpus
    :param tweet: the tweet
    :return: True if the tweet is classified as English, False otherwise
    """
    # raise NotImplementedError # remove when you finish defining this function

    # Check if the tweet is alphabet and the length of the tweet is over 5 tokens
    if len(tweet) < 5 or not all(word.isalpha() for word in tweet):
        return False

    # Calculate average word entropy in the tweet
    word_entropies = [bigram_model.entropy(word, pad_left=True, pad_right=True, verbose=False, perItem=True)
                      for word in tweet]
    average_ent = sum(word_entropies)/len(word_entropies)

    # Define the English entropy based on the distribution grams
    english_ent = 4.5

    return average_ent <= english_ent


# Question 1.7 [16 marks]
def essay_question():
    """

    THIS IS AN ESSAY QUESTION WHICH IS INDEPENDENT OF THE PREVIOUS
    QUESTIONS ABOUT TWITTER DATA AND THE BROWN CORPUS!

    See the Coursework 1 instructions for a question about the average
    per word entropy of English.
    1) Name 3 problems that the question glosses over
    2) What kind of experiment would you perform to get a better estimate
       of the per word entropy of English?

    There is a limit of 400 words for this question.
    :return: your answer
    """
    return inspect.cleandoc("""Your answer""")


#############################################
# SECTION II - RESOLVING PP ATTACHMENT AMBIGUITY
#############################################

# Question 2.1 [15 marks]
class NaiveBayes:
    """
    Naive Bayes model with Lidstone smoothing (parameter alpha).
    """

    def __init__(self, data: List[Tuple[List[Any], str]], alpha: float):
        """
        :param data: A list with tuples of the form (list with features, label)
        :param alpha: \alpha value for Lidstone smoothing
        """
        self.vocab = self.get_vocab(data)
        self.alpha = alpha
        self.prior, self.likelihood = self.train(data, alpha, self.vocab)

    @staticmethod
    def get_vocab(data: List[Tuple[List[Any], str]]) -> Set[Any]:
        """
        Compute the set of all possible features from the (training) data.
        :param data: A list with tuples of the form (list with features, label)

        :return: The set of all features used in the training data for all classes.
        """
        # raise NotImplementedError  # remove when you finish defining this function
        vocab = set()
        for features, _ in data:
            vocab.update(features)

        return vocab

    @staticmethod
    def train(data: List[Tuple[List[Any], str]], alpha: float, vocab: Set[Any]) -> Tuple[Dict[str, float],
    Dict[str, Dict[
        Any, float]]]:
        """
        Estimates the prior and likelihood from the data with Lidstone smoothing.

        :param data: A list of tuples ([f1, f2, ... ], c) with
                    the first element being a list of features and
                    the second element being its class.
        :param alpha: alpha value for Lidstone smoothing
        :param vocab: The set of all features used in the training data
                      for all classes.

        :return: Two dictionaries: the prior and the likelihood
                 (in that order).
        The returned values should relate as follows to the probabilities:
            prior[c] = P(c)
            likelihood[c][f] = P(f|c)
        """
        assert alpha >= 0.0

        # Initialize dictionaries
        class_counts = defaultdict(int)
        feature_counts = defaultdict(lambda: defaultdict(int))
        total_count = 0

        # Count occurrences of classes and features
        for features, label in data:
            class_counts[label] += 1
            total_count += 1
            for feature in features:
                feature_counts[label][feature] += 1

        # Calculate prior probabilities, map each class to its prior probabilities P(c).
        prior_probabilities = {label: count / total_count for label, count in class_counts.items()}

        # Initialize likelihood dictionary
        likelihood_dict = defaultdict(dict)

        # Calculate likelihood probabilities using Lidstone smoothing,
        # map each feature to its conditional probabilities P(f|c).
        for label in class_counts:
            total_features = sum(feature_counts[label].values())
            denominator = total_features + alpha * len(vocab)
            for feature in vocab:
                count = feature_counts[label][feature]
                likelihood_dict[label][feature] = (count + alpha) / denominator

        return prior_probabilities, likelihood_dict

    def prob_classify(self, d: List[Any]) -> Dict[str, float]:
        """
        Compute the probability P(c|d) for all classes.
        :param d: A list of features.

        :return: The probability p(c|d) for all classes as a dictionary.
        """
        joint_probabilities = dict()

        # Calculate the joint probability P(c,d) for each class
        for c in self.prior:
            # Start with the prior probability P(c)
            joint_prob_c_d = self.prior[c]
            for feature in d:
                if feature in self.vocab:
                    # Multiply by P(f|c) if feature is in vocab, else treat as neutral (1)
                    joint_prob_c_d *= self.likelihood[c].get(feature, 1)
            joint_probabilities[c] = joint_prob_c_d

        # Normalization to get P(c|d)
        total_joint_prob = sum(joint_probabilities.values())
        if total_joint_prob == 0:
            # Avoid division by zero if no features are in the vocab
            return {c: 0 for c in self.prior}
        conditional_probabilities = {c: joint_prob_c_d / total_joint_prob for c, joint_prob_c_d in
                                     joint_probabilities.items()}

        return conditional_probabilities

    def classify(self, d: List[Any]) -> str:
        """
        Compute the most likely class of the given "document" with ties broken arbitrarily.
        :param d: A list of features.

        :return: The most likely class.
        """
        # Get the probabilities for each class
        class_probabilities = self.prob_classify(d)

        # Find the class with the highest probability
        most_likely_class = max(class_probabilities, key=class_probabilities.get)

        return most_likely_class


# Question 2.2 [15 marks]
def open_question_2_2() -> str:
    """
    See the Coursework 1 instructions for detail of the following:
    1) The differences in accuracy between the different ways
        to extract features?
    2) The difference between Naive Bayes vs Logistic Regression
    3) An explanation of a binary feature that returns 1
        if V=`imposed' AND N_1 = `ban' AND P=`on' AND N_2 = `uses'.

    Limit: 150 words for all three sub-questions together.
    """
    return inspect.cleandoc("""The individual features provide basic information about the sentence structure but can not capture the relationships 
between these phrases. When combining these features ([V=N1=P=N2]), there's a significant increase in accuracy, which 
suggests that the interaction between the phrases is critical for predicting the correct class. It implies that the task 
requires how these elements relate to each other within the sentence structure to make accurate predictions.

The Naive Bayes model accuracy is 79.49987620698192%, slightly lower than the logistic regression. Naive Bayes assumes 
all the features are independent given the class label. Logistic regression captures the iteration between features.

I would be against it. A binary feature captures a rule that may strongly indicate the correct classification. It may 
not apply to sparse data if the rule is too specific. Additionally, if these conditions rarely occur together, the 
feature may not contribute much to the model's performance.""")


# Feature extractors used in the table:

def feature_extractor_1(v: str, n1: str, p: str, n2: str) -> List[Any]:
    return [("v", v)]


def feature_extractor_2(v: str, n1: str, p: str, n2: str) -> List[Any]:
    return [("n1", n1)]


def feature_extractor_3(v: str, n1: str, p: str, n2: str) -> List[Any]:
    return [("p", p)]


def feature_extractor_4(v: str, n1: str, p: str, n2: str) -> List[Any]:
    return [("n2", n2)]


def feature_extractor_5(v: str, n1: str, p: str, n2: str) -> List[Any]:
    return [("v", v), ("n1", n1), ("p", p), ("n2", n2)]


# Question 2.3, part 1 [10 marks]
def your_feature_extractor(v: str, n1: str, p: str, n2: str) -> List[Any]:
    """
    Takes the head words and produces a list of features. The features may
    be of any type as long as they are hashable.

    :param v: The verb.
    :param n1: Head of the object NP.
    :param p: The preposition.
    :param n2: Head of the NP embedded in the PP.

    :return: A list of features produced by you.
    """
    features=[]

    # Basic features
    features.append(f'verb:{v}')
    features.append(f'np1:{n1}')
    features.append(f'prep:{p}')
    features.append(f'np2:{n2}')

    # Pairwise combinations to capture local relationships
    features.append(f'verb_np1:{v}_{n1}')
    features.append(f'verb_prep:{v}_{p}')
    features.append(f'verb_np2:{v}_{n2}')
    features.append(f'np1_prep:{n1}_{p}')
    features.append(f'prep_np2:{p}_{n2}')
    features.append(f'np1_np2:{n1}_{n2}')

    # Three-word combinations to capture broader context
    features.append(f'verb_np1_prep:{v}_{n1}_{p}')
    features.append(f'np1_prep_np2:{n1}_{p}_{n2}')

    # Potential syntactic features for prepositions
    common_followers = {'put': 'on', 'take': 'off', 'give': 'to'}
    if v in common_followers and common_followers[v] == p:
        features.append(f'common_verb_prep_combo:{v}_{p}')

    # Length-based features: the length of a noun phrase may influence attachment decisions
    features.append(f'np1_length:{len(n1)}')
    features.append(f'np2_length:{len(n2)}')

    return features

# Question 2.3, part 2 [10 marks]
def open_question_2_3() -> str:
    """
    Briefly describe your feature templates and your reasoning for them.
    Pick three examples of informative features and discuss why they make sense or why they do not make sense
    and why you think the model relies on them.

    There is a limit of 300 words for this question.
    """
    return inspect.cleandoc("""Your answer""")


"""
Format the output of your submission for both development and automarking. 
!!!!! DO NOT MODIFY THIS PART !!!!!
"""


def answers():
    # Global variables for answers that will be used by automarker
    global ents, lm, top10_ents, bottom10_ents
    global answer_open_question_2_2, answer_open_question_2_3
    global answer_short_1_4, answer_short_1_5, answer_short_1_3, answer_essay_question

    global naive_bayes
    global acc_extractor_1, naive_bayes_acc, lr_acc, logistic_regression_model, dev_features
    global dev_tweets_preds

    print("*** Part I***\n")

    print("*** Question 1.1 ***")
    print('Building Brown news bigram letter model ... ')
    lm = train_LM(brown)
    print('Letter model built')

    print("*** Question 1.2 ***")
    ents = tweet_ent(twitter_file_ids, lm)

    top10_ents = ents[:10]
    bottom10_ents = ents[-10:]

    answer_short_1_3 = short_answer_1_3()
    print("*** Question 1.3 ***")
    print(answer_short_1_3)

    answer_short_1_4 = short_answer_1_4()
    print("*** Question 1.4 ***")
    print(answer_short_1_4)

    answer_short_1_5 = short_answer_1_5()
    print("*** Question 1.5 ***")
    print(answer_short_1_5)

    print("*** Question 1.6 ***")
    all_dev_ok = True
    dev_tweets_preds = []
    for tweet, gold_answer in get_annotated_tweets():
        prediction = is_English(lm, tweet)
        dev_tweets_preds.append(prediction)
        if prediction != gold_answer:
            all_dev_ok = False
            print("Missclassified", tweet)
    if all_dev_ok:
        print("All development examples correctly classified! "
              "We encourage you to test and tweak your classifier on more tweets.")

    answer_essay_question = essay_question()
    print("*** Question 1.7 (essay question) ***")
    print(answer_essay_question)

    print("*** Part II***\n")

    print("*** Question 2.1 ***")
    naive_bayes = NaiveBayes(apply_extractor(feature_extractor_5, ppattach.tuples("training")), 0.1)
    naive_bayes_acc = compute_accuracy(naive_bayes, apply_extractor(feature_extractor_5, ppattach.tuples("devset")))
    print(f"Accuracy on the devset: {naive_bayes_acc * 100}%")

    print("*** Question 2.2 ***")
    answer_open_question_2_2 = open_question_2_2()
    print(answer_open_question_2_2)

    print("*** Question 2.3 ***")
    training_features = apply_extractor(your_feature_extractor, ppattach.tuples("training"))
    dev_features = apply_extractor(your_feature_extractor, ppattach.tuples("devset"))
    logistic_regression_model = NltkClassifierWrapper(MaxentClassifier, training_features, max_iter=10)
    lr_acc = compute_accuracy(logistic_regression_model, dev_features)

    print("30 features with highest absolute weights")
    logistic_regression_model.show_most_informative_features(30)

    print(f"Accuracy on the devset: {lr_acc * 100}")

    answer_open_question_2_3 = open_question_2_3()
    print("Answer to open question:")
    print(answer_open_question_2_3)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--answers':
        from autodrive_embed import run, carefulBind
        import adrive1

        with open("userErrs.txt", "w") as errlog:
            run(globals(), answers, adrive1.extract_answers, errlog)
    else:
        answers()
