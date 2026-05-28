"""
NLP DOMAIN
==========
Text analytics modules for Nexlytics.

Components:
- SentimentAnalyzer: lexicon + rule-based (extensible to BERT)
- TopicModeler: LDA / NMF for topic discovery
- KeywordExtractor: TF-IDF + RAKE
- TextClassifier: scikit-learn pipeline (TF-IDF + classifier)
- NamedEntityExtractor: regex + dictionary based (extensible to spaCy)
- AspectSentimentAnalyzer: aspect-based sentiment

CRISP-DM compliance:
- Data Understanding: text statistics, language detection
- Preparation: cleaning, tokenization, stopword removal, lemmatization
- Modeling: TF-IDF + classification / topic modeling
- Evaluation: confusion matrix, F1, topic coherence
"""
from .sentiment_analyzer import SentimentAnalyzer
from .topic_modeler import TopicModeler
from .keyword_extractor import KeywordExtractor
from .text_classifier import TextClassifier
from .ner_extractor import NamedEntityExtractor
from .aspect_sentiment import AspectSentimentAnalyzer
from .text_preprocessor import TextPreprocessor

__all__ = [
    "SentimentAnalyzer",
    "TopicModeler",
    "KeywordExtractor",
    "TextClassifier",
    "NamedEntityExtractor",
    "AspectSentimentAnalyzer",
    "TextPreprocessor",
]
