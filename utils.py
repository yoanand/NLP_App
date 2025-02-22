import re
import base64
from collections import Counter
from io import BytesIO
import spacy
import matplotlib.pyplot as plt
import nltk
from gensim import corpora, models
from googletrans import Translator
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer
from textblob import TextBlob
from wordcloud import WordCloud
from urllib.parse import quote

nlp = spacy.load("en_core_web_sm")
    
class NLPUtils:
    """Advanced NLP Utilities with robust error handling and optimizations."""

    @staticmethod
    def _download_nltk_resources():
        resources = ['punkt', 'stopwords', 'averaged_perceptron_tagger_eng', 'wordnet']
        for resource in resources:
            try:
                nltk.data.find(resource)
            except LookupError:
                nltk.download(resource, quiet=True)

    @staticmethod
    def sentiment_analysis(text):
        if not text.strip():
            return "Input text is empty."
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        if polarity > 0:
            return "Positive"
        elif polarity < 0:
            return "Negative"
        return "Neutral"

    
    @staticmethod
    def named_entity_recognition(text):
        """Perform named entity recognition using spaCy instead of NLTK"""
        try:
            # Use spaCy for NER
            doc = nlp(text)
            # Extract named entities
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            return entities if entities else [("No entities found", "NONE")]
            
        except Exception as e:
            return [("Error", str(e))]


    
    @staticmethod
    def word_frequency_analysis(text):
        """Analyze word frequency with improved formatting"""
        try:
            # Convert to lowercase and remove punctuation
            text = re.sub(r'[^\w\s]', '', text.lower())
            
            # Split into words
            words = text.split()
            
            # Remove common stop words
            stop_words = set(nltk.corpus.stopwords.words('english'))
            words = [word for word in words if word not in stop_words 
                    and len(word) > 1]  # Filter out single characters
            
            # Count frequencies and return all words
            word_freq = Counter(words)
            
            # Sort by frequency (most common first)
            return dict(sorted(word_freq.items(), 
                             key=lambda x: x[1], 
                             reverse=True))
            
        except Exception as e:
            return {"error": str(e)}


    @staticmethod
    def spelling_correction(text):
        if not text.strip():
            return "Input text is empty."
        blob = TextBlob(text)
        return str(blob.correct())

    
    @staticmethod
    def text_similarity(text1, text2):
        if not text1.strip() or not text2.strip():
            return "One or both texts are empty."
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        similarity = (tfidf_matrix * tfidf_matrix.T).A[0, 1]
        return similarity
    
    @staticmethod
    def keyword_extraction(text, top_n=5):
        if not text.strip():
            return ["No text provided."]
        
        # Adjust max_df and min_df values
        vectorizer = TfidfVectorizer(stop_words="english", max_df=1.0, min_df=1)
        
        # Fit and transform the text data
        tfidf_matrix = vectorizer.fit_transform([text])
        keywords = [(word, tfidf_matrix[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
        return sorted(keywords, key=lambda x: x[1], reverse=True)[:top_n]


    @staticmethod
    def topic_modeling(text, num_topics=2):
        if not text.strip():
            return []
        words = [word for word in word_tokenize(text.lower()) if word.isalpha()]
        dictionary = corpora.Dictionary([words])
        corpus = [dictionary.doc2bow(words)]
        lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)
        
        topics = lda_model.print_topics()
        structured_topics = []
        for topic in topics:
            topic_id, topic_terms = topic
            structured_topic = {
                "Topic": topic_id,
                "Terms": [term.split("*")[1].strip('"') for term in topic_terms.split(" + ")]
            }
            structured_topics.append(structured_topic)
        
        return structured_topics

    @staticmethod
    def generate_wordcloud(text):
        if not text.strip():
            return ""
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
        buf = BytesIO()
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.savefig(buf, format="png")
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode("utf-8")
        return quote(image_base64)
    
    @staticmethod
    def dependency_parsing(text):
        if not text.strip():
            return []
        
        doc = nlp(text)
        dependencies = []
        for token in doc:
            dependencies.append((token.text, token.dep_, token.head.text))
        
        # Filter out unique dependencies
        unique_dependencies = list(dict((word, (word, dep, head)) for word, dep, head in dependencies).values())
        return unique_dependencies

    @staticmethod
    def translation(text, target_language="es"):
        if not text.strip():
            return "Input text is empty."
        translator = Translator()
        # Detect language synchronously
        detected_lang = translator.detect(text)
        if detected_lang.lang == target_language:
            return "Text is already in the target language."
        # Translate text synchronously
        translated_text = translator.translate(text, dest=target_language)
        return translated_text.text

    @staticmethod
    def text_summarization(text):
        if not text.strip():
            return "Input text is empty."
        try:
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            summarizer = LsaSummarizer()
            summary = summarizer(parser.document, 2)
            return ' '.join(str(sentence) for sentence in summary)
        except Exception as e:
            return f"Error in summarization: {str(e)}"
