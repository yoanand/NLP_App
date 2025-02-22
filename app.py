import streamlit as st
from utils import NLPUtils
import asyncio

dependency_explanations = {
    "nsubjpass": "Nominal subject in a passive construction",
    "auxpass": "Auxiliary passive, a helping verb in passive voice constructions",
    "prep": "Preposition, relating a noun to another word",
    "pobj": "Prepositional object, the object of a preposition",
    "compound": "Part of a compound noun",
    "det": "Determiner, a modifying word that determines the kind of reference a noun or noun group has",
    "amod": "Adjectival modifier, a word that modifies a noun and is an adjective",
    "conj": "Conjunct, the relation between coordinated elements",
    "cc": "Coordinating conjunction, used to link words or phrases",
    "appos": "Appositional modifier, a noun or noun phrase that renames another noun right beside it",
    "aux": "Auxiliary, a helping verb",
    "ROOT": "Root of the sentence",
    "acl": "Clausal modifier of noun",
    "nsubj": "Nominal subject",
    "advmod": "Adverbial modifier",
    "advcl": "Adverbial clause modifier",
    "dobj": "Direct object",
    "pcomp": "Complement of preposition",
    "mark": "Marker, a word introducing a clause",
    "relcl": "Relative clause modifier",
    "punct": "Punctuation",
    "dep": "Unspecified dependency",
    "preconj": "Pre-correlative conjunction",
    "csubj": "Clausal subject",
    "xcomp": "Open clausal complement",
    "expl": "Expletive",
    "npadvmod": "Noun phrase as adverbial modifier",
}

# Task descriptions
task_descriptions = {
    "Sentiment Analysis": "Analyze the sentiment of the input text and classify it as positive, negative, or neutral.",
    "Named Entity Recognition": "Identify and classify named entities (e.g., people, organizations, locations) in the text.",
    "Text Summarization": "Generate a concise summary of the input text.",
    "Word Frequency Analysis": "Calculate the frequency of each word in the input text.",
    "Spelling Correction": "Correct any spelling errors in the input text.",
    "Keyword Extraction": "Extract the most relevant keywords from the input text.",
    "Text Similarity": "Compare the similarity between two pieces of text.",
    "Topic Modeling": "Identify the main topics present in the input text.",
    "Word Cloud": "Generate a word cloud to visualize the most frequent words in the text.",
    "Dependency Parsing": "Analyze the grammatical structure of the sentence and show the relationships between words.",
    "Translation": "Translate the input text into another language."
}

# Streamlit App
st.set_page_config(page_title="Advanced NLP App", page_icon="🌟")

st.title("Advanced Pre-Transformer NLP App")

st.sidebar.title("Select an NLP Task")
task = st.sidebar.selectbox(
    "Choose a Task",
    list(task_descriptions.keys())
)

# Display task description below the sidebar
st.sidebar.write(f"**{task}:** {task_descriptions[task]}")

# Input text
st.markdown(
    """
    <style>
    .big-font {
        font-size:18px !important;
        color: #2c3e50;
    }
    </style>
    """, unsafe_allow_html=True)
text_input = st.text_area("Enter your text:", placeholder="Type some text here...", height=200, label_visibility="collapsed")

if text_input:
    if task == "Sentiment Analysis":
        st.header("Sentiment Analysis")
        sentiment = NLPUtils.sentiment_analysis(text_input)
        st.write(f"**Sentiment:** {sentiment}")

    elif task == "Named Entity Recognition":
        st.header("Named Entity Recognition")
        entities = NLPUtils.named_entity_recognition(text_input)
        st.write("**Named Entities:**")
        for entity, entity_type in entities:
            st.write(f"- **{entity}** ({entity_type})")

    elif task == "Text Summarization":
        st.header("Text Summarization")
        summary = NLPUtils.text_summarization(text_input)
        st.write(f"**Summary:** {summary}")

    elif task == "Word Frequency Analysis":
        st.header("Word Frequency Analysis")
        word_freq = NLPUtils.word_frequency_analysis(text_input)
        st.write("**Word Frequencies:**")
        for word, freq in word_freq.items():
            st.write(f"- **{word}:** {freq}")

    elif task == "Spelling Correction":
        st.header("Spelling Correction")
        corrected_text = NLPUtils.spelling_correction(text_input)
        st.write(f"**Corrected Text:** {corrected_text}")

    elif task == "Keyword Extraction":
        st.header("Keyword Extraction")
        keywords = NLPUtils.keyword_extraction(text_input)
        # Extract only the keyword strings (first element of the tuple)
        keyword_strings = [kw[0] for kw in keywords]
        st.write("**Keywords:**")
        st.write(", ".join(keyword_strings))

    elif task == "Text Similarity":
        st.header("Text Similarity")
        text2 = st.text_area("Enter another text for comparison:", height=200, label_visibility="collapsed")
        if text2:
            similarity = NLPUtils.text_similarity(text_input, text2)
            st.write(f"**Similarity Score:** {similarity:.2f}")

    elif task == "Topic Modeling":
        st.header("Topic Modeling")
        topics = NLPUtils.topic_modeling(text_input)
        st.write("**Topics:**")
        for topic in topics:
            st.write(f"**Topic {topic['Topic']}:**")
            for term in topic['Terms']:
                st.write(f"- {term}")

    elif task == "Word Cloud":
        st.header("Word Cloud")
        wordcloud_base64 = NLPUtils.generate_wordcloud(text_input)
        st.image(f"data:image/png;base64,{wordcloud_base64}")

    elif task == "Dependency Parsing":
        st.header("Dependency Parsing")
        dependencies = NLPUtils.dependency_parsing(text_input)
        # Display the structured dependency parse tree with explanations
        st.write("**Dependency Parse Tree:**")
        st.write("Word\t(POS)\t→\tHead\t | Explanation")
        for word, dep, head in dependencies:
            explanation = dependency_explanations.get(dep, "No explanation available")
            st.write(f"{word}\t({dep})\t→\t{head}\t | {explanation}")

    elif task == "Translation":
        st.header("Translation")
        target_lang = st.text_input("Enter target language code (e.g., 'es' for Spanish):", "es")
        translation = asyncio.run(NLPUtils.translation(text_input, target_lang))
        st.write(f"**Translated Text:** {translation}")
