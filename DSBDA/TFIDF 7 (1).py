import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def setup_nltk():
    """Ensure all required NLTK resources are available"""
    resources = {
        'punkt': 'tokenizers/punkt',
        'stopwords': 'corpora/stopwords',
        'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger',
        'wordnet': 'corpora/wordnet',
        'omw-1.4': 'corpora/omw-1.4'
    }
    
    for resource, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"Downloading {resource}...")
            nltk.download(resource)

def preprocess_text(document):
    """Perform all text preprocessing steps"""
    # Tokenization
    tokens = word_tokenize(document.lower())
    
    # POS Tagging and filtering
    pos_tags = pos_tag(tokens)
    filtered_tokens = [
        word for word, tag in pos_tags 
        if word.isalpha() and word not in stopwords.words('english')
    ]
    
    # Stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    
    # Lemmatization with POS
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = []
    for word, tag in pos_tag(filtered_tokens):
        pos = get_wordnet_pos(tag)
        lemmatized_tokens.append(lemmatizer.lemmatize(word, pos=pos))
    
    return lemmatized_tokens

def get_wordnet_pos(treebank_tag):
    """Convert treebank POS tags to WordNet POS tags"""
    tag = treebank_tag[0].upper()
    return {'J': 'a', 'N': 'n', 'V': 'v', 'R': 'r'}.get(tag, 'n')

def main():
    # Initialize NLTK resources
    setup_nltk()
    
    # Sample document
    document = """The quick brown fox jumps over the lazy dog. Dogs are great pets, but foxes are wild animals. Foxes and dogs have different behaviors."""
    
    # Preprocess the document
    processed_tokens = preprocess_text(document)
    print("Final Processed Tokens:")
    print(processed_tokens)
    
    # TF-IDF Representation
    corpus = [
        " ".join(processed_tokens),
        "dog make wonderful companion",
        "fox live forest"
    ]
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    df_tfidf = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=vectorizer.get_feature_names_out(),
        index=["Document 1", "Document 2", "Document 3"]
    )
    
    print("\nTF-IDF Representation:")
    print(df_tfidf)

if __name__ == "__main__":
    main()