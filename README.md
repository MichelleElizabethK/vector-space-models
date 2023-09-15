# Vector Space Model for Information Retrieval

This is an experimental retrieval system based on the vector space model. Experiments with various methods for text processing, query construction, term and document weighting, similarity measurement, etc. has been done to create an optimised system.

## Baseline Model:
The baseline model was developed based on the given specifications for both English and Czech datasets:
- token delimiters: any sequence of whitespace and punctuation marks
- term equivalence classes: no case normalisation or other equivalence classing
- removing stopwords: no
- query construction: all words from topic "title"
- term frequency weighting: natural
- document frequency weighting: none
- vector normalisation: cosine
- similarity measure: cosine
- pseudo-relevance feedback: none
- query expansion: none

Mean average precision for:
    English = 0.0450
    Czech = 0.0596

## Improved Model:

For English dataset:
- token delimiters: any sequence of whitespace and punctuation marks
- term equivalence classes: lowercasing and stemming (SnowballStemmer)
- removing stopwords: true
- query construction: ‘title’ of topics
- term frequency weighting: logarithmic
- document frequency weighting: inverse document freq
- vector normalisation: pivot
- similarity measure: cosine
- pseudo-relevance feedback: none
- query expansion: none

Mean average precision = 0.3761

For Czech dataset:
- token delimiters: any sequence of whitespace and punctuation marks
- term equivalence classes: lowercasing and lemmatization
- removing stopwords: true
- query construction: ‘title’ of topics
- term frequency weighting: logarithmic
- document frequency weighting: inverse document freq, prob-idf(for query)
- vector normalisation: pivot
- similarity measure: cosine
- pseudo-relevance feedback: none
- query expansion: none

Mean average precision = 0.3694

## How To Run the System

On the terminal, run the following command from the root directory:

`
python vector-space-models/run.py -q <topics.xml> -d <documents.lst> -r <run-test> -o <outputfile.res>
`
The configuration can be changed in the vector-space-models/config.py file.