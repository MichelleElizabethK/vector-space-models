config = {
    "en": {
        "run-0": {
            "lowercase": False,
            "stemming": False,
            "lemmatize": False,
            "remove_stopwords": False,
            "doc-term-weighting": "",
            "document-weighting": "",
            "query-term-weighting": "",
            "query-weighting": "",
            "vector-normalization": "cosine",
            "similarity measure": "cosine",
            "pseudo-relevance feedback": "",
            "query expansion": ""
        },
        "run-1": {
            "lowercase": True,
            "stemming": True,
            "lemmatize": False,
            "remove_stopwords": True,
            "doc-term-weighting": "log",
            "document-weighting": "idf",
            "query-term-weighting": "log",
            "query-weighting": "idf",
            "vector-normalization": "pivot",
            "similarity measure": "cosine",
            "pseudo-relevance feedback": "",
            "query expansion": ""
        }
    },
    "cs": {
        "run-0": {
            "lowercase": False,
            "stemming": False,
            "lemmatize": False,
            "remove_stopwords": False,
            "doc-term-weighting": "",
            "document-weighting": "",
            "query-term-weighting": "",
            "query-weighting": "",
            "vector-normalization": "cosine",
            "similarity measure": "cosine",
            "pseudo-relevance feedback": "",
            "query expansion": ""
        },
        "run-1": {
            "lowercase": True,
            "stemming": False,
            "lemmatize": True,
            "remove_stopwords": True,
            "doc-term-weighting": "log",
            "document-weighting": "prob-idf",
            "query-term-weighting": "log",
            "query-weighting": "idf",
            "vector-normalization": "pivot",
            "similarity measure": "cosine",
            "pseudo-relevance feedback": "",
            "query expansion": ""
        }
    }
}
