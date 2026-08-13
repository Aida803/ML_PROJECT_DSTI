import re
import numpy as np
import pandas as pd

LANG_GROUPS = ['asian', 'classical', 'english', 'other', 'western_european']

#Defining a function for all title features
def title_features(title):
    return {
        'title_chars':      len(title),
        'has_subtitle':     int(":" in title),
        'title_has_number': int(bool(re.search(r'\d', title))),
        'is_collection':    int(bool(re.search(
                              r'boxed set|box set|omnibus|collection|anthology',
                              title, re.I))),
    }

#Defining a function for all author features
def author_features(authors, num_pages):
    num_authors = authors.count("/") + 1
    return{
        'num_authors': num_authors,
        'multiple_authors' : int(num_authors > 1),
        'pages_per_author' : num_pages / num_authors
    }

#Defining a function for Audiobook
def audiobook_feature(title, publisher):
    pub_kw   = r'audio|tantor|listening library|recorded books|books on tape|highbridge|your coach'
    title_kw = r'audio|unabridged|abridged|\bcd\b|audiobook|spoken|cassette'
    return {
        'is_audiobook' : int(bool(re.search(title_kw, title, re.I)) or
                   bool(re.search(pub_kw, publisher, re.I)))
    }

#Defining a function for Arithemtic features
def rating_features(ratings_count, text_reviews_count):
    log_ratings = float(np.log1p(ratings_count))
    rev_per_rating = text_reviews_count / ratings_count if ratings_count > 0 else 0
    return {
    'log_ratings':  log_ratings,
    'rev_per_rating': rev_per_rating
        }

#Defining a function for the frequency and ordinal features
def lookup_features(authors, publisher, bundle):
    a_ord, p_ord = bundle['ordinal_encoder'].transform(
        pd.DataFrame([[authors, publisher]], columns=['authors', 'publisher']))[0]
    return {
        'author_freq':    bundle['author_freq_map'].get(authors, 1),
        'publisher_freq': bundle['publisher_freq_map'].get(publisher, 1),
        'author_ord':     float(a_ord),
        'publisher_ord':  float(p_ord),
    }

#Defining a function for the language features
def language_features(language_code, bundle):
    lang = bundle['language_groups'].get(language_code, 'other')
    return {f'language_group_{g}': int(lang == g) for g in LANG_GROUPS}

#Combine all feature functions into the final model input
def build_features(title, authors, publisher, num_pages, language_code,
                   publication_year, ratings_count, text_reviews_count, bundle):
    feats = {'num_pages': num_pages, 'publication_year': publication_year}
    feats |= title_features(title)
    feats |= author_features(authors, num_pages)
    feats |= audiobook_feature(title, publisher)
    feats |= rating_features(ratings_count, text_reviews_count)
    feats |= lookup_features(authors, publisher, bundle)
    feats |= language_features(language_code, bundle)
    return pd.DataFrame([feats])[bundle['feature_columns']]