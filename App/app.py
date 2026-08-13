from pathlib import Path
import joblib
import streamlit as st
from features import build_features

#Changing path for streamlit
MODEL_PATH = Path(__file__).resolve().parent.parent / "Model" / "book_rating_random_forest_model.joblib"

@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)

#Precaution in case the model isn't loaded correctly
if not MODEL_PATH.exists():
    st.error("Model file not found. Expected it at: Model/book_rating_random_forest_model.joblib")
    st.stop()
bundle = load_bundle()

st.title("Book Rating Prediction")

#Placeholder for our input boxes with help bubbles when applicable
with st.form("book_form"):
    title      = st.text_input("Title", placeholder="e.g. The Hobbit or There and Back Again")
    authors = st.text_input("Authors", placeholder="e.g. J.R.R. Tolkien", help="Separate multiple authors with / — this affects the prediction")
    publisher  = st.text_input("Publisher", placeholder="e.g. Houghton Mifflin")
    num_pages  = st.number_input("Number of pages", min_value=0, value=None, placeholder="e.g. 366")
    language_code = st.selectbox("Language", sorted(bundle['language_groups'].keys()))
    publication_year   = st.number_input("Publication year", min_value=1900, max_value=2026, value=None, placeholder="e.g. 2002")
    ratings_count      = st.number_input("Ratings count", min_value=0, value=None, placeholder="e.g. 2530894")
    text_reviews_count = st.number_input("Text reviews count", min_value=0, value=None, placeholder="e.g. 32871")

    submitted = st.form_submit_button("Predict rating")


if submitted:
#Stripping leading spaces
    title     = title.strip()
    authors   = authors.strip()
    publisher = publisher.strip()
#All values must be filled to predict a book!
    missing = []
    if not title:                  missing.append("Title")
    if not authors:                missing.append("Authors")
    if not publisher:              missing.append("Publisher")
#Checks if different authors are separated by values other than /
    if authors and "/" not in authors and any(s in authors for s in [",", "&", " and "]):
        st.info("Multiple authors must be separated with `/` — e.g. `J.R.R. Tolkien/Christopher Tolkien`")
    if num_pages is None:                  missing.append("Number of pages")
    if publication_year is None:           missing.append("Publication year")
    if ratings_count is None:              missing.append("Ratings count")
    if text_reviews_count is None:         missing.append("Text reviews count")
#Checks if there are more written reviews than ratings, which isn't allowed on most rating websites
    if (ratings_count is not None and text_reviews_count is not None
            and text_reviews_count > ratings_count):
        missing.append("Text reviews count (cannot exceed ratings count)")

    if missing:
        st.warning("Please check: " + ", ".join(missing))

#Nothing is missing, from features import build_features function  and predict result
    else:
        row = build_features(
            title=title,
            authors=authors,
            publisher=publisher,
            num_pages=num_pages,
            language_code=language_code,
            publication_year=publication_year,
            ratings_count=ratings_count,
            text_reviews_count=text_reviews_count,
            bundle=bundle,
        )
        pred = bundle['model'].predict(row)[0]
        st.metric("Predicted rating", f"{pred:.2f} / 5")
#Small robustness check for authors or publisher not present in dataset or year is too recent
        notes = []
        if authors not in bundle['author_freq_map']:
            notes.append(f"Author “{authors}” is not in the training data — "
                         "author features fall back to defaults.")
        if publisher not in bundle['publisher_freq_map']:
            notes.append(f"Publisher “{publisher}” is not in the training data — "
                         "publisher features fall back to defaults.")
        if publication_year > 2020:
            notes.append("The training data ends in 2020. Predictions for later "
                         "publications are less reliable.")

        for n in notes:
            st.caption(n)