import numpy as np, pandas as pd, string, scipy
from matplotlib.cbook import boxplot_stats
from sklearn.feature_extraction.text import TfidfVectorizer


def calculate_distance(lat1: np.ndarray, long1: np.ndarray, lat2: np.ndarray = np.array([40.689247]), long2: np.ndarray = np.array([-74.044502])) -> np.ndarray:
    """
    Calculate distance from the apartments to the other location
    
    ## Parameters

    lat1 : np.ndarray
        Array of the apartments' latitudes
    long1 : np.ndarray
        Array of the apartments' longitudes
    lat2 : np.ndarray
        Array of the other location latitude of length 1, default np.array([40.689247])
    long2 : np.ndarray
        Array of the other location longitude of length 1, default np.array([-74.044502])

    ## Returns

    np.ndarray
        Array of distances from the apartments to the other location in kilometers
    """
    R = 6373.0 # ~radius of Earth, km

    lat2 = np.repeat(lat2, len(lat1))
    long2 = np.repeat(long2, len(long1))

    # convert to radians to calculate distance in km
    lat1 = np.radians(lat1)
    long1 = np.radians(long1)
    lat2 = np.radians(lat2)
    long2 = np.radians(long2)
    
    dlat = lat2 - lat1
    dlong = long2 - long1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlong / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def get_outliers(variable: pd.Series) -> list:
    """
    Extract a list of outliers from the boxplot and prints their share in a given variable

    ## Parameters
    variable : pd.Series
        A column from which the boxplot was created
    """
    outliers = [y for stat in boxplot_stats(variable) for y in stat['fliers']]
    print(f"Share of outliers: {100 * (len(outliers) / variable.shape[0]):.2f}%")
    return outliers


def preprocess_names(names: pd.Series) -> scipy.sparse._csr.csr_matrix:
    names = names.apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)).strip())
    tfidf = TfidfVectorizer(stop_words={'english'}, ngram_range=[2,2], max_features=1000)
    names = tfidf.fit_transform(names)
    return names