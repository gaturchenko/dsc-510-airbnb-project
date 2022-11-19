import pickle, numpy as np
from math import sin, cos, sqrt, atan2, radians
from sklearn.preprocessing import StandardScaler


class PredictionGenerator:
    def __init__(self, lat, long, neigh, room, night, avail, exp, reviews, lists) -> None:
        with open('app/src/rf.pkl', 'rb') as f:
            self.model = pickle.load(f)
        self.lat, self.long, self.neigh, self.room, self.night, self.avail, self.exp, self.reviews, self.lists = lat, long, neigh, room, night, avail, exp, reviews, lists
    
    def calculate_distance(self, lat1, long1, lat2 = 40.689247, long2 = -74.044502):
        R = 6373.0
        lat1 = radians(lat1)
        long1 = radians(long1)
        lat2 = radians(lat2)
        long2 = radians(long2)

        dlon = long2 - long1
        dlat = lat2 - lat1

        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return R * c

    def prepare_input(self):
        dist_to_stl = self.calculate_distance(self.lat, self.long)
        reviews_per_month = self.reviews / self.exp

        if self.room == 'Private Room':
            room = 1
        else:
            room = 0
        
        if self.neigh == 'Brooklyn':
            brooklyn = 1
            manhattan = 0
            other = 0
        elif self.neigh == 'Manhattan':
            brooklyn = 0
            manhattan = 1
            other = 0
        else:
            brooklyn = 0
            manhattan = 0
            other = 1

        num_features = np.array([dist_to_stl, self.night, reviews_per_month, self.lists, self.avail])
        cat_features = np.array([room, brooklyn, manhattan, other])
        
        scaler = StandardScaler()
        num_features = scaler.fit_transform(num_features.reshape(-1, 1)).reshape((1, -1)).ravel()

        self.X = np.concatenate((num_features, cat_features)).reshape(1, -1)

    def get_prediction(self):
        return np.exp(self.model.predict(self.X))