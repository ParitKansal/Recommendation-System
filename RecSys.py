"""
@author: Parit Kansal
"""
from MovieLens import MovieLens
from RBMAlgorithm import RBMAlgorithm
from ContentKNNAlgorithm import ContentKNNAlgorithm
from AutoRecAlgorithm import AutoRecAlgorithm
from surprise import SVD, SVDpp
from surprise import NormalPredictor
from HybridAlgorithm import HybridAlgorithm
from Evaluator import Evaluator


import random
import numpy as np

def LoadMovieLensData():
    ml = MovieLens()
    print("Loading movie ratings...")
    data = ml.loadMovieLensLatestSmall()
    print("\nComputing movie popularity ranks so we can measure novelty later...")
    rankings = ml.getPopularityRanks()
    return (ml, data, rankings)


# Load up common data set for the recommender algorithms
(ml, evaluationData, rankings) = LoadMovieLensData()

# Construct an Evaluator to, you know, evaluate them
evaluator = Evaluator(evaluationData, rankings)

# SVD
SVD = SVD(random_state=10)
# SVD++
SVDPlusPlus = SVDpp(random_state=10)
#contentKNN
contentKNN = ContentKNNAlgorithm()
#RBM
RBM = RBMAlgorithm(epochs=20)
#Autoencoder
AutoRec = AutoRecAlgorithm()
#Combine them
Hybrid = HybridAlgorithm([SVD, SVDPlusPlus, contentKNN, RBM, AutoRec], [0.2, 0.2, 0.2, 0.2, .2])
# Just make random recommendations
Random = NormalPredictor()

# SVD
evaluator.AddAlgorithm(SVD, "SVD")
# SVD++
evaluator.AddAlgorithm(SVDPlusPlus, "SVD++")
#contentKNN
evaluator.AddAlgorithm(contentKNN, "ContentKNN")
#RBM
evaluator.AddAlgorithm(RBM, "RBM")
#Autoencoder
evaluator.AddAlgorithm(AutoRec, "AutoRec")
#Combine them
evaluator.AddAlgorithm(Hybrid, "Hybrid")
# Just make random recommendations
evaluator.AddAlgorithm(Random, "Random")


# Fight!
evaluator.Evaluate(True)