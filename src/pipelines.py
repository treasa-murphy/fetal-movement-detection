from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from sktime.transformations.panel.rocket import MiniRocket, MiniRocketMultivariate
from aeon.classification.convolution_based import HydraClassifier
from aeon.transformations.collection.convolution_based import QUANTTransformer
from aeon.classification.convolution_based import QUANTClassifier
from sktime.classification.kernel_based import RocketClassifier
from sktime.classification.kernel_based import MiniRocketClassifier

# --- piplines based on QUANTClassifier directly
def get_quant_classifiers():
    return [
        QUANTClassifier(),
        make_pipeline(StandardScaler(), QUANTClassifier())
    ]

# --- pipelines using QUANTTransformer + Extra Trees
def get_quant_extratrees_pipelines():
    return [
        make_pipeline(QUANTTransformer(), ExtraTreesClassifier()),
        make_pipeline(QUANTTransformer(), StandardScaler(), ExtraTreesClassifier()),
        make_pipeline(StandardScaler(), QUANTTransformer(), ExtraTreesClassifier())
    ]

# --- pipelines using QUANTTransformer + Ridge
def get_quant_ridge_pipelines():
    return [
        make_pipeline(QUANTTransformer(), RidgeClassifierCV()),
        make_pipeline(StandardScaler(), QUANTTransformer(), RidgeClassifierCV()),
        make_pipeline(QUANTTransformer(), StandardScaler(), RidgeClassifierCV())
    ]

# --- pipelines using QUANTTransformer + LDA
def get_quant_lda_pipelines():
    return [
        make_pipeline(QUANTTransformer(), LinearDiscriminantAnalysis()),
        make_pipeline(StandardScaler(), QUANTTransformer(), LinearDiscriminantAnalysis()),
        make_pipeline(QUANTTransformer(), StandardScaler(), LinearDiscriminantAnalysis())
    ]

# --- pipelines using MiniRocket + Ridge
def get_minirocket_ridge_pipelines():
    return [
        make_pipeline(MiniRocket(), StandardScaler(), RidgeClassifierCV())
    ]

# --- pipelines using MiniRocketClassifier directly
def get_minirocket_classifiers():
    return [
        make_pipeline(StandardScaler(), MiniRocketClassifier()),
        make_pipeline(QUANTTransformer(), MiniRocketClassifier()),
        make_pipeline(QUANTTransformer(), StandardScaler(), MiniRocketClassifier()),
        make_pipeline(StandardScaler(), QUANTTransformer(), MiniRocketClassifier())
    ]

# --- pipelines using HydraClassifier
def get_hydra_pipelines():
    return [
        make_pipeline(HydraClassifier()),
        make_pipeline(StandardScaler(), HydraClassifier()),
        make_pipeline(QUANTTransformer(), HydraClassifier()),
        make_pipeline(QUANTTransformer(), StandardScaler(), HydraClassifier()),
        make_pipeline(StandardScaler(), QUANTTransformer(), HydraClassifier())
    ]

# --- export all pipelines 
def get_all_pipelines():
    return (
        get_quant_classifiers() +
        get_quant_extratrees_pipelines() +
        get_quant_ridge_pipelines() +
        get_quant_lda_pipelines() +
        get_minirocket_ridge_pipelines() +
        get_minirocket_classifiers() +
        get_hydra_pipelines()
    )

