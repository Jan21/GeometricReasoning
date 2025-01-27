from .model import NeuroSAT

models_with_args = {
    'NeuroSAT': {
        'model_class': NeuroSAT,
        'model_args': {
            'd': 96,
            'final_reducer': 'mean',
        }
    }
}
