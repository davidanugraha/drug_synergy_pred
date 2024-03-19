import argparse
import json

from .utils import *
from .preprocess import get_data_loaders
from .models.gat import GATNet

def main(model_config_path):
    # Read json for training
    model_config = json.load(open(model_config_path))
    logging.info(f"Running model with config of:\n {model_config}")
    
    val_split = model_config['val_split']
    batch_size = model_config['batch_size']
    
    # Get Dataloaders
    train_loader, val_loader, test_loader = get_data_loaders(val_split=val_split, batch_size=batch_size)
    
    # Read and initialize intended model
    model_name = model_config['model_name']
    if model_name == 'gat':
        model = GATNet(**model_config)
    else:
        raise NotImplementedError("Model not recognized")
    
    # Run train
    model.run_train(train_loader, val_loader)
    
    # Run prediction
    model_prediction = model.perform_prediction(test_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-mcp', '--model_config_path', type=str, required=True, help="Model config path for training")
    args = parser.parse_args()
    
    main(args.model_config_path)
