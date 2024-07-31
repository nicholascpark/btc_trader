import argparse
from train import main as train_main
from predict import main as predict_main
from evaluate import main as evaluate_main
import warnings
warnings.filterwarnings("ignore")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Trading Bot")
    parser.add_argument('--mode', '-m', choices=['train', 'predict', 'evaluate'], 
                        help='Mode to run the bot in (train, predict, evaluate)', required=True)
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    if args.mode == 'train':
        train_main()
    elif args.mode == 'predict':
        predict_main()
    elif args.mode == 'evaluate':
        evaluate_main()

if __name__ == "__main__":
    main()