from run_simulation import SimulationRunner
from src.config.simulation_config import load_simulation_config
import sys

if __name__ == "__main__":
    # Choose configuration file based on command line argument
    if len(sys.argv) > 1:
        config_name = sys.argv[1].lower()
    else:
        config_name = "sparsenet"  # duncs , sparsenet
    
    # Switch case for configuration selection
    match config_name:
        case "duncs":
            config_path = "src/config/DUNCS.yaml"
        case "subspacenet":
            config_path = "src/config/subspaceNet.yaml"
        case "sparsenet":
            config_path = "src/config/sparseNet.yaml"
        case "dumfocuss":
            config_path = "src/config/duMfocuss.yaml"
        case "doaformer":
            config_path = "src/config/doaFormer.yaml"
        case _:
            print(f"Unknown configuration: {config_name}")
            print("Available options: 'duncs', 'subspacenet', 'sparsenet', 'dumfocuss', 'doaformer'")
            sys.exit(1)
    
    print(f"Loading configuration: {config_path}")
    config = load_simulation_config(config_path)
    SimulationRunner(config).run()
