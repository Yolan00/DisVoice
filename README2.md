
 _____  _  __      __   _          
|  __ \(_) \ \    / /  (_)         
| |  | |_ __\ \  / /__  _  ___ ___ 
| |  | | / __\ \/ / _ \| |/ __/ _ \
| |__| | \__ \\  / (_) | | (_|  __/
|_____/|_|___/ \/ \___/|_|\___\___|
                                    
                                    

This fork is tested on Python 3.13.9 (Conda). It fixes path issues in batch mode and adds a script to extract all six DisVoice feature families, writing results to CSV and JSON.


Quick Start (Conda)
# 1) Install Praat (Linux)
apt-get install praat

# 2) Create and activate the environment (Python 3.13.9)
conda create -n <env name> python=3.13.9 -y
conda activate <env name>

# 3) Install this repository and its dependencies
pip install -e .


Run the basic features extraction script:
python extract_all_features.py /path/to/audios -o ./features_out

use ¨--include <feature name>¨ right after the path to audios to extract only the features of the selected family
