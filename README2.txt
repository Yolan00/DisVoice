
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
conda create -n disvoice-py313 python=3.13.9 -y
conda activate disvoice-py313

# 3) Install this repository and its dependencies
pip install -e .

# 4) Run the basic features extraction script
python extract_all_features.py /path/to/audios -o ./features_out

# To extract only specific families, use --include with any of:
#   glottal phonation articulation prosody phonological replearning
python extract_all_features.py /path/to/audios --include glottal prosody -o ./features_out

# (Optional) Use RAE instead of CAE for RepLearning
python extract_all_features.py /path/to/audios --include replearning --model RAE -o ./features_out

