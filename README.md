# automatic_speech_eval

Automatic speech evaluation toolkit for L2 pronunciation assessment. This project implements neural network-based models for evaluating non-native speech across multiple dimensions including accentedness, fluency, and comprehensibility.

## Overview

This toolkit provides:
- Audio feature extraction (MFCC, mel spectrogram) for speech evaluation
- RNN and multi-input neural network models for automatic proficiency judgment
- Praat scripts for TextGrid manipulation and acoustic analysis
- Data preparation utilities for speech evaluation experiments

## Related Publications

- Park, S. & Culnan, J. (2019). "A comparison between native and non-native speech for automatic speech recognition." *JASA* 145(3_Supplement).
- Park, S. & Culnan, J. (2019). "Automatic perceptual judgment using neural networks." *JASA* 146(4_Supplement).
- Park, S. (2021). "Human and Machine Judgment of Non-Native Speakers' Speech Proficiency." PhD Thesis, The University of Arizona.

## Project Structure

```
├── bin/                    # Main scripts
│   ├── RNN_automatic_judgment.ipynb  # Main notebook
│   ├── run_models_cv.py    # Cross-validation model training
│   ├── run_multi_model.py  # Multi-input model training
│   ├── calculateRhythm.py  # Speech rhythm calculation
│   ├── utils/              # Model architectures and utilities
│   ├── *.praat             # Praat scripts for acoustic analysis
│   └── *.py                # Data processing scripts
├── data/                   # Sample data
│   └── TextGrid/           # Praat TextGrid files
├── results/                # Evaluation results
├── docs/                   # Documentation
│   └── kaldi-instructional/  # Kaldi ASR instructional notebooks
└── README.md
```

## Requirements

- Python 3.6+
- TensorFlow / Keras
- Praat (for acoustic scripts)
- NumPy, pandas, scikit-learn

## Usage

See `bin/RNN_automatic_judgment.ipynb` for the main workflow.

## Author

Seongjin Park — [seongjinpark.com](https://seongjinpark.com)

## License

MIT
