# Deepfake Detection in Financial KYC

This project implements a fusion-based approach for deepfake detection in financial Know Your Customer (KYC) processes, combining deepfake detection and microexpression analysis.

## Components

- **Deepfake Detection**: Identifies manipulated facial images using MobileNetV2
- **Microexpression Analysis**: Detects emotional responses in facial video sequences
- **Fusion System**: Combines both signals for improved liveness detection

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Fusion Batch Prediction

Run the fusion batch prediction script to analyze a set of images and generate results:

```bash
python fusion/fusion_batch_predict.py
```

This will:
1. Process 100 real and 100 fake deepfake frames
2. Analyze matching microexpression sequences
3. Generate fusion predictions (LIVE/SPOOF)
4. Save results to:
   - CSV file: `fusion/batch_results/fusion_batch_results_<timestamp>.csv`
   - Word document: `fusion/batch_results/fusion_batch_results_<timestamp>.docx`

### Word Document Output

The generated Word document includes:
- Title and timestamp
- Description of the models used
- Summary statistics (total samples, LIVE/SPOOF counts)
- Accuracy analysis (if ground truth is available)
- Detailed results table with all predictions

## Requirements

- Python 3.8+
- TensorFlow 2.x
- OpenCV
- pandas
- python-docx (for Word document generation)
