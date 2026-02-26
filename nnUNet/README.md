# nnUNet with Topological Correction

This folder contains the code for integrating our methodology — persistent homology-based topological correction (TC) — into the nnUNet framework.

## Prerequisites

1. Install the modified nnUNet with topological correction from our fork:
```bash
   pip install git+https://github.com/lino202/nnUNet
```
2. Set up the required nnUNet environment variables (`nnUNet_raw`, `nnUNet_preprocessed`, 
   `nnUNet_results`) as described in the [original nnUNet documentation](https://github.com/MIC-DKFZ/nnUNet).
3. Prepare your dataset in nnUNet format and install all dependencies.

## Usage

The scripts can/should be executed following the calling order defined in [launch.json](../.vscode/launch.json). 
Look for the `# nnUNet REVIEW` marker as the entry point for the relevant pipeline steps.


## Acknowledgements

This implementation builds on the original nnUNet framework:
Original repository: [MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet).