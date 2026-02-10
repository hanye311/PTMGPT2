The original repo is: https://github.com/pallucs/PTMGPT2

GPT-prepare_data.py is for extracting 21nt peptides(PTM site is in the center) from a sequence and saving them.

GPT-inference.py is used for predicting if the center position of 21nt peptide is PTM site. So input should be 21nt-peptide. Output will be 0 or 1. GPU version.

GPT-inference-cpu.py is CPU version for predicting.

GPT-train.py for retraining new models.

Please download the checkpoints from the original repo.
