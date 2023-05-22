#!/bin/bash

python main.py --multirun -cn config '+instance=range(10)' 'architecture.edge_mode=lstm2,handcrafted' 'log.project=HGRL_Main' 'env.train=-S1,-S2,-S3,-S4,-S5' 'env.eval=opposite'
