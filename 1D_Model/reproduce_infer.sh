#!/bin/bash
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"

# from Vincent
python infer.py --model_config V2
python infer.py --model_config resnet34
python infer.py --model_config V2SD
python infer.py --model_config M3D

# from Maxim
python infer.py --model_config M-1D
python infer.py --model_config M-1DC16
python infer.py --model_config M-SD16
python infer.py --model_config M-SD32

# from Richard
python infer.py --model_config R-35
python infer.py --model_config R-112
python infer.py --model_config R-120
python infer.py --model_config R-121
python infer.py --model_config R-122
python infer.py --model_config R-124
python infer.py --model_config R-133