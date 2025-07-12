Before Runing TCRGoogle.
We highly recommend you to create a new conda enviroment:
conda create -n llm-env python=3.10 ipykernel numpy scipy pandas matplotlib tqdm
install nessasary package:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -U transformers datasets accelerate peft einops transformers_stream_generator
And U can just run everthing on your CPU(High computational efficiency, no need for GPU involvement.)
python TCRGoogle_Run.py
