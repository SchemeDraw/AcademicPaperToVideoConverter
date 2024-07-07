#!/bin/bash
conda create -n papertovideo python=3.11 -y
conda activate papertovideo
pip install llama-index-embeddings-azure-openai
pip install llama-index-llms-azure-openai
pip install llama-index
pip install pydub
pip install opencv-python
pip install git+https://github.com/titipata/scipdf_parser
python -m spacy download en_core_web_sm
pip install google-cloud-texttospeech
pip install ffmpeg-python
pip install pdf2image
