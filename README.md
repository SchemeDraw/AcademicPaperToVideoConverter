# PaperToVideo

This repo converts academic papers into videos that explain the paper contents briefly. Usually the videos are of length 10-20 minutes.

Note that this model can hallucinate and may not be the best fit for all papers. 


You have two ways to run paper to video generation. 

One, you can use our #free service: [TrendGPT](https://www.trendgpt.site/), and sign up to upload paper and convert to video (this would typically takes about 10 - 15 minutes, and we suggest you trim the paper first before uploading, you can in general remove every thing after the reference section of a paper). 

Second, you can do it on your own using our code base. Though you have to have access to Microsoft Azure GTP-4 model and GCP TTS Service (you need to pay the fees)

## Step 0: preparation

First of all, run `install.sh` to install required packages. A virtual environment with python==3.11 is recommended. 

To run paper to video generation, you need to download the paper you want to convert and save it under `papers` folder, with name you assign. Let's call the paper name `placeholder.pdf`,
then you need to get your microsoft Azure Open AI model served, and get the model version, api keys, end point, for both the model and the embedding model. Here we used gpt-4-turbo. But you can feel free to change the model version (without guarantee that the model still works when it's changed to weaker models like GPT-3.5).

You would also need to get your google cloud credentials and save it in a json file, and put the path in the file `create_video_from_paper.py` as `google_credential_path = YOUR_GOOGLE_CREDENTIAL_PATH`. We use google cloud service mainly for doing text-to-speech generation.

## Step 1: run a separate serve program

Run `serve.sh` and keep it running for about 2 minutes (this is crucial to ensure no bugs in the next step)

## Step 2: run video generation

run `python create_video_from_paper.py` and the paper slides pdf file and generated mp4 file will be in the folder `resultant_latex/placeholder`. Where `placeholder` is the original paper pdf name. 



