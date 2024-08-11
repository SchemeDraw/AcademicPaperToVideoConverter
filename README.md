# PaperToVideo - Convert Academic Papers to Engaging Videos

Welcome to PaperToVideo! This repository converts academic papers into engaging video summaries, typically 10-20 minutes in length. It's an excellent tool for researchers, students, and educators looking to quickly understand the core concepts of a paper.

Produce Demo: [demo](https://youtu.be/va4cEo35VqQ?si=pR-t7PKVtcuney3B)
## Overview

PaperToVideo uses advanced AI models to generate concise video explanations of academic papers. Note that the model may occasionally hallucinate and might not be suitable for all types of papers.

**Bonus:** Check out some great paper videos on YouTube: [Trend in Research](https://www.youtube.com/@trendinresearch)

## How to Use

You have two options to generate videos from academic papers:

### Option 1: Use Our Free Service

Visit [TrendGPT](https://www.trendgpt.site/) to sign up and upload your paper for conversion. This process usually takes about 10-15 minutes. We recommend trimming the paper to remove sections after the references before uploading.

### Option 2: Run Locally Using Our Code

If you prefer to run the process on your own machine, follow these steps. Note that you will need access to Microsoft Azure GPT-4 model and Google Cloud TTS Service (fees may apply).

### Step 0: Preparation

1. Run `install.sh` to install the required packages. A virtual environment with Python 3.11 is recommended.
2. Download the paper you want to convert and save it in the `papers` folder. Let's call the paper `placeholder.pdf`.
3. Obtain your Microsoft Azure OpenAI model credentials, including the model version, API keys, and endpoint. We use GPT-4-turbo, but other versions may work (though not guaranteed with weaker models like GPT-3.5).
4. Get your Google Cloud credentials, save them in a JSON file, and update the path in `create_video_from_paper.py` as `google_credential_path = YOUR_GOOGLE_CREDENTIAL_PATH`. Google Cloud is used for text-to-speech generation. Feel free to modify to use other free TTS service.

### Step 1: Run the Serve Program

Run `serve.sh` and keep it running for about 2 minutes to avoid bugs in the next step. Then keep it running in the backend.

### Step 2: Generate the Video

Run `python create_video_from_paper.py`. The paper slides PDF file and generated MP4 file will be saved in the `resultant_latex/placeholder` folder, where `placeholder` is the original paper PDF name.

## Contributing

We welcome contributions! 
