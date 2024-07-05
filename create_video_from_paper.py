from llama_index.llms.azure_openai import AzureOpenAI
import re
from llama_index.core import Settings
from llama_index.core import Document, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
import json
import pickle as pkl
import os
import numpy as np
from tqdm import tqdm
import multiprocessing
from functools import partial
import cv2
import time
from pydub import AudioSegment
import io
from create_video_from_pdf_and_audio import generate_video_from_pdf
from step_1_extract_info import extract_formula_image, get_paper_info
import scipdf
from google.cloud import texttospeech  
from google.oauth2 import service_account
import os
import google.cloud.texttospeech as tts
from os.path import commonpath, relpath


google_credential_path = YOUR_GOOGLE_CREDENTIAL_PATH  # should be a json file containing gcp credentials

def get_path_difference(path1, path2):
    """
    Get the difference between two paths, assuming one is the other's parent path.

    Args:
        path1 (str): The first path.
        path2 (str): The second path.

    Returns:
        str: The difference between the two paths, or an empty string if they are the same.
    """
    common_path = commonpath([path1, path2])

    if common_path == path1:
        return relpath(path2, path1)
    elif common_path == path2:
        return relpath(path1, path2)
    else:
        raise ValueError("The paths are not related as parent and child.")

    return ""

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(google_credential_path)

def split_text(text, max_length=300):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_length:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


class Agent:
    def __init__(self, api_key, end_point, api_version,
        max_retries=6, timeout=600, max_tokens=28000,
        embed_endpoint=None, embed_api_version=None,
        embed_key=None,
        chunk_size=2000,
        overlap=300):
        self.llm = AzureOpenAI(
            model="gpt-4",
            deployment_name="gpt-4-turbo",
            api_key=api_key,
            azure_endpoint=end_point,
            api_version=api_version,
            max_retries=max_retries,
            timeout=timeout,
        )
        self.embed_model = AzureOpenAIEmbedding(
            mdoel_name = 'text-embedding-ada-002',
            deployment_name = 'text-embedding-ada-002',
            api_key=embed_key,
            azure_endpoint=embed_endpoint,
            api_version=embed_api_version
        )
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        Settings.chunk_size = chunk_size
        Settings.overlap = overlap

    def complete(self, prompt):
        response = self.llm.complete(prompt)
        response = re.sub(r"\'re", "'re", response.text)
        return response

    def load_file(self, pdf_folder_path):
        self.documents = SimpleDirectoryReader(pdf_folder_path).load_data()
        self.index_ = VectorStoreIndex.from_documents(self.documents)
        self.query_engine = self.index_.as_query_engine(similarity_top_k=3)
    
    def load_from_text(self, documents):
        self.documents = documents
        self.index_ = VectorStoreIndex.from_documents(self.documents)
        self.query_engine = self.index_.as_query_engine(similarity_top_k=3)

    def query(self, prompt):
        success = False
        try_num = 0
        res = ''
        while not success:
            res = self.query_engine.query(prompt).response
            success = True
        return res


def convert_sections_to_documents(sections):
    """
    Convert sections with section headings, section text to documents.
    """
    documents = []
    for i in range(len(sections)):
        this_text = sections[i]['text']
        this_heading = sections[i]['heading']
        this_document = Document(text=this_text, metadata={'heading': this_heading})
        documents.append(this_document)
    return documents

def text_to_wav_old(voice_name, text, output_path):
    language_code = "-".join(voice_name.split("-")[:2])
    text_input = tts.SynthesisInput(text=text)
    voice_params = tts.VoiceSelectionParams(
        language_code=language_code, name=voice_name
    )
    audio_config = tts.AudioConfig(audio_encoding=tts.AudioEncoding.LINEAR16)

    client = texttospeech.TextToSpeechClient()  
    response = client.synthesize_speech(
        input=text_input,
        voice=voice_params,
        audio_config=audio_config,
    )

    with open(output_path, "wb") as out:
        out.write(response.audio_content)
        print(f'Generated speech saved to "{output_path}"')

def text_to_wav_sub(voice_name, text):
    """
    Convert text to speech and return the audio data as a BytesIO object.
    """
    language_code = "-".join(voice_name.split("-")[:2])
    text_input = tts.SynthesisInput(text=text)
    voice_params = tts.VoiceSelectionParams(
        language_code=language_code, name=voice_name
    )
    audio_config = tts.AudioConfig(audio_encoding=tts.AudioEncoding.MP3)

    client = tts.TextToSpeechClient()
    response = client.synthesize_speech(
        input=text_input,
        voice=voice_params,
        audio_config=audio_config,
    )

    return io.BytesIO(response.audio_content)

def text_to_wav(voice_name, text, output_path):
    """
    Generate speech from text in chunks, combine them, and save to a specified output path.
    """
    chunks = split_text(text)
    combined = AudioSegment.empty()
    
    for chunk in chunks:
        if len(chunk) == 0:
            continue
        success = False
        while success == False:
            try:
                audio_data = text_to_wav_sub(voice_name, chunk)
                success = True
            except:
                time.sleep(1)
        audio_segment = AudioSegment.from_mp3(audio_data)
        combined += audio_segment
    
    combined.export(output_path, format="mp3")
    print(f"Final combined speech saved to {output_path}")


def mergable(latex_page_1, latex_page_2):
    if '{figure}' not in latex_page_1 and '{figure}' not in latex_page_2:
        if len(latex_page_1) + len(latex_page_2) <= 1000:
            return True
        else:
            return False
    else:
        return False

def merge_latex(latex_page_1, latex_page_2):
    end_of_page_1 = latex_page_1.find('\\end{frame}')
    part_1 = latex_page_1[:end_of_page_1]
    start_of_page_2 = latex_page_2.find('\\begin{itemize}')
    part_2 = latex_page_2[start_of_page_2:]
    final_latex = part_1 + part_2
    return final_latex


def delete_prefix_space(input_str):
    res = input_str.split(' ')
    new_res = []
    for i in range(len(res)):
        if len(res[i]) > 0:
            new_res.append(res[i])
    res = ' '.join(new_res)
    return res


def query_llm(prompt_in, agent_in):
    return agent_in.query(prompt_in)

def worker(task):
    si, speech, output_latex_path = task
    audio_file_path = os.path.join(output_latex_path, 'audios', f'{str(si).zfill(3)}.mp3')
    text_to_wav("en-US-Journey-O", speech, audio_file_path)

def parallel_tts(speeches, sis, output_latex_path):
    # tasks = list(zip(sis, speeches))

    tasks = [(si, speech, output_latex_path) for si, speech in zip(sis, speeches)]
    # Set up the pool and the worker function
    pool = multiprocessing.Pool()
    results = pool.map(worker, tasks)
    # query_func = partial(worker, output_latex_path=output_latex_path)

    # Execute the worker function across the speech data using map
    # results = pool.map(query_func, tasks)

    # Close the pool and wait for all processes to complete
    pool.close()
    pool.join()

    # results would typically be used here, but for file I/O we don't collect results
    results = []
    return

 
def parallelize_queries(prompts_in, agent_in):
    #pool = multiprocessing.Pool()
    #query_func = partial(query_llm, agent_in=agent_in)
    #results = pool.map(query_func, prompts_in)
    #pool.close()
    #pool.join()
    results = []
    for i in range(len(prompts_in)):
        success = False
        while not success:
            #try:
            this_res = query_llm(prompts_in[i], agent_in)
            success = True
            #except:
            # time.sleep(1)
        results.append(this_res)
        print('finished generating ', prompts_in[i])
    return results


def extract_paper_data(paper_path = 'papers/example.pdf', extracted_data_path='extracted_paper_data',
                      output_html_parent_path='resultant_htmls'):
    # create a path that exclusively contain the paper pdf file
    paper_path_name = paper_path.split('/')[-1].split('.pdf')[0]
    paper_name = paper_path.split('/')[-1]
    parent_path = os.path.dirname(paper_path)
    paper_new_parent_path = os.path.join(parent_path, paper_path_name)
    os.makedirs(paper_new_parent_path, exist_ok=True)
    os.system('cp '+paper_path+' '+paper_new_parent_path+'/.')
    paper_new_path = os.path.join(paper_new_parent_path, paper_name)
    data_output_path = os.path.join(extracted_data_path, paper_path_name)
    os.system('rm -rf '+data_output_path)
    os.makedirs(data_output_path, exist_ok=False)
    os.makedirs('generated_pdf', exist_ok=True)

    figure_path = os.path.join(data_output_path, 'figures')
    article_dict, title, author, abstract, sections, figures, all_formulas, real_figures, real_tables = get_paper_info(paper_new_path, figure_path)
    return sections, real_figures, real_tables, title, author


def split_bullet_points(bullet_points, max_words=500):
    new_list = []
    i = 0
    this_len = 0
    this_list = []
    while i < len(bullet_points):
        if (this_len < max_words and this_len + len(bullet_points[i]) < max_words ) or len(this_list) == 0:
            this_len += len(bullet_points[i])
            this_list.append(bullet_points[i])
            i += 1
        else:
            this_len = 0
            new_list.append(this_list)
            this_list = []
    return new_list



class LatexConverter:
    def __init__(self):
        self.img_paths = []

    def convert_to_latex(self, title, contents, img_path=None, caption=None, max_points=5):
        result_latex_pages = []
        new_set_of_contents = split_bullet_points(contents, max_words=500 if img_path is None else 300)
        for content in new_set_of_contents:
            if img_path is not None and img_path not in self.img_paths:
                this_latex_page = convert_to_latex_small(title, content, img_path, caption)
                self.img_paths.append(img_path)
            else:
                this_latex_page = convert_to_latex_small(title, content) 
            result_latex_pages.append(this_latex_page)
        final_result_latex_pages = []
        
        return result_latex_pages


def convert_to_latex_small(title, contents, img_path=None, caption=None):
    latex_code = '''\\begin{frame}
                \\frametitle{''' + f'{title}' + '}'''
    new_contents = []
    if caption is not None:
        caption = caption.replace('%', '\\%')
    for content in contents:
        content = str(content) # .replace("'", '"')
        content = content.replace('%', '\%')
        start = content.find('{')
        end = content.find('}')
        content = content[start:end+1]
        try:
            content = json.loads(content)
        except:
            continue
        this_new_content = []
        for k in content:
            if type(content[k]) == type('abc'):
                this_new_content = [content[k]]
                break
        for k in content:
            if type(content[k]) == type([1]):
                this_new_content.append(content[k])
                break
        new_contents.append(this_new_content)
    contents = new_contents
                
    if img_path is None:
        latex_code = latex_code + '''\\begin{itemize}'''
        for content in contents:
            if len(content) == 0:
                continue
            elif len(content) == 1:
                latex_code = latex_code + f'\item {content[0]}'
            elif len(content) == 2 and type(content[1]) == type([1]) and len(content[1]) == 1:
                latex_code = latex_code + f'\item {content[0]}. {content[1][0]}'
            elif len(content) == 2 and type(content[1]) == type([1]) and len(content[1]) > 1:
                latex_code = latex_code + f'\item {content[0]}' +'\\begin{itemize}'
                for cc in content[1]:
                    latex_code = latex_code + f'\item {cc}'
                latex_code = latex_code + '\\end{itemize}'
            else:
                continue
                
        latex_code = latex_code + '\\end{itemize}\\end{frame}'
    else:
        r = np.random.randint(2)
        img = cv2.imread(img_path)
        if img.shape[0] * 1.5 < img.shape[1]:
            latex_code = latex_code + \
            '''\\begin{figure}
                    \\centering
                    \\includegraphics[width=0.8\\textwidth,height=0.45\\textheight]{'''
            latex_code = latex_code + f'{img_path}' + '}'
            latex_code = latex_code + '\\caption{\\tiny{'''+f'{caption}' + '}}'
            latex_code = latex_code + '''
                \\end{figure}'''
            latex_code = latex_code + '''\\begin{itemize}'''
            for content in contents:
                # latex_code = latex_code + f'\item {content}'
                if len(content) == 0:
                    continue
                elif len(content) == 1:
                    latex_code = latex_code + f'\item {content[0]}'
                elif len(content) == 2 and type(content[1]) == type([1]) and len(content[1]) == 1:
                    latex_code = latex_code + f'\item {content[0]}. {content[1][0]}'
                elif len(content) == 2 and type(content[1]) == type([1]) and len(content[1]) > 1:
                    latex_code = latex_code + f'\item {content[0]}' +'\\begin{itemize}'
                    for cc in content[1]:
                        latex_code = latex_code + f'\item {cc}'
                    latex_code = latex_code + '\\end{itemize}'
                else:
                    continue
            latex_code = latex_code + '\\end{itemize}\\end{frame}'
        else:
            if r== 0:
                latex_code = latex_code + \
                        '''\\begin{minipage}[t]{0.5\\linewidth}
                        \\begin{figure}
                        \\centering
                        \\includegraphics[height=0.8\\textheight,width=0.9\\textwidth]{'''+f'{img_path}' + \
                        '}' + \
                '''\\end{figure}
                \\end{minipage}\\hfill
                \\begin{minipage}[t]{0.5\\linewidth}
                \\begin{itemize}''' + f'\item Figure: {caption}'
                for content in contents:
                    # latex_code = latex_code + f'\item {content}'
                    if len(content) == 0:
                        continue
                    elif len(content) == 1:
                        latex_code = latex_code + f'\item {content[0]}'
                    elif len(content) == 2 and type(content[1]) == type([1]) and len(content[1]) == 1:
                        latex_code = latex_code + f'\item {content[0]}. {content[1][0]}'
                    elif len(content) == 2 and type(content[1]) == type([1]) and len(content[1]) > 1:
                        latex_code = latex_code + f'\item {content[0]}' +'\\begin{itemize}'
                        for cc in content[1]:
                            latex_code = latex_code + f'\item {cc}'
                        latex_code = latex_code + '\\end{itemize}'
                    else:
                        continue
                latex_code = latex_code + '\\end{itemize}\\end{minipage}\\end{frame}'
            else:
                latex_code = latex_code + \
                    '''\\begin{minipage}[t]{0.5\\linewidth}
                    \\begin{itemize}''' + f'\item Figure: {caption}'
                for content in contents:
                    if len(content) == 0:
                        continue
                    elif len(content) == 1:
                        latex_code = latex_code + f'\item {content[0]}'
                    elif len(content) == 2 and type(content[1]) == type([1]) and len(content[1]) == 1:
                        latex_code = latex_code + f'\item {content[0]}. {content[1][0]}'
                    elif len(content) == 2 and type(content[1]) == type([1]) and len(content[1]) > 1:
                        latex_code = latex_code + f'\item {content[0]}' +'\\begin{itemize}'
                        for cc in content[1]:
                            latex_code = latex_code + f'\item {cc}'
                        latex_code = latex_code + '\\end{itemize}'
                    else:
                        continue
                    # latex_code = latex_code + f'\item {content}'
                latex_code = latex_code + '''\\end{itemize}
                    \\end{minipage}\\hfill
                    \\begin{minipage}[t]{0.5\\linewidth}
                    \\begin{figure}
                    \\centering
                    \\includegraphics[height=0.8\\textheight, width=0.9\\textwidth]{'''+f'{img_path}'+\
                    '''}'''+ \
                    '''\\end{figure}
                    \\end{minipage}\\end{frame}'''
    return latex_code
        

def create_latex_from_paper(
    pdf_path,
    extracted_data_path,
    output_latex_parent_path='resultant_latex',
    azure_api_key,
    azure_endpoint,
    api_version,
    embed_azure_endpoint,
    embed_api_version,
    embed_key):

    # create agent
    agent = Agent(api_key=azure_api_key, end_point=azure_endpoint, api_version=api_version,
    embed_key=embed_key, embed_api_version=embed_api_version, embed_endpoint=embed_azure_endpoint)

    # extract paper content
    sections, real_figures, real_tables, paper_title, authors = extract_paper_data(pdf_path, extracted_data_path)
    print(authors)
    authors = authors.replace('%', '\\%')
    paper_title = paper_title.replace('%', '\\%')
    figures_captions = {}
    for f in real_figures:
        figures_captions[f] = real_figures[f]['caption']
    figures_captions_str = str(figures_captions)

    # extract documents
    documents = convert_sections_to_documents(sections)
    agent.load_from_text(documents)

    summary_prompt = """
    You are ChatGPT, a large language model trained by OpenAI.
    I need to present this paper on a conference and need to make a
    presentation slide. Please give a detailed
    summarization of the paper in terms of: motivation of this approach, related works and
    existing approaches, why existing approaches do not work well or have troubles, what
    are the individual methods' problems? then summarize the proposed method, including
    the very details on the exact method they are using, including but not limited to the
    pipeline they used, the method's main steps, what dataset they use, and how they are
    evaluating their results, metrics they used to evaluate their results. Then summarize
    their technical results, including the paper's own results and their ablation study results:
    what are the key take-aways from these results? Finally talk about the key insights from
    this paper. For each part, please summarize in bullet points, and make sure you
    generate no less than 5 bullet points, especially for the technical part, try to first
    understand the big picture, the training pipeline before you go on to dive deep to the
    details."""
    summary = agent.query(summary_prompt)

    # generate outlines
    outline_keys = ['Key Highlights', 'Introduction and Problem Setting',
                # 'Related Works', 
                'Proposed Research and Technical Details',
                'Technical Results', 'Conclusions and Remarks']
    
    outlines = {
        'Key Highlights': """This section should provide the key highlights of the
        paper, and the purpose is to give the most critical information about the paper
        such that readers can benefit from these key highlights. Make sure your contents contains at
        least 5 bullet points, also include the motivation and prior works briefly, and make sure
        to state the problem and the key highlights of the research.
        The key highlights should
        include but not limit to the key motivation of the paper, what problem it tries to solve,
        key insights of the approach taken by the paper, key contribution of
        the paper, main idea or method of the paper including the sketch of the method, 
        key results of the paper and what the authors want to convey based on the results, 
        and key conclusion of the paper. You will be punished if you cannot come up with meaningful contents""",
      
        'Introduction and Problem Setting': """This section should discuss about 
        problem settings, the motivation, and the specific problem the authors try to tackle,
        why they want to tackle this problem, why that problem is important and why previously 
        it is not solved appropriately or not solved
        with high quality, and what are the new key things they did to improve for the problem setting.
        Please find appropriate sections in the
        paper first to find these details and then give your outlines. Please do not add in details
        that are not presented in the original paper.""",
            
        # 'Related Works': """This part should include several slides for the related works, 
        # and when appropriate mention the limitations of existing approaches if this paper tries
        # to improve over existing baselines. Please include these baseline approach details 
        # and the reason why they are not good enough, or other relevant research that are important
        # for discussing this paper, for example, the papers or methods that are mentioned with great
        # attention in the paper can be good candidates for these baselines. """,
        
        'Proposed Research and Technical Details' : """This part should include several slides
        for describing the methods: make sure
        to describe the exact method they proposed in the research, including the steps to train models
        if they contain such details. To do this, first read the method or main body of the paper one by one,
        end think about how to introduce the method in a way that is easy for audience to understand, and keep
        this in mind when generating the details. A good example would be to list steps of executing the algorithms or layout the pipeline to train and / or evaluate the model. 
        
        Make sure to also summarize the key methods section first to help you generate the section contents; 
        instead of boringly list a laundry list of training parameters or networks, think about 
        how the authors formulate this problem, how they approach this problem, is the way they solve the problem
        anything different from their previous approach? Discuss about what questions they
        sought to ask, how they design the experiments, what particular method they use, what
        aspects they have improved, etc. You should definitely include more details in the method
        section. Provide examples that appears in the paper if possible to demonstrate how the method works if possible. If there are algorithms presentend in the paper, give a detailed step-by-step explanation of the algorithm by listing out the details.
 
        You should bear this in mind: after reading the slides content, a student should be able to
        summarize the paper's proposed approach. If you find the results cannot achieve this level of details,
        go back and regenerate the content.""",
        
        'Technical Results': """This part should contain 2 sections for the results of the paper and summarize what are the key
        results, what does the result implicate, and if there are any ablation studies, what are
        those results and their implications. If you cannot come up with results, and make sure to come up with details as much as you can, please find technical results and their discussion details, otherwise you would be punished.""",
        
        'Conclusions and Remarks': """This part should contain 1 to 2 sections for conclusions and
        remarks. restate the key contributions of the approach,
        and list key takeaways for the audience to take a final look. If you struggle with 
        coming up with details, refer to the paper for the conclusion and discussions. Or you
        can use your own knowledge to present ideas or discussions on this research. Do not
        leave blank and you will be punished if this is blank.""",
    }

    section_topic = []
    section_outlines = {}
    section_contents = {}
    section_figures = {}

    all_slide_outline_prompts = []
    for k in outline_keys:
        print('generating ', k, ' ...... ')
        section_topic.append(k)
        this_section = outlines[k]
        slide_outline_prompt = f"""
        You are ChatGPT, a large language model trained by OpenAI.
        I need to present this paper on a conference with a presentation slides, 
        and I already generated a summary of this paper here quoted in triple backticks: ```{summary}```. 
        Your task is to generate the contents outlines that can be used for 
        making the presentation slides. You will be given a specific section topic
        about this paper and you will need to make content outlines based on the given
        section topic. Please make sure to include as much details as possible so later on
        you can based on the outlines to generate the actual slides. Now generate slide content
        detailed outlines for this section topic: {k}, with the detailed requirement here
        quoted with double backticks: ``{this_section}``. Please output your contents directly in bullet points
        so that I can directly put the content into a slide. Make sure to add a \n symbol after each
        bullet points.

        For example, your output should be like:
        
        - CONTENT1\n
        - CONTENT2\n
        - CONTENT3\n
        
        Some hints and requirements on outline generation: 
        Once you generated the content, if you cannot find the relevant details, 
        try to look into the paper for multiple times and bring your own knowledge 
        to find the details and required contents as much as you can. If you cannot 
        find the referred details, please find something closely relevant. For example,
        for a paper that is not working on new methods, but rather studying existing system's
        capabiltiies, when discussing related works, instead of discussing the pros and cons 
        of various baseline methods if the paper did not mention the pros
        and cons of any baselines, you can discuss what people have
        done, what lessons or conclusions can be drawn from those related research.
       
        Do not put too much words here as you will need to expand them later on. 
 
        You cannot leave a note like ``The paper
        does not have this part ``, you will be punished if you cannot find the 
        required contents or cannot find related contents
        since all contents should be within the paper I give you."""
        all_slide_outline_prompts.append(slide_outline_prompt)

    all_slide_outlines = parallelize_queries(all_slide_outline_prompts, agent)
    print('generated all slide outlines')
    all_slide_content_prompts = []
    all_slide_figure_prompts = []

    for k, section_idx in zip(outline_keys, range(len(all_slide_outlines))):
        section_outlines[k] = all_slide_outlines[section_idx].replace('\n', '')

        prompt_for_content = f"""You are ChatGPT, a large language model trained by OpenAI.
        I need to present this paper on a research conference and want a presentation slides, 
        and I already generated a summary of this paper here quoted in triple backticks: ```{summary}```. 
        Your task is to generate the actual slide contents based on the outline
        presented later. When generating the slides content,
        make sure to use include the details that you can find in the original research paper.
        Make sure to use concise words to present the content as it is not a paper but rather
        a presentation. Stick to the contents outline I provided and once you generate the contents,
        try to imagine that you are the audience, do you find the contents interesting and also easy
        to understand? If not, try to improve that slide contents. 

        Please also make sure you limit the total number of words under 200 words in your generated content.

        Please output your contents directly in JSON format. With the key being bullet points and details,
        and each bullet point should be separated by the symbol \n while each bullet point follow the format
        of """ + """{"bullet_point": YOUR_BULLET_POINT_CONTENT_AS_A_STRING, "details":[ADDITIONAL_DETAILS_AS_A_LIST]}

        For example, your output should be like 

        {"bullet_point": "BULLET_POINT_CONTENT 1", "details": [DETAILS_1, DETAILS_2, ...]} \n
        {"bullet_point": "BULLET_POINT_CONTENT 2", "details": [DETAILS_1, DETAILS_2, ...]} \n""" + \
        f"""Please make sure to use double quote for strings since JSON cannot parse single quote as string quoter.
        Now based on this content outline about this paper: ```{section_outlines[k]}```, your output should be:"""

        prompt_for_figure = f"""You are ChatGPT, a large language model trained by OpenAI.
        I am a researcher trying to present 
        this paper on a conference and need to make a presentation slides, 
        and I already generated a summary here quoted in triple backticks
        about an overview of this paper: ```{summary}```.
        Your task is to select one figure from a list of candidate figures,
        so as to help with explaining concepts or ideas in the section the actual slide contents based on the outline
        presented later. I will give you the section topic you are going to cover, as well as a list of 
        candidate figures captions, presented in JSON format,
        with the key being the figure symbol, and value being the caption. Please evaluate each of the figure to see if
        it aligns well with the topic and select the BEST fit figure that you think it is approriate to put in the slide.
        Add a bullet point in the format of figure symbol
        and an appropriate caption that you generate based on the original caption to better showcase the idea.

        For example, to add figure with symbol `1`, add the following bullet point:
        ```Figure: """
        prompt_for_figure = prompt_for_figure + """{1: CAPTIONS} ```"""
        prompt_for_figure = prompt_for_figure + """

        Please output your contents directly in bullet points
        so that I can directly put the content into a slide. 

        For example, your output should be 

        Figure: {1: CAPTIONS} """
        prompt_for_figure = prompt_for_figure + f"""
        Some hints: when discussing the general framework of a paper, you should include the teaser figure that discovers
        the research overview, while discussing results of the research paper, you should include the figure that presents
        the result, etc. Now based on this content outline of this section: ```{section_outlines[k]}```, and given these figures:
        ```{figures_captions_str}```,  the only ONE figure you selected is: """
        
        if k == 'Introduction and Problem Setting' or k == 'Proposed Research and Technical Details' or k == 'Technical Results':
            all_slide_figure_prompts.append(prompt_for_figure)
        all_slide_content_prompts.append(prompt_for_content)

    all_slide_contents = parallelize_queries(all_slide_content_prompts, agent)
    print('generated all slide contents')
    all_slide_figures = parallelize_queries(all_slide_figure_prompts, agent)
    print('generated all slide figures')

    section_contents = {}
    fig_idx = 0
    for k, section_idx in zip(outline_keys, range(len(all_slide_contents))):
        section_contents[k] = all_slide_contents[section_idx]
        if section_idx in [1,3,4]:
            section_contents[k] = all_slide_figures[fig_idx] + '\n' + section_contents[k]
            fig_idx += 1
    
    all_section_contents = {}
    real_figures_new = {}
    for k in real_figures:
        # remove empty space in figure name
        k_new = k.replace(' ', '')
        real_figures_new[k_new] = real_figures[k]['fig_path']
    for kk in outline_keys:
        all_section_contents[kk] = []
        for cnt in section_contents[kk].split('\n'):
            if len(cnt) > 1:
                if ('Figure' in cnt or 'figure' in cnt) and '{' in cnt and '}' in cnt:
                    cnt_type = 'figure'
                else:
                    cnt_type = 'bullet_point'
                if cnt_type == 'figure':
                    figure_id = cnt.split('{')[1].split(':')[0]
                    figure_id = figure_id.replace(' ', '')
                    captions = cnt.split('{')[1].split(':')[1].split('}')[0]
                    captions = delete_prefix_space(captions)
                    fig_path = None
                    figure_id = figure_id.replace('"', '')
                    figure_id = figure_id.replace("'", '')
                    for key in real_figures_new:
                        key = key.replace('"', '').replace("'", '')
                        try:
                            if int(figure_id) == int(key):
                                fig_path = real_figures_new[key]
                                break
                        except:
                            pass
                    if fig_path is None:
                        print('figure id is ', len(figure_id), ' key is ', key, ' and it causes fig path to be None')
                    else:
                        # fig_path_new = get_path_difference((), fig_path)
                        fig_path_new = fig_path
                        all_section_contents[kk].append({cnt_type: [fig_path_new, captions]})
                else:    
                    cnt = delete_prefix_space(cnt)
                    all_section_contents[kk].append({cnt_type: cnt})

    latex_pages = []
    output_latex_path = os.path.join(output_latex_parent_path, pdf_path.split('/')[-1].split('.pdf')[0])
    os.makedirs(output_latex_path, exist_ok=True)
    number_of_points_per_page = 10
    page_idx = 0
    used_figures = []
    latex_converter = LatexConverter()

    for key in outline_keys:
        idx = 0
        latex_pages_this_part = []
        while idx < len(all_section_contents[key])-number_of_points_per_page+1:
            all_cnt_types = [list(all_section_contents[key][idx+j].keys())[0] for j in range(number_of_points_per_page)]
            page_latex = None
            if 'figure' not in all_cnt_types:
                these_contents = [all_section_contents[key][idx+j]['bullet_point'] for j in range(number_of_points_per_page)]
                page_latex = latex_converter.convert_to_latex(title=key, contents=these_contents)
                idx += number_of_points_per_page
            else:
                these_contents = []
                idxes = [idx+j for j in range(number_of_points_per_page)]
                this_fig_path = None
                this_caption = None 
                for cnt_type, ii in zip(all_cnt_types, idxes):
                    if cnt_type == 'bullet_point':
                        this_content = all_section_contents[key][ii]['bullet_point']
                        these_contents.append(this_content)
                    else:
                        this_caption = all_section_contents[key][ii]['figure'][1]
                        this_fig_path = all_section_contents[key][ii]['figure'][0]
                if this_fig_path is not None and this_caption is not None and this_fig_path not in used_figures:
                    page_latex = latex_converter.convert_to_latex(title=key, contents=these_contents, img_path=this_fig_path, caption=this_caption)
                    used_figures.append(this_fig_path)
                else:
                    page_latex = latex_converter.convert_to_latex(title=key, contents=these_contents)
                idx += number_of_points_per_page
            if page_latex is not None:
                latex_pages_this_part = latex_pages_this_part + page_latex
        if idx < len(all_section_contents[key]): 
            rest_cnt_types = [list(all_section_contents[key][idx+j].keys())[0] for j in range(len(all_section_contents[key])-idx)]
            page_latex = None
            if 'figure' not in rest_cnt_types:
                these_contents = [all_section_contents[key][idx+j]['bullet_point'] for j in range(len(all_section_contents[key])-idx)]
                page_latex = latex_converter.convert_to_latex(title=key, contents=these_contents)
            else:
                these_contents = []
                idxes = [idx+j for j in range(len(all_section_contents[key])-idx)]
                this_fig_path, this_caption = None, None
                for cnt_type, ii in zip(rest_cnt_types, idxes):
                    if cnt_type == 'bullet_point':
                        this_content = all_section_contents[key][ii]['bullet_point']
                        these_contents.append(this_content)
                    else:
                        this_caption = all_section_contents[key][ii]['figure'][1]
                        this_fig_path = all_section_contents[key][ii]['figure'][0]
                if this_fig_path is not None and this_caption is not None and this_fig_path not in used_figures:
                    page_latex = latex_converter.convert_to_latex(title=key,
                                                contents=these_contents,
                                                img_path=this_fig_path,
                                                caption=this_caption)
                    used_figures.append(this_fig_path)
                else:
                    page_latex = latex_converter.convert_to_latex(title=key, contents=these_contents)
            if page_latex is not None:
                latex_pages_this_part = latex_pages_this_part + page_latex
        # merge latex pages
        if len(latex_pages_this_part) > 0:
            merged_latex_pages = [latex_pages_this_part[0]]
            for tmp_latex_pg_idx in range(1, len(latex_pages_this_part)):
                is_mergable = mergable(merged_latex_pages[-1], latex_pages_this_part[tmp_latex_pg_idx])
                if is_mergable:
                    this_merged_latex_page = merge_latex(merged_latex_pages[-1], latex_pages_this_part[tmp_latex_pg_idx])
                    merged_latex_pages[-1] = this_merged_latex_page
                else:
                    merged_latex_pages.append(latex_pages_this_part[tmp_latex_pg_idx])
            latex_pages = latex_pages + merged_latex_pages
        
    preface = '''\\documentclass[aspectratio=169]{beamer}
    \\usepackage[utf8]{inputenc}
    \\usepackage{graphicx}
    \\usepackage{amsmath}
    \\usepackage{amsfonts}
    \\usepackage{amssymb}
    \\usepackage{multicol}
    \\title{\\fontsize{34}{34}\\selectfont ''' + f'{paper_title}' + '''}
    \\author{''' + f'{authors}' + '''}
    \\date{}


    \\begin{document}

    \\begin{frame}
    \\titlepage
    \\end{frame}'''
    
    with open(os.path.join(output_latex_path, 'final.tex'), 'w') as f:
        f.write(preface)
        for p in latex_pages:
            f.write(p)
        f.write('\\end{document}')
    
    # print('pdflatex')
    cwd = os.getcwd()
    os.chdir(output_latex_path)
    os.system('pdflatex -interaction=nonstopmode final.tex')
    os.chdir(cwd)

    return latex_pages, paper_title, authors, summary, output_latex_path, agent

def create_speech(latex_pages, paper_title, authors, summary, output_latex_path, agent):
    print('starting generating speech')
    section_speeches = []
    for page_idx in tqdm(range(len(latex_pages)+1)):
        if page_idx == 0:
            speech_generation_prompt = f"""
            You are ChatGPT, a large language model trained by OpenAI.
            You are tasked with creating an engaging introduction speech 
            script for the first slide of a conference presentation, which displays the title and authors
            of the academic paper. Begin by greeting the audience and introducing the topic and authors 
            in a way that captures interest and conveys the significance of the research.

            Instructions:
            - Start with a welcoming greeting suitable for a diverse audience of researchers and possibly a wider YouTube viewership but
            avoid boring words like "good morning everyone", "good day everyone", be more creative. By the way, the youtube channel name
            is "Trend in Research" if that helps. 
            - Mention the title of the paper prominently and clearly to emphasize the focus of the presentation.
            - Briefly set the context or the motivation behind the research, preparing the audience for the detailed discussion that will follow in subsequent slides.
            - No need to introduce the authors

            Aim for a speech script that is concise yet informative, ideally lasting about 10 seconds or below 50 words 
            to keep the introduction brisk but substantial.
            This introduction should not only inform but also intrigue the audience about the research's implications and the insights that will be discussed.

            Please generate the speech script for this introductory slide based on above guidelines, given the title is ```{paper_title}```,
            paper summary is ```{summary}```, and author list is ```{authors}```."""

            speech_content = agent.query(speech_generation_prompt)
            section_speeches.append(speech_content)
        else:
            speech_generation_prompt = f"""You are ChatGPT, a large language model trained by OpenAI.
            You are tasked with continuing the speech script for a conference presentation 
            based on this academic paper. The summary of the paper and previous slide's speech script are provided. Please
            aim for a smooth and logical progression but do not need to always mention again previous discussed content. Treat
            these slides as a coherent speech that you will tell in one shot instead of always summarize the previous one and 
            introduce the next one. Please do not say something like `In this slide, we introduce the highlights of this research`
            or something similar, and please directly introduce the content. Also do not say something like `stay tuned as we introduce
            something later`, please keep your conversation concise and informative, and avoid such useless sentences. 

            Previous slide's speech content is provided within triple backticks ```{speech_content[-1]}```. 
            For the current slide's content, listed as ```{latex_pages[page_idx-1]}```, include bullet points and any 
            figures with captions. Each slide should ideally be covered in about 20 seconds, targeting approximately 50 to 100 words 
            to maintain succinctness and clarity. Remember, your audience consists of researchers with varying levels of experience, 
            and the presentation will also be adapted for a YouTube video. Aim for engaging and accessible content, 
            make sure to explain complex concepts. Please do not spell out the latex code like `item` `itemized` etc., and make sure your output is a articulated presentation speech script. Now, please generate the speech script for the 
            current slide based on the provided content:"""


            speech_content = agent.query(speech_generation_prompt)
            section_speeches.append(speech_content)
        print(page_idx, len(latex_pages))
    os.makedirs(output_latex_path + '/audios', exist_ok=True)
    s_i = 0
    sis = list(np.arange(len(section_speeches)))
    parallel_tts(section_speeches, sis, output_latex_path)
    return


if __name__ == '__main__':
    # please replace these credentials with your microsoft azure account details
    azure_api_key = AZURE_API_KEY
    azure_endpoint = AZURE_END_POINT
    api_version = AZURE_API_VERSION
    embed_azure_endpoint = EMBED_AZURE_ENDPOINT
    embed_api_version = EMBED_API_VERSION
    embed_key = EMBED_KEY
  
    latex_pages, paper_title, authors, summary, output_latex_path, agent = create_latex_from_paper(
      pdf_path='papers/example.pdf',
      extracted_data_path='extracted_paper_data',
      output_latex_parent_path='resultant_latex',
      azure_api_key=azure_api_key,
      azure_endpoint=azure_endpoint,
      api_version=api_version,
      embed_azure_endpoint=embed_azure_endpoint,
      embed_api_version=embed_api_version,
      embed_key=embed_key,
    )
    pkl.dump([latex_pages, paper_title, authors, summary, output_latex_path, agent], open('result.pkl','wb'))
    # latex_pages, paper_title, authors, summary, output_latex_path, agent = pkl.load(open('result.pkl', 'rb'))
    create_speech(latex_pages, paper_title, authors, summary, output_latex_path, agent)
    generate_video_from_pdf(pdf_file=output_latex_path+'/final.pdf', audio_dir=output_latex_path+'/audios', output_video=output_latex_path+'/video.mp4')
    #"""
