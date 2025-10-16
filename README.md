<div align="center">
  <img src="images/icon.png" alt="Algorithm icon">
  <h1 align="center">infer_nanonets_ocr2</h1>
</div>
<br />
<p align="center">
    <a href="https://github.com/Ikomia-hub/infer_nanonets_ocr2">
        <img alt="Stars" src="https://img.shields.io/github/stars/Ikomia-hub/infer_nanonets_ocr2">
    </a>
    <a href="https://app.ikomia.ai/hub/">
        <img alt="Website" src="https://img.shields.io/website/http/app.ikomia.ai/en.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/Ikomia-hub/infer_nanonets_ocr2/blob/main/LICENSE.md">
        <img alt="GitHub" src="https://img.shields.io/github/license/Ikomia-hub/infer_nanonets_ocr2.svg?color=blue">
    </a>    
    <br>
    <a href="https://discord.com/invite/82Tnw9UGGc">
        <img alt="Discord community" src="https://img.shields.io/badge/Discord-white?style=social&logo=discord">
    </a> 
</p>

[Nanonets-OCR2](https://nanonets.com/research/nanonets-ocr-2/) by Nanonets is a family of powerful, state-of-the-art image-to-markdown OCR models that go far beyond traditional text extraction. It transforms documents into structured markdown with intelligent content recognition and semantic tagging, making it ideal for downstream processing by Large Language Models (LLMs).

Nanonets-OCR2 is packed with features designed to handle complex documents with ease:

- `LaTeX Equation Recognition`: Automatically converts mathematical equations and formulas into properly formatted LaTeX syntax.
- `Intelligent Image Description`: Describes images within documents using structured <img> tags, making them digestible for LLM processing. It can describe various image types, including logos, charts, graphs and so on, detailing their content, style, and context.
- `Signature Detection & Isolation`: Identifies and isolates signatures from other text, outputting them within a <signature> tag. This is crucial for processing legal and business documents.
- `Watermark Extraction`: Detects and extracts watermark text from documents, placing it within a <watermark> tag.
- `Smart Checkbox Handling`: Converts form checkboxes and radio buttons into standardized Unicode symbols (☐, ☑, ☒) for consistent and reliable processing.
- `Complex Table Extraction: Accurately extracts complex tables from documents and converts them into both markdown and HTML table formats.
- `Flow charts & Organisational charts`: Extracts flow charts and organisational as mermaid code.
- `Handwritten Documents`: The model is trained on handwritten documents across multiple languages.
- `Multilingual`: Model is trained on documents of multiple languages, including English, Chinese, French, Spanish, Portuguese, German, Italian, Russian, Japanese, Korean, Arabic, and many more.
Visual Question Answering (VQA): The model is designed to provide the answer directly if it is present in the document; otherwise, it responds with "Not mentioned."


## :rocket: Use with Ikomia API

#### 1. Install Ikomia API

We strongly recommend using a virtual environment. If you're not sure where to start, we offer a tutorial [here](https://www.ikomia.ai/blog/a-step-by-step-guide-to-creating-virtual-environments-in-python).

```sh
pip install ikomia
```

#### 2. Create your workflow

```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_nanonets_ocr2", auto_connect=True)

# Run on your image  
wf.run_on(url="https://github.com/NanoNets/Nanonets-OCR2/blob/main/assets/bank_statement.jpg?raw=true")

# Save output .json
nanonets = algo.get_output(1)
nanonets.save('nanonets_output.json')
```

## :sunny: Use with Ikomia Studio

Ikomia Studio offers a friendly UI with the same features as the API.

- If you haven't started using Ikomia Studio yet, download and install it from [this page](https://www.ikomia.ai/studio).
- For additional guidance on getting started with Ikomia Studio, check out [this blog post](https://www.ikomia.ai/blog/how-to-get-started-with-ikomia-studio).

## :pencil: Set algorithm parameters

| Parameters             | Description |
|----------------------|-------------|
| `input_path`       | Path to a single PDF file to process or to a directory containing multiple PDFs. |
| `model_name`       | Name or path of the Qwen VL model. Default: `"nanonets/Nanonets-OCR2-3B"`. |
| `prompt`          | Custom prompt to guide the model's response for the given image. Default: `"Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes."` |
| `system_prompt`   | System prompt to set the behavior and context for the model. Default: `"You are a helpful assistant that extracts text from documents. You are given a document and you need to extract the text from the document. You are also given a prompt that tells you what to extract. You need to extract the text from the document and return it in the format specified in the prompt."` |
| `cuda`             | If True, CUDA-based inference (GPU). If False, run on CPU. |
| `max_new_tokens`  | The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt. Default: `4096`.|
| `input_size`     | Size of the input image. Default: `1024`.|



```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_nanonets_ocr2", auto_connect=True)

algo.set_parameters({
    "model_name": "nanonets/Nanonets-OCR2-3B",
    "cuda": "True",
    "prompt": "Describe the image in detail.",
    "max_new_tokens": "4096", 
    "input_size": "1024"
})

# Run on your image  
wf.run_on(url='https://github.com/Ikomia-dev/notebooks/blob/main/examples/img/img_text_inspiration.jpg?raw=true')

# Save output .json
qwen_output = algo.get_output(1)
qwen_output.save('nanonets_output.json')
```

## :mag: Explore algorithm outputs

Every algorithm produces specific outputs, yet they can be explored them the same way using the Ikomia API. For a more in-depth understanding of managing algorithm outputs, please refer to the [documentation](https://ikomia-dev.github.io/python-api-documentation/advanced_guide/IO_management.html).

```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_nanonets_ocr2", auto_connect=True)

# Run on your image  
wf.run_on(url='https://github.com/Ikomia-dev/notebooks/blob/main/examples/img/img_text_inspiration.jpg?raw=true')

# Iterate over outputs
for output in algo.get_outputs():
    # Print information
    print(output)
    # Export it to JSON
    output.to_json()
```
