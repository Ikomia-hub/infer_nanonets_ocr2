import copy
import os
import torch
from PIL import Image
from ikomia import core, dataprocess, utils


from transformers import AutoProcessor, AutoModelForImageTextToText


# --------------------
# - Class to handle the algorithm parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferNanonetsOcr2Param(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        self.model_name = "nanonets/Nanonets-OCR2-3B"
        self.cuda = torch.cuda.is_available()
        self.prompt = "Extract the text from the above document as if you were reading " \
                    "it naturally. Return the tables in html format. Return the " \
                    "equations in LaTeX representation. If there is an image in " \
                    "the document and image caption is not present, add a small " \
                    "description of the image inside the <img></img> tag; otherwise, " \
                    "add the image caption inside <img></img>. Watermarks should be " \
                    "wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. " \
                    "Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> " \
                    "or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes."
        self.system_prompt = "You are a helpful assistant that extracts text from documents. " \
                        "You are given a document and you need to extract the text " \
                        "from the document. You are also given a prompt that tells " \
                        "you what to extract. You need to extract the text from the " \
                        "document and return it in the format specified in the prompt."
        self.max_new_tokens = 4096
        self.input_size = 1024
        self.update = False

    def set_values(self, params):
        self.model_name = str(params["model_name"])
        self.cuda = utils.strtobool(params["cuda"])
        self.prompt = str(params["prompt"])
        self.system_prompt = str(params["system_prompt"])
        self.max_new_tokens = int(params["max_new_tokens"])
        self.input_size = int(params["input_size"])
        self.update = True

    def get_values(self):
        # Send parameters values to Ikomia Studio or API
        # Create the specific dict structure (string container)
        params = {
            "model_name": str(self.model_name),
            "cuda": str(self.cuda),
            "prompt": str(self.prompt),
            "system_prompt": str(self.system_prompt),
            "max_new_tokens": str(self.max_new_tokens),
            "input_size": str(self.input_size),
        }
        return params


# --------------------
# - Class which implements the algorithm
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferNanonetsOcr2(dataprocess.C2dImageTask):

    def __init__(self, name, param):
        dataprocess.C2dImageTask.__init__(self, name)
        self.add_output(dataprocess.DataDictIO())

        # Create parameters object
        if param is None:
            self.set_param_object(InferNanonetsOcr2Param())
        else:
            self.set_param_object(copy.deepcopy(param))
        
        self.model = None
        self.processor = None           
        self.base_dir = os.path.dirname(os.path.realpath(__file__))
        self.model_folder = os.path.join(self.base_dir, "weights")
        self.device = torch.device("cpu")

    def load_model(self):
        param = self.get_param_object()
        self.device = torch.device(
            "cuda") if param.cuda and torch.cuda.is_available() else torch.device("cpu")
        torch_tensor_dtype = torch.float16 if param.cuda and torch.cuda.is_available() else torch.float32
        # Initialize model and processor
        self.model = AutoModelForImageTextToText.from_pretrained(
            param.model_name,
            dtype=torch_tensor_dtype,
            # attn_implementation="flash_attention_2",
            device_map=self.device,
            cache_dir=self.model_folder
        )
        self.processor = AutoProcessor.from_pretrained(
            param.model_name,
            cache_dir=self.model_folder
        )

        param.update = False


    def init_long_process(self):
        self.load_model()
        super().init_long_process()


    def resize_with_min_size(self, image: Image.Image, min_size: int = 2048) -> Image.Image:
        width, height = image.size
        short_side = min(width, height)

        # if short_side >= min_size:
        #     return image

        scale = min_size / short_side
        new_width = int(width * scale)
        new_height = int(height * scale)

        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return resized_image


    def get_progress_steps(self):
        # Function returning the number of progress steps for this algorithm
        # This is handled by the main progress bar of Ikomia Studio
        return 1

    def run(self):
        self.begin_task_run()
        # Get parameters
        param = self.get_param_object()

        # Get input image (np array):
        img_input = self.get_input(0)

        # Get image from input/output (numpy array):
        src_image = img_input.get_image()

        # transform image to PIL format
        src_image = Image.fromarray(src_image)
        src_image = self.resize_with_min_size(src_image, param.input_size)

        # Set output
        output_txt = self.get_output(1)

        if param.update:
            self.load_model()       

        # Get parameters
        param = self.get_param_object()

        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": param.system_prompt},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": src_image},
                    {"type": "text", "text": param.prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[src_image], padding=True, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        
        output_ids = self.model.generate(**inputs, max_new_tokens=param.max_new_tokens, do_sample=False)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        
        output_text = self.processor.batch_decode(
                                    generated_ids,
                                    skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False
                            )[0]

        print(output_text)
        output_txt.data = {
            "response": output_text,
        }
        # Step progress bar (Ikomia Studio):
        self.emit_step_progress()

        # Call end_task_run() to finalize process
        self.end_task_run()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferNanonetsOcr2Factory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set algorithm information/metadata here
        self.info.name = "infer_nanonets_ocr2"
        self.info.short_description = "Transform documents into structured markdown with " \
                                    "intelligent content recognition and semantic tagging"
        # relative path -> as displayed in Ikomia Studio algorithm tree
        self.info.path = "Plugins/Python/VLM"
        self.info.version = "1.0.1"
        self.info.icon_path = "images/icon.png"
        self.info.authors = "Souvik Mandal and Ashish Talewar and Siddhant " \
                            "Thakuria and Paras Ahuja and Prathamesh Juvatkar"
        self.info.article = "Nanonets-OCR2: A model for transforming documents into structured " \
                             "markdown with intelligent content recognition and semantic tagging"
        self.info.journal = ""
        self.info.year = 2025
        self.info.license = "Apache 2.0"

        # Ikomia API compatibility
        self.info.min_ikomia_version = "0.15.0"

        # Python compatibility
        self.info.min_python_version = "3.9.0"

        # URL of documentation
        self.info.documentation_link = "https://arxiv.org/abs/2502.13923"

        # Code source repository
        self.info.repository = "https://github.com/Ikomia-hub/infer_qwen2_5_vl"
        self.info.original_repository = "https://github.com/QwenLM/Qwen2.5-VL"

        # Keywords used for search
        self.info.keywords = "OCR,Markdown,Nanonets,VQA,Vision-Language"

        # General type: INFER, TRAIN, DATASET or OTHER
        self.info.algo_type = core.AlgoType.INFER
        self.info.algo_tasks = "OCR"

        # Min hardware config
        self.info.hardware_config.min_cpu = 4
        self.info.hardware_config.min_ram = 16
        self.info.hardware_config.gpu_required = False
        self.info.hardware_config.min_vram = 6

    def create(self, param=None):
        # Create algorithm object
        return InferNanonetsOcr2(self.info.name, param)
