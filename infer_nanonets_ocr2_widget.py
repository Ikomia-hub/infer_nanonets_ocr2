from torch.cuda import is_available
from ikomia import core, dataprocess
from ikomia.utils import pyqtutils, qtconversion
from infer_nanonets_ocr2.infer_nanonets_ocr2_process import InferNanonetsOcr2Param

# PyQt GUI framework
from PyQt5.QtWidgets import *


# --------------------
# - Class which implements widget associated with the algorithm
# - Inherits PyCore.CWorkflowTaskWidget from Ikomia API
# --------------------
class InferNanonetsOcr2Widget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = InferNanonetsOcr2Param()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.grid_layout = QGridLayout()

        # Model name
        self.edit_model = pyqtutils.append_edit(
            self.grid_layout, "Model name", self.parameters.model_name)
        
                # Prompt
        self.edit_prompt = pyqtutils.append_edit(self.grid_layout, "Prompt", self.parameters.prompt)
        self.edit_system_prompt = pyqtutils.append_edit(self.grid_layout, "System Prompt", self.parameters.system_prompt)

        # Cuda
        self.check_cuda = pyqtutils.append_check(
            self.grid_layout, "Cuda", self.parameters.cuda and is_available())

        # Max New Tokens
        self.spin_max_new_tokens = pyqtutils.append_spin(
            self.grid_layout, "Max New Tokens", self.parameters.max_new_tokens, min=1, max=10000)


        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.grid_layout)
        self.set_layout(layout_ptr)

    def on_apply(self):
        # Update parameters from widget values
        self.parameters.model_name = self.edit_model.text()
        self.parameters.prompt = self.edit_prompt.text()
        self.parameters.system_prompt = self.edit_system_prompt.text()
        self.parameters.cuda = self.check_cuda.isChecked()
        self.parameters.max_new_tokens = self.spin_max_new_tokens.value()
        self.parameters.update = True

        # Send signal to launch the process
        self.emit_apply(self.parameters)



# --------------------
# - Factory class to build algorithm widget object
# - Inherits PyDataProcess.CWidgetFactory from Ikomia API
# --------------------
class InferNanonetsOcr2WidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the algorithm name attribute -> it must be the same as the one declared in the algorithm factory class
        self.name = "infer_nanonets_ocr2"

    def create(self, param):
        # Create widget object
        return InferNanonetsOcr2Widget(param, None)
