from ikomia import dataprocess


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits PyDataProcess.CPluginProcessInterface from Ikomia API
# --------------------
class IkomiaPlugin(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def get_process_factory(self):
        # Instantiate algorithm object
        from infer_nanonets_ocr2.infer_nanonets_ocr2_process import InferNanonetsOcr2Factory
        return InferNanonetsOcr2Factory()

    def get_widget_factory(self):
        # Instantiate associated widget object
        from infer_nanonets_ocr2.infer_nanonets_ocr2_widget import InferNanonetsOcr2WidgetFactory
        return InferNanonetsOcr2WidgetFactory()
