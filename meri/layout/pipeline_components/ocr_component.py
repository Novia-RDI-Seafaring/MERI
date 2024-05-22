import deepdoctection as dd
import os
from .utils import CONFIGS_PATH

class OCRComponent(dd.TextExtractionService):
    ''' Extracts text. Depending on specified method different OCR methods
    are executed.

    Supported methods:
        - tesseract: apply tesseract OCR
        - doctr: requires that words have been detected (doctr_textdetector). Extracts text from 
                detected words.

    Detections are added to annotations as dd.LayoutType.word
    '''

    def __init__(self, cfg_path: str ='config/tesseract_config.yaml', method='tesseract'):
        ''' provide path to config.yaml. 
        '''

        if not os.path.exists(cfg_path):
            cfg_path = os.path.join(CONFIGS_PATH, cfg_path)
        
        if method == 'tesseract':
            detector = dd.TesseractOcrDetector(cfg_path)
            kwargs = {}
        elif method == 'doctr':
            # runs doctr recognizer. text detection with doctr text detection needs to be done before!
            config = dd.set_config_by_yaml(cfg_path)

            path_weights_tr = dd.ModelDownloadManager.maybe_download_weights_and_configs(config.WEIGHTS.DOCTR_RECOGNITION.PT)
            detector = dd.DoctrTextRecognizer("crnn_vgg16_bn", path_weights_tr, config.DEVICE, config.LIB)
            kwargs = {'extract_from_roi': 'word'}

        else:
            raise NotImplementedError

        super().__init__(detector, **kwargs)