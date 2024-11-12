<<<<<<< HEAD
from .pipeline import Pipeline
import matplotlib.pyplot as plt
import os
from PIL import Image

class LayoutDetector:

    def __init__(self, pipeline_config_path: str) -> None:
        
        self.pipeline = Pipeline.from_config(cfg_path=pipeline_config_path)
        self.pipeline.build()


    def detect(self, pdf_path):
        self.dps, self.page_dicts = self.pipeline.run(pdf_path)

        return self.dps, self.page_dicts

    def vis(self, save=False, save_path=None):

        assert self.dps

        for i, dp in enumerate(self.dps):
            image = dp.viz(show_words=False, show_tables=True)

            plt.figure(figsize = (25,17))
            plt.axis('off')
            #plt.imshow(image)

            if save:
                if save_path is None:
                    raise ValueError("save_path must be provided if save is True.")
                # Ensure the directory exists
                os.makedirs(save_path, exist_ok=True)

                image = Image.fromarray(image)
                image_file_path = os.path.join(save_path, f'layout_detections_{i}.png')
                image.save(image_file_path)
                #plt.savefig(image_file_path, bbox_inches='tight', pad_inches=0)
                #plt.close()  # Close the figure to free memory
            else:
                plt.imshow(image)
=======
from .pipeline import Pipeline
import matplotlib.pyplot as plt
import os

class LayoutDetector:

    def __init__(self, pipeline_config_path: str):
        """
        Initialize LayoutDetector
        
        Args:
            pipeline_config_path (str): Path to the pipeline config file
        """
        self.pipeline_config_path = pipeline_config_path

    def detect(self, pdf_path: str):
        from .pipeline import Pipeline
        
        # Use the config file directly instead of trying to join paths
        pipeline = Pipeline.from_config(self.pipeline_config_path)
        
        # Build the pipeline before running
        pipeline.build()
        
        # Run the pipeline
        dps, page_dicts = pipeline.run(pdf_path)
        
        return dps, page_dicts

    def vis(self):
        assert self.dps

        for dp in self.dps:
            image = dp.viz(show_words=False, show_tables=True)
            plt.figure(figsize = (25,17))
            plt.axis('off')
            plt.imshow(image)
>>>>>>> 8e73fb6 (feat: project restructuring and config management improvements)
