from .pipeline import Pipeline
import matplotlib.pyplot as plt
import os
from PIL import Image

class LayoutDetector:
    def __init__(self, pipeline_config_path: str) -> None:
        """
        Initialize LayoutDetector
        
        Args:
            pipeline_config_path (str): Path to the pipeline config file
        """
        self.pipeline_config_path = pipeline_config_path
        self.pipeline = Pipeline.from_config(cfg_path=pipeline_config_path)
        self.pipeline.build()
        self.dps = None

    def detect(self, pdf_path: str):
        """
        Detect layout in PDF
        
        Args:
            pdf_path (str): Path to PDF file
        
        Returns:
            tuple: (dps, page_dicts) containing detection pages and processed dictionaries
        """
        # Run the pipeline
        dps, page_dicts = self.pipeline.run(pdf_path)
        self.dps = dps
        
        return dps, page_dicts

    def vis(self, save=False, save_path=None):
        """
        Visualize detected layout
        
        Args:
            save (bool): Whether to save visualization
            save_path (str): Path to save visualization if save=True
        """
        assert self.dps, "No detection results available. Run detect() first."

        for i, dp in enumerate(self.dps):
            image = dp.viz(show_words=False, show_tables=True)
            plt.figure(figsize=(25,17))
            plt.axis('off')
            plt.imshow(image)

            if save:
                if save_path is None:
                    raise ValueError("save_path must be provided if save is True.")
                # Ensure the directory exists
                os.makedirs(save_path, exist_ok=True)
                image = Image.fromarray(image)
                image.save(os.path.join(save_path, f"page_{i}.png"))
