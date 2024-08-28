from .pipeline import Pipeline
import matplotlib.pyplot as plt

class LayoutDetector:

    def __init__(self, pipeline_config_path: str) -> None:
        
        self.pipeline = Pipeline.from_config(cfg_path=pipeline_config_path)
        self.pipeline.build()


    def detect(self, pdf_path):
        self.dps, self.page_dicts = self.pipeline.run(pdf_path)

        return self.dps, self.page_dicts

    def vis(self):
        assert self.dps

        for dp in self.dps:
            image = dp.viz(show_words=False, show_tables=True)
            plt.figure(figsize = (25,17))
            plt.axis('off')
            plt.imshow(image)
