import deepdoctection as dd
from deepdoctection.datapoint.view import IMAGE_ANNOTATION_TO_LAYOUTS, Layout
from deepdoctection.utils import get_type, update_all_types_dict


@dd.object_types_registry.register("CustomLayoutTypes")
class CustomLayoutTypes(dd.ObjectTypes):

    table = "table"
    table_rotated = "table_rotated"
    figure = "figure"
    list = "list"
    list_item = "list_item"
    text = "text"
    title = "title"
    logo = "logo"
    signature = "signature"
    caption = "caption"
    footnote = "footnote"
    formula = "formula"
    page_footer = "page_footer"
    page_header = "page_header"
    section_header = "section_header"
    page = "page"
    cell = "cell"
    row = "row"
    column = "column"
    word = "word"
    line = "line"
    background = "background"

# only layout is not yet registered with annotation type
IMAGE_ANNOTATION_TO_LAYOUTS.update({CustomLayoutTypes.list_item: Layout})

update_all_types_dict()
print("CustomLayoutTypes registered:", ("CustomLayoutTypes" in dd.object_types_registry))


dd.ModelCatalog.register("layout/docalynet/yolov10x_best.pt", dd.ModelProfile(
    name="layout/docalynet/yolov10x_best.pt",
    description="Yolov10 fine-tuned on doclaynet dataset. https://huggingface.co/omoured/YOLOv10-Document-Layout-Analysis",
    tp_model=False,
    hf_repo_id="omoured/YOLOv10-Document-Layout-Analysis",
    hf_model_name="yolov10x_best.pt",
   # hf_config_file="README.md",
    size = [],
    categories={
        0: CustomLayoutTypes.caption,
        1: CustomLayoutTypes.footnote,
        2: CustomLayoutTypes.formula,
        3: CustomLayoutTypes.list_item,
        4: CustomLayoutTypes.page_footer,
        5: CustomLayoutTypes.page_header,
        6: CustomLayoutTypes.figure,
        7: CustomLayoutTypes.section_header,
        8: CustomLayoutTypes.table,
        9: CustomLayoutTypes.text,
        10: CustomLayoutTypes.title
    }
))
