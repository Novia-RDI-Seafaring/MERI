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