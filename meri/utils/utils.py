import base64
from PIL import Image
import fitz
import io

def merge_bboxes(boxes):

    x0 = min(box[0] for box in boxes)
    y0 = min(box[1] for box in boxes)
    x1 = max(box[2] for box in boxes)
    y1 = max(box[3] for box in boxes)

    return [x0, y0, x1, y1]

def pdf_to_im(page: fitz.Page, cropbbox=None):
    """ Converts pdf to image and if provided crops image by cropbox
    """

    pix = page.get_pixmap()
    pil_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    if cropbbox is None:
        return pil_image
    cropped_im = pil_image.crop(cropbbox)
    return cropped_im 
    
def scale_coords(source_coords, source_height, source_width, target_height, target_width):
    '''Transforms source coordinates (x0, y0, x1, y1)
    to target coordinates (x0,y0, x1,y1)'''

    x0, y0, x1, y1 = source_coords

    x0_rel = x0/source_width
    x1_rel = x1/source_width

    y0_rel = y0/source_height
    y1_rel = y1/source_height

    #rect_shape = [int(x0_rel*target_width+0.5),int(y0_rel*target_height+0.5), int(x1_rel*target_width+0.5), int(y1_rel*target_height+0.5)]
    rect_shape = [int(x0_rel*target_width),int(y0_rel*target_height), int(x1_rel*target_width), int(y1_rel*target_height)]

    return rect_shape


def pil_to_base64(pil_image: Image):
    """ Converts PIL to base64 string
    """
    # Convert PIL Image to bytes
    with io.BytesIO() as buffer:
        pil_image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()

    # Convert bytes to base64 string
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    return base64_image