from meri.utils.utils import pdf_to_im, pil_to_base64
import os
import aiofiles
import fitz

def get_pdf_images(pdf_path: str):
    pdf_pages = fitz.open(pdf_path)

    pdf_ims_base64 = []
    for page in pdf_pages:
        im = pdf_to_im(page)
        base_64_im = pil_to_base64(im, raw=False)
        pdf_ims_base64.append((im, base_64_im))

    return pdf_ims_base64

async def save_file(uploaded_file, path):

    os.makedirs(os.path.dirname(path), exist_ok=True)
    async with aiofiles.open(path, 'wb') as out_file:
        content = await uploaded_file.read()
        await out_file.write(content)
