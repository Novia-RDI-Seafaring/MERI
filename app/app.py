from fasthtml.common import *
import os
import aiofiles
import fitz
from statemanager import StateManager
from meri.meri import MERI
from utils import *
import json
import argparse


# Set up argument parsing
parser = argparse.ArgumentParser(description="Run the MERI demo.")
parser.add_argument('--model', type=str, default='gpt-4o-mini', help='LLM to use')
args = parser.parse_args()

tlink = Script(src="https://cdn.tailwindcss.com"),
dlink = Link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/daisyui@4.11.1/dist/full.min.css")
int_format_link = Link(rel="stylesheet", href="intermediate_format.css")

public_path = os.path.join(os.path.abspath(os.path.join(__file__, os.pardir, "public")))

app = FastHTML(hdrs=(tlink, dlink, int_format_link), static_path=public_path)

upload_dir = Path("uploads")
upload_dir.mkdir(exist_ok=True)

# Define a route to serve static files
@app.route("/{fname:path}.{ext:static}")
async def get(fname: str, ext: str):
    return FileResponse(f'{public_path}/{fname}.{ext}')
    #return FileResponse(f'public/{fname}.{ext}')

def submit_button(text: str):
    return Button(text, type="submit", cls="btn btn-neutral"),

def action_button(text: str, *args, **kwargs):

    return Div(
        Button(text, *args, **kwargs),
        cls="flex w-full justify-center"
    )

def carossel_item(id, base_64_str):
    return Div(Img(src=base_64_str, cls="w-fill"), id=id, cls="carousel-item w-full")

def json_collapse(summary: str, json_dict):
    return Details(
            Summary(summary, cls="collapse-title text-m font-medium"),
                Div(
                    Pre(
                    json.dumps(json_dict, indent=4),
                    cls="language-json whitespace-pre-wrap font-mono bg-gray-100 p-4 rounded-lg max-h-[50vh] overflow-y-auto"
                ),
                cls="collapse-content", style="padding: 1rem", id = "schema_panel"),
            cls="collapse bg-base-200")



def header():
    
    return Title('MERI demo'), Div(
        Img(src="meri_logo.svg", cls="object-scale-down h-10"),
            #H1("MERI", cls="text-3xl font-bold text-center text-black"),
        cls="navbar bg-base-300 rounded-2xl shadow-lg p-2")

def pdf_panel():
        return Div(
            Div(
                Form(
                    Fieldset(
                        Input(name="pdf_file", type="file", id="select-pdf", cls="file-input file-input-bordered", hx_sync="closest form:abort"),
                        submit_button("Upload") 
                    ),
                    cls="join", hx_post="/upload_pdf", hx_swap="innerHTML", target_id="pdf_panel"
                ),
                cls="flex items-center justify-center h-full"
                #Div(
                #    cls="w-full", id="pdf_display"
                #)
            ),
            id="pdf_panel",
            cls="w-1/2 p-4")

def int_format_collapse():
     
     return Details(
            Summary("Create Intermediate Format", cls="collapse-title text-xl font-medium"),
            Div(
                Div(
                    P("Document ins transferred into an intermediate format."),
                    Div(action_button("To Intermediate", hx_post="/to_intermediate", hx_swap="innerHTML", target_id="int_format",
                        cls="btn btn-primary mt-4"),
                        id="int_format"
                    ),
                    cls="flex flex-col space-y-4 max-w-full"
                ),
                cls="collapse-content", id="extraction_panel"
            ),
            cls="collapse bg-base-200")

def target_schema_form():

    return Div(
                Form(
                    Fieldset(
                        Input(name="target_schema", type="file", id="select-schema", cls="file-input file-input-bordered", hx_sync="closest form:abort"),
                        submit_button("Upload")
                    ),
                    cls="join", hx_post="/upload_schema", hx_swap="innerHTML", target_id="schema_panel"
                ), cls="flex items-center justify-center h-full", id="schema_panel"
            )

def extract_collapse():
     return Details(
            Summary("Extract parameters", cls="collapse-title text-xl font-medium"),
            Div(
                Div(
                    P("Extract parameters from intermediate format given uploaded json schema."),
                    target_schema_form(),
                    Div(action_button("Extract Parameters", hx_post="/extract_parameters", hx_swap="innerHTML", target_id="extract_btn",
                        cls="btn btn-primary mt-4", id="extract-btn"),
                        id="extract_btn"
                    ),
                    cls="flex flex-col space-y-4 max-w-full"
                ),
                cls="collapse-content", id="extraction_panel"
            ),
            cls="collapse bg-base-200")

def text_paragraph():
    return Div(P("""In a first step the document is converted into an intermediate format. In a second step the intermediate
                 format is used to extract the specific parameters specified through a json schema."""))


def action_panel():

    return Div(
        text_paragraph(),
        int_format_collapse(),
        extract_collapse(),
        
        cls="w-1/2 p-4 space-y-4")



@app.post("/extract_parameters")
async def extract_parameters():
    
    with open(StateManager().schema_path) as f:
        json_schema = json.load(f)

    populated_schema = StateManager().meri.run(json.dumps(json_schema))
    StateManager().set_populated_schema(json.dumps(populated_schema))

    return json_collapse("Show populated schema", populated_schema)

@app.post("/upload_schema")
async def upload_schema(target_schema: File, request: Request = None):
    form = await request.form()

    # Save uploaded files
    target_schema_f_name = os.path.join(upload_dir, target_schema.filename)
    await save_file(target_schema, target_schema_f_name)

    StateManager().set_schema_path(target_schema_f_name)

    with open(target_schema_f_name) as f:
        json_data = json.load(f)

    return json_collapse("Show schema", json_data)

@app.post("/to_intermediate")
async def to_intermediate():

    meri = MERI(StateManager().pdf_path, model=args.model)
    meri.to_intermediate()

    StateManager().set_meri(meri)
    return Div(NotStr(meri.int_format), id="int_format")
        

@app.post('/upload_pdf')
async def upload_pdf(pdf_file: File, request: Request = None):
    form = await request.form()

    # Save uploaded files
    pdf_f_name = os.path.join(upload_dir, pdf_file.filename)
    await save_file(pdf_file, pdf_f_name)

    StateManager().set_pdf_path(pdf_f_name)

    carossel_items = [] 
    for i, (im, base_64_str) in enumerate(get_pdf_images(StateManager().pdf_path)):
        carossel_items.append(carossel_item(id=f"item{i}", base_64_str=base_64_str))
        
    return Div(Div(*carossel_items, cls="carousel w-full"), 
               Div(*[A(str(i+1), href=f"#item{i}", cls="btn btn-xs") for i in range(len(carossel_items))], cls="flex w-full justify-center gap-2 py-2"),
                cls="w-full")
   


@app.route('/')
def get():

    return Div(
        header(), 
        Div(pdf_panel(), Div(cls="divider divider-horizontal pt-4 pb-4"), action_panel(), cls="flex h-screen"),
        cls="flex flex-col h-screen", data_theme="pastel"
    )
       


serve(port=5010)