from meri.meri import MERI

class StateManager():

    pdf_path: str = None
    schema_path: str = None
    meri: MERI = None
    populated_schema: str = None

    @classmethod
    def set_pdf_path(cls, pdf_path: str):
        cls.pdf_path = pdf_path

    @classmethod
    def set_schema_path(cls, schema_path: str):
        cls.schema_path = schema_path
    
    @classmethod
    def set_meri(cls, meri: MERI):
        cls.meri = meri
    
    @classmethod
    def set_populated_schema(cls, schema: str):
        cls.populated_schema = schema