from langchain_community.document_loaders import PyMuPDFLoader, PyPDFLoader, DirectoryLoader


# load pdf documents from data folder
dir_loader = DirectoryLoader(
    "../data",
    glob="**/*.pdf",
    loader_cls=PyMuPDFLoader,
    show_progress=False
)

pdf_documents = dir_loader.load()
print(pdf_documents)
