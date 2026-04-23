from langchain_community.document_loaders import CSVLoader,JSONLoader,TextLoader,PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
loader = CSVLoader(
    file_path="./data/sample.csv", 
    encoding="utf-8",
    csv_args={"delimiter": ",",
              "quotechar": '"',}
    )

for doc in loader.lazy_load():
    print(doc)
    
    
jsloader = JSONLoader(
    file_path="./data/sample.json", 
    encoding="utf-8",
    jq_schema= ".",
    text_content= False
    ).load()

testloader = TextLoader(
    file_path="./data/sample.txt",  
    encoding="utf-8"
    ).load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=10, 
    chunk_overlap=0,
    separators=["\n\n", "\n", " ", ""],
    length_function=len
    )
texts = text_splitter.split_documents(testloader)




pyloader = PyPDFLoader(
    file_path="./data/sample.pdf",
    mode="page",
    encoding="utf-8"
    ).load()