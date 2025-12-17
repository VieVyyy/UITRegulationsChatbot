import os
import shutil
from flask import Flask, request, jsonify, render_template

# Import LangChain & các thư viện cần thiết
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Cấu hình Flask App
app = Flask(__name__)

# --- KHAI BÁO CÁC BIẾN TOÀN CỤC (GLOBAL VARIABLES) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FINE_TUNED_MODEL_PATH = os.path.join(BASE_DIR, "fine_tuned_model") 
CHROMA_DB_PATH = os.path.join(BASE_DIR, "chroma_db") 
CSV_FILE_PATH = os.path.join(BASE_DIR, "Data", "train.csv")

os.environ["CHROMA_TELEMETRY_ENABLED"] = "False"

# Biến lưu trữ RAG Chain đã khởi tạo
rag_chain = None
# Biến lưu trữ Vectorstore (Chroma)
vectorstore = None
final_retriever = None

# Hàm tạo Embedding Model
def getEmbeddingModel():
    # Model Fine-tuned hoặc model mặc định
    return HuggingFaceEmbeddings(
        model_name=FINE_TUNED_MODEL_PATH,
        model_kwargs={'device': 'cpu'}, # CHẮC CHẮN dùng 'cpu' nếu không có GPU mạnh
        encode_kwargs={'normalize_embeddings': True}
    )

# Hàm khởi tạo ChromaDB (Được gọi nếu DB chưa tồn tại)
def buildVectorDB(csv_path):
    print("--- BẮT ĐẦU XÂY DỰNG DATABASE TỪ CSV ---")
    if not os.path.exists(FINE_TUNED_MODEL_PATH):
        raise FileNotFoundError(f"Không tìm thấy model tại {FINE_TUNED_MODEL_PATH}. Hãy tải model về!")
    
    # 1. Đọc và tạo Document
    df = pd.read_csv(csv_path).dropna(subset=['context', 'article'])
    docs = [Document(page_content=row['context'], metadata={"source": row.get('document', 'N/A'), "article": row.get('article', 'N/A')}) for _, row in df.iterrows()]

    # 2. Xóa DB cũ và Chunking
    if os.path.exists(CHROMA_DB_PATH):
        shutil.rmtree(CHROMA_DB_PATH)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    print(f"Đã chia thành {len(splits)} chunks.")

    # 3. Tạo Vector Store
    embedding_model = getEmbeddingModel()
    print("Đang tạo Vector Store (ChromaDB)...")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding_model,
        persist_directory=CHROMA_DB_PATH
    )
    vectorstore.persist()
    print("✅ Đã xây dựng xong Vector Database!")
    return vectorstore, docs

# Hàm thiết lập RAG Pipeline
def setupRAGPipeline(csv_path):
    global vectorstore, rag_chain
    
    # 1. LOAD/BUILD CHROMA DB
    docs = None # documents gốc (dùng cho BM25)
    if not os.path.exists(CHROMA_DB_PATH):
        print("Chưa thấy Database, đang tạo mới...")
        vectorstore, docs = buildVectorDB(csv_path)
    else:
        print("Đã tìm thấy Database, đang kết nối lại...")
        embedding_model = getEmbeddingModel()
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embedding_model
        )
        # Nếu đã có DB thì cần đọc lại file CSV để có documents gốc cho BM25
        df = pd.read_csv(csv_path).dropna(subset=['context', 'article'])
        docs = [Document(page_content=row['context'], metadata={"source": row.get('document', 'N/A'), "article": row.get('article', 'N/A')}) for _, row in df.iterrows()]
        
    # 2. THIẾT LẬP RETRIEVER (Ensemble + Reranker)
    # BM25 Retriever
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 30 
    
    # Chroma Retriever
    chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
    
    # Ensemble Retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever],
        weights=[0.5, 0.5]
    )
    
    # Re-ranker
    print("Đang tải model Re-ranking...")
    cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    compressor = CrossEncoderReranker(model=cross_encoder, top_n=3)
    final_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever
    )

    # 3. THIẾT LẬP LLM & PROMPT
    # Đảm bảo biến môi trường GOOGLE_API_KEY đã được thiết lập
    if "GOOGLE_API_KEY" not in os.environ:
        raise ValueError("Lỗi: GOOGLE_API_KEY chưa được thiết lập!")
        
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash-lite", temperature=0.3)

    template = """Bạn là trợ lý ảo hỗ trợ sinh viên trường đại học.
Hãy trả lời câu hỏi dựa trên các thông tin ngữ cảnh được cung cấp dưới đây.
Nếu không tìm thấy thông tin trong ngữ cảnh, hãy nói "Tôi chưa tìm thấy thông tin này trong văn bản quy định".

NGỮ CẢNH:
{context}

CÂU HỎI: {question}

CÂU TRẢ LỜI:"""
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join([f"--- Nguồn: {d.metadata.get('article', 'N/A')} ---\n{d.page_content}" for d in docs])

    # 4. Xây dựng Chain
    rag_chain = (
        {"context": final_retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("✅ RAG Pipeline đã sẵn sàng!")
    return rag_chain, final_retriever

# --- CÁC ROUTE CỦA FLASK ---

@app.route("/")
def home():
    """Route cho trang chủ (render giao diện HTML)"""
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    """Route xử lý câu hỏi từ người dùng"""
    global rag_chain, final_retriever
    if rag_chain is None or final_retriever is None:
        return jsonify({"response": "Lỗi: Hệ thống chưa được khởi tạo.", "source": "N/A"}), 500
        
    data = request.json
    user_query = data.get("query", "")
    
    if not user_query:
        return jsonify({"response": "Vui lòng nhập câu hỏi.", "source": "N/A"})

    try:
        # --- BƯỚC 1: GỌI RETRIEVER ĐỘC LẬP ---
        # Sử dụng retriever đã tách ra để tìm kiếm documents
        retrieved_docs = final_retriever.invoke(user_query)
        
        # Ghi lại nguồn tài liệu
        source_info = "; ".join(list(set([d.metadata.get('article', 'N/A') for d in retrieved_docs])))

        # --- BƯỚC 2: GỌI RAG CHAIN CHÍNH THỨC ---
        # RAG Chain sẽ tự động gọi lại retriever này (hoặc bạn có thể tối ưu hơn)
        # Vì rag_chain đã được xây dựng bằng final_retriever, chúng ta chỉ cần invoke nó
        response = rag_chain.invoke(user_query)
        
        return jsonify({
            "response": response,
            "source": source_info # Trả về thông tin nguồn cho frontend
        })
        
    except Exception as e:
        print(f"Lỗi trong quá trình xử lý: {e}")
        return jsonify({"response": f"Đã xảy ra lỗi hệ thống: {e}", "source": "N/A"}), 500
    
# --- KHỞI TẠO HỆ THỐNG TRƯỚC KHI CHẠY FLASK ---
if __name__ == "__main__":
    
    try:
        # Chắc chắn rằng API Key đã được thiết lập trước khi chạy setupRAGPipeline
        # Vì nếu không có Key, ChatGoogleGenerativeAI sẽ báo lỗi.
        if "GOOGLE_API_KEY" not in os.environ:
            print("❌ Lỗi: Vui lòng thiết lập biến môi trường GOOGLE_API_KEY!")
        else:
            # <-- Khai báo lại global
            rag_chain, final_retriever = setupRAGPipeline(CSV_FILE_PATH)
            # Chạy Flask Server
            # host='0.0.0.0' để truy cập từ mạng ngoài, debug=True để tự động reload
            app.run(host='0.0.0.0', port=5000, debug=True)
            
    except Exception as e:
        print(f"LỖI KHỞI TẠO HỆ THỐNG: {e}")
        print("Vui lòng kiểm tra lại: 1. API Key, 2. Đường dẫn model, 3. Thư viện đã cài.")