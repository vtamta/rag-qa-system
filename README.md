# RAG-Based Question Answering System

A Retrieval-Augmented Generation (RAG) system built with LLaMA 2, LangChain, and FAISS for intelligent document question answering with an interactive Gradio interface.

## 📋 Overview

This project implements a complete RAG pipeline that enables users to:
- Upload PDF documents
- Ask natural language questions about the content
- Receive accurate, context-aware answers powered by LLaMA 2 7B
- Interact through an intuitive Gradio web interface

The system combines **semantic search** with **large language model generation** to provide answers grounded in your documents, reducing hallucinations and improving accuracy.

## 🎯 Key Features

- **PDF Processing**: Automatic text extraction and intelligent chunking
- **Vector Embeddings**: HuggingFace sentence transformers for semantic understanding
- **FAISS Vector Store**: Fast similarity search for relevant context retrieval
- **LLaMA 2 7B Chat**: Meta's powerful language model with 4-bit quantization
- **Streaming Responses**: Real-time token-by-token answer generation
- **Interactive UI**: Clean, user-friendly Gradio interface
- **Context-Aware Answers**: Retrieves relevant passages before generating responses

## 🛠️ Technology Stack

### Core Libraries
- **LangChain** - RAG pipeline orchestration and document processing
- **Transformers (HuggingFace)** - Model loading and inference
- **PyTorch** - Deep learning framework
- **FAISS** - Fast vector similarity search
- **Gradio** - Interactive web interface
- **BitsAndBytes** - 4-bit model quantization for efficiency

### Models & Components
- **LLaMA 2 7B Chat** (`meta-llama/Llama-2-7b-chat-hf`)
- **HuggingFace Sentence Transformers** - Document embeddings
- **RecursiveCharacterTextSplitter** - Intelligent text chunking
- **TextIteratorStreamer** - Real-time response streaming

## 📊 System Architecture

```
PDF Document
     ↓
PyPDFLoader (Extract Text)
     ↓
Text Splitter (Create Chunks)
     ↓
Embeddings (Convert to Vectors)
     ↓
FAISS Index (Store Vectors)
     ↓
User Question → Similarity Search → Retrieve Context
     ↓
Context + Question → LLaMA 2 Generation
     ↓
Streamed Answer (Gradio UI)
```

## 🚀 Getting Started

### Prerequisites

- **Python 3.8+**
- **GPU**: CUDA-capable GPU recommended (12GB+ VRAM ideal)
- **HuggingFace Account**: Required for LLaMA 2 access

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/rag-qa-system.git
   cd rag-qa-system
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Get LLaMA 2 Access**:
   - Request access at [Meta AI - LLaMA](https://ai.meta.com/llama/)
   - Accept terms on [HuggingFace LLaMA 2 page](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
   - Create API token at [HuggingFace Settings](https://huggingface.co/settings/tokens)

### Running the Application

1. **Open Jupyter Notebook**:
   ```bash
   jupyter notebook rag_qa_system.ipynb
   ```

2. **Authenticate with HuggingFace**:
   
   When you run the notebook, it will prompt for your token:
   ```python
   from huggingface_hub import login
   login()  # Enter your token when prompted
   ```

3. **Run all cells** to:
   - Load the LLaMA 2 model (with 4-bit quantization)
   - Initialize the RAG pipeline
   - Launch the Gradio interface

4. **Use the web interface**:
   - Upload a PDF document
   - Type your question in the chat box
   - Receive AI-generated answers with relevant context

## 💡 How It Works

### 1. Document Processing
The system loads PDF documents and splits them into manageable chunks for processing.

### 2. Vector Store Creation
Text chunks are converted into vector embeddings and stored in a FAISS index for fast similarity search.

### 3. Question Answering
When you ask a question:
1. Your question is converted to a vector embedding
2. The system finds the most relevant document chunks
3. Retrieved context is combined with your question
4. LLaMA 2 generates a comprehensive answer
5. The answer is streamed to the Gradio interface in real-time

## 📁 Repository Structure

```
rag-qa-system/
├── README.md                    # Project documentation (this file)
├── requirements.txt             # Python dependencies
├── rag_qa_system.ipynb         # Main implementation notebook
└── .gitignore                   # Git ignore rules
```

## ⚙️ Configuration

### Model Settings
- **Model**: LLaMA 2 7B Chat (`meta-llama/Llama-2-7b-chat-hf`)
- **Quantization**: 4-bit (NF4) with double quantization
- **Compute Type**: bfloat16

### RAG Parameters
- **Chunk Size**: 1000 characters
- **Chunk Overlap**: 200 characters
- **Top-K Retrieval**: 3-5 most relevant chunks
- **Temperature**: 0.7 (adjustable for creativity)

### Hardware Requirements

| Configuration | RAM | GPU VRAM | Performance |
|--------------|-----|----------|-------------|
| Minimum | 16GB | 8GB | Functional |
| Recommended | 32GB | 12GB+ | Optimal |
| CPU-only | 32GB+ | N/A | Very slow |

## 🎓 Use Cases

- **Academic Research**: Query research papers and academic documents
- **Legal Analysis**: Extract information from contracts and legal documents
- **Technical Documentation**: Navigate complex technical manuals
- **Business Intelligence**: Analyze reports and business documents
- **Education**: Interactive learning from textbooks and study materials
- **Customer Support**: Build knowledge base Q&A systems

## 🔒 Security & Privacy

### Important Notes

1. **API Token Management**: 
   - Never commit your HuggingFace token to GitHub
   - Use environment variables or interactive login
   - The notebook uses `login()` which prompts for token securely

2. **Data Privacy**:
   - Documents are processed locally
   - No data sent to external servers (except HuggingFace for model download)
   - Vector store remains on your machine

3. **Model License**:
   - LLaMA 2 has specific usage terms
   - Review [Meta's LLaMA 2 license](https://ai.meta.com/llama/license/)
   - Commercial use allowed under certain conditions

## 📈 Performance Tips

### Optimization Strategies

1. **GPU Utilization**: Ensure CUDA is properly installed
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Memory Management**: Use 4-bit quantization (already configured)

3. **Batch Processing**: Process multiple documents before querying

4. **Caching**: Save and load vector stores to avoid reprocessing

## 🐛 Troubleshooting

### Common Issues

**Issue**: CUDA out of memory
- **Solution**: Use smaller chunk sizes or reduce batch size

**Issue**: Model download fails
- **Solution**: Check LLaMA 2 access approval on HuggingFace

**Issue**: Slow generation
- **Solution**: Verify GPU is being used, check CUDA installation

**Issue**: Empty responses
- **Solution**: Ensure document was properly loaded and chunked

## 🤝 Contributing

This project is open for improvements! Areas for contribution:
- Support for more document formats (DOCX, TXT, HTML)
- Multi-document querying
- Conversation history/memory
- Custom embedding models
- API deployment

## 📄 License

This project is for educational and research purposes. 

**Important**: LLaMA 2 usage is subject to Meta's license agreement. Please review the license before commercial use.

## 🙏 Acknowledgments

- **Meta AI** - LLaMA 2 language model
- **HuggingFace** - Transformers library and model hosting
- **LangChain** - RAG framework and tools
- **Facebook AI** - FAISS vector search
- **Gradio** - Web interface framework

## 📚 Resources & Further Reading

- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [LLaMA 2 Paper](https://arxiv.org/abs/2307.09288)
- [RAG Explained](https://www.pinecone.io/learn/retrieval-augmented-generation/)
- [FAISS Documentation](https://faiss.ai/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)

## ✉️ Contact

**Vaibhav Tamta**
- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com

---

*Built with ❤️ using LLaMA 2, LangChain, and open-source AI tools*
