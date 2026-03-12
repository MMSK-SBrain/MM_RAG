"""
Core Multimodal RAG pipeline.

Handles PDF parsing (text, tables, images), embedding via IBM Granite,
FAISS vector indexing, and LangChain RetrievalQA for querying.

LLM access is via OpenRouter (OpenAI-compatible API).
"""

import base64
import io
import logging
import os
from pathlib import Path

from langchain_core.documents import Document

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

GROUNDING_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions
based strictly on the provided context. If the context does not contain enough
information to answer the question, say "I don't have enough information to
answer that question based on the available documents."

CONTEXT (from retrieved documents — treat as the ONLY source of truth):
{context}

QUESTION: {question}

ANSWER:"""


class RAGPipeline:
    """Manages the full multimodal RAG lifecycle."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._mock_doc_count = 0

        if self.settings.mock_mode:
            logger.info("Running in MOCK MODE — no models loaded")
            self.vectorstore = None
            self.qa_chain = None
            return

        # Real mode imports (heavy dependencies)
        from langchain.prompts import PromptTemplate
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_openai import ChatOpenAI
        from openai import OpenAI

        # Embeddings (local model)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.settings.embedding_model_name
        )

        # Load existing FAISS index if available
        index_path = self.settings.faiss_index_path
        if os.path.exists(index_path):
            self.vectorstore = FAISS.load_local(
                index_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info("Loaded FAISS index from %s", index_path)
        else:
            self.vectorstore = None
            logger.info("No existing FAISS index found")

        # LLM via OpenRouter (OpenAI-compatible)
        self.llm = ChatOpenAI(
            model=self.settings.granite_instruct_model,
            openai_api_key=self.settings.openrouter_api_key,
            openai_api_base=self.settings.openrouter_base_url,
            max_tokens=1000,
            temperature=0.1,
        )

        # Vision model client (OpenAI SDK directly for multimodal)
        self.vision_client = OpenAI(
            api_key=self.settings.openrouter_api_key,
            base_url=self.settings.openrouter_base_url,
        )

        # Docling converter (hoisted to instance level)
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        pdf_opts = PdfPipelineOptions(do_ocr=False, generate_picture_images=True)
        fmt_opts = {InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)}
        self.converter = DocumentConverter(format_options=fmt_opts)

        # Build QA chain if index exists
        self._prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=GROUNDING_PROMPT_TEMPLATE,
        )
        self.qa_chain = self._build_qa_chain() if self.vectorstore else None

    def _build_qa_chain(self):
        """Build or rebuild the RetrievalQA chain."""
        from langchain.chains import RetrievalQA

        if not self.vectorstore:
            return None

        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.settings.retriever_top_k}
        )
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": self._prompt_template},
            return_source_documents=True,
        )

    def parse_pdf(self, pdf_path: str) -> list[Document]:
        """
        Parse a single PDF into Document chunks using Docling.

        Returns a list of LangChain Document objects with three types:
        1. Text chunks — via HybridChunker
        2. Table chunks — extracted as Markdown
        3. Image chunks — image summarised by Granite Vision,
                          summary text becomes the chunk content
        """
        if self.settings.mock_mode:
            return [
                Document(
                    page_content="Mock text chunk content from the document.",
                    metadata={
                        "source": Path(pdf_path).name,
                        "chunk_type": "text",
                        "page": 1,
                    },
                ),
                Document(
                    page_content="| Col A | Col B |\n|---|---|\n| 1 | 2 |",
                    metadata={
                        "source": Path(pdf_path).name,
                        "chunk_type": "table",
                        "page": 2,
                    },
                ),
                Document(
                    page_content="Mock image summary: diagram showing system architecture.",
                    metadata={
                        "source": Path(pdf_path).name,
                        "chunk_type": "image",
                        "page": 3,
                    },
                ),
            ]

        from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
        from docling_core.types.doc.document import TableItem
        from transformers import AutoTokenizer

        doc = self.converter.convert(source=str(pdf_path)).document
        tok = AutoTokenizer.from_pretrained(self.settings.embedding_model_name)
        chunks = []
        filename = Path(pdf_path).name

        # 1) Text chunks
        for ch in HybridChunker(tokenizer=tok).chunk(doc):
            items = ch.meta.doc_items
            if len(items) == 1 and isinstance(items[0], TableItem):
                continue
            page_no = items[0].prov[0].page_no if items and items[0].prov else 0
            chunks.append(
                Document(
                    page_content=ch.text,
                    metadata={"source": filename, "chunk_type": "text", "page": page_no},
                )
            )

        # 2) Tables
        for table_idx, tbl in enumerate(doc.tables, start=1):
            page_no = tbl.prov[0].page_no if tbl.prov else 0
            md = tbl.export_to_markdown(doc=doc)
            chunks.append(
                Document(
                    page_content=md,
                    metadata={
                        "source": filename,
                        "chunk_type": "table",
                        "page": page_no,
                        "table_num": table_idx,
                    },
                )
            )

        # 3) Images — summarize with vision model via OpenRouter
        for image_idx, pic in enumerate(doc.pictures, start=1):
            page_no = pic.prov[0].page_no if pic.prov else 0
            img = pic.get_image(doc)
            if img is None:
                continue

            buf = io.BytesIO()
            img.convert("RGB").save(buf, "PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            data_uri = f"data:image/png;base64,{b64}"

            caption = pic.caption_text(doc) if hasattr(pic, "caption_text") else ""

            try:
                response = self.vision_client.chat.completions.create(
                    model=self.settings.granite_vision_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": data_uri}},
                                {
                                    "type": "text",
                                    "text": f"Caption: {caption}\nPlease analyze this image "
                                    "and summarize in 3 sentences.",
                                },
                            ],
                        }
                    ],
                    max_tokens=200,
                )
                summary = response.choices[0].message.content
            except Exception as e:
                logger.warning("Vision model failed for image %d: %s", image_idx, e)
                summary = f"Image on page {page_no} (vision summary unavailable)"

            chunks.append(
                Document(
                    page_content=summary,
                    metadata={
                        "source": filename,
                        "chunk_type": "image",
                        "page": page_no,
                        "image_num": image_idx,
                    },
                )
            )

        return chunks

    def ingest_documents(self, pdf_paths: list[str]) -> int:
        """
        Parse multiple PDFs and add all chunks to the FAISS index.

        Returns the number of chunks added.
        Persists the updated index to disk.
        """
        if self.settings.mock_mode:
            count = sum(3 for _ in pdf_paths)
            self._mock_doc_count += count
            return count

        from langchain_community.vectorstores import FAISS

        all_chunks = []
        for path in pdf_paths:
            chunks = self.parse_pdf(path)
            all_chunks.extend(chunks)
            logger.info("Parsed %s: %d chunks", path, len(chunks))

        if not all_chunks:
            return 0

        texts = [d.page_content for d in all_chunks]
        metas = [d.metadata for d in all_chunks]

        if self.vectorstore is None:
            self.vectorstore = FAISS.from_texts(texts, self.embeddings, metadatas=metas)
        else:
            self.vectorstore.add_texts(texts, metadatas=metas)

        self.vectorstore.save_local(self.settings.faiss_index_path)
        self.qa_chain = self._build_qa_chain()

        return len(all_chunks)

    def query(self, question: str) -> dict:
        """
        Run a query through the RetrievalQA chain.

        Returns dict with "answer" and "sources" keys.
        """
        if self.settings.mock_mode:
            return {
                "answer": (
                    "Based on the retrieved documents, the key finding is that "
                    "multimodal RAG systems improve accuracy by 35% when combining "
                    "text, table, and image data. "
                    "[MOCK RESPONSE — enable real mode by setting MOCK_MODE=false]"
                ),
                "sources": [
                    {
                        "content": "Mock text chunk content...",
                        "metadata": {
                            "source": "sample.pdf",
                            "chunk_type": "text",
                            "page": 1,
                        },
                    },
                    {
                        "content": "Mock table chunk content...",
                        "metadata": {
                            "source": "sample.pdf",
                            "chunk_type": "table",
                            "page": 3,
                        },
                    },
                ],
            }

        if not self.qa_chain:
            raise ValueError("No documents ingested. Use /ingest first.")

        result = self.qa_chain.invoke({"query": question})
        sources = []
        for doc in result.get("source_documents", []):
            sources.append(
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                }
            )

        return {
            "answer": result.get("result", ""),
            "sources": sources,
        }

    @property
    def document_count(self) -> int:
        """Return the number of documents/chunks in the FAISS index."""
        if self.settings.mock_mode:
            return self._mock_doc_count
        if self.vectorstore is None:
            return 0
        return self.vectorstore.index.ntotal

    @property
    def is_index_loaded(self) -> bool:
        """Return True if a FAISS index is loaded in memory."""
        if self.settings.mock_mode:
            return self._mock_doc_count > 0
        return self.vectorstore is not None
