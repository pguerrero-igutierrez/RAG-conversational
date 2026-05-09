# RAG-conversational

01_load_dataset.py → descarga SQAC y construye el corpus + split de evaluación

02_build_index.py → genera embeddings y construye el index FAISS

03_retriever.py → recuperación con FAISS + reranking

04_load_llm.py

05_router.py → router RAG basado en LLM

06_generator.py → generador LLM + self-feedback

07_pipeline.py → ejecución completa del RAG con el router 

08_evaluate.py → métricas: token F1, BERTScore, Recall@k, MRR, oracle. comparación: never / always / router RAG


## Project Structure

The pipeline is divided into modular scripts:

* `load_dataset.py`: Parses and deduplicates raw SQAC JSON files into a flattened `.jsonl` corpus and evaluation set.
* `build_index.py`: Builds the BM25 index (with stopword removal and stemming) and generates normalized dense embeddings.
* `retriever.py`: Executes the hybrid retrieval + reranking pipeline.
* `load_llm.py`: Configures and loads the 4-bit quantized Mixtral model.
* `router.py`: Trains and executes the few-shot SetFit model for intent classification.
* `generator.py`: Handles prompt construction, generation, and the self-feedback verification loop.
* `main.py`: The main orchestrator connecting routing, retrieval, and generation.
* `evaluate.py`: Calculates quantitative metrics (MRR, Recall@k, Token F1, and BERTScore) comparing Always-RAG, Never-RAG, and Router-RAG strategies.