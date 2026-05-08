# RAG-conversational


1. generar embeddings a partir de los docs y generar el retriever
2. design router and train
3. generator
4. evaluation


01_load_dataset.py → descarga SQAC y construye el corpus + split de evaluación
02_build_index.py → genera embeddings y construye el index FAISS
03_retriever.py → recuperación con FAISS + reranking
04_load_llm.py
05_router.py → router RAG basado en LLM
06_generator.py → generador LLM + self-feedback
07_pipeline.py → ejecución completa del RAG con el router 
08_evaluate.py → métricas: token F1, BERTScore, Recall@k, MRR, oracle. comparación: never / always / router RAG