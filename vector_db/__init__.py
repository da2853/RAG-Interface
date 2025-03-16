# vector_db/__init__.py
"""
Vector database interface package for ExamGPT application.
"""
from vector_db.qdrant_manager import QdrantManager

__all__ = ['QdrantManager']