import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Database configuration
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
    COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "mental_health_advisor")
    
    # Embedding Model - Optimized for Vietnamese + Mental Health Terms
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
    
    # Chunking Strategy - Optimized for Medical/Psychological Content
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
    
    # Retrieval settings - Tuned for Mental Health Consultation
    TOP_K_DOCUMENTS = int(os.getenv("TOP_K_DOCUMENTS", "3"))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.65"))
    
    # Advanced chunking options
    CHUNK_STRATEGY = os.getenv("CHUNK_STRATEGY", "recursive")
    OVERLAP_METHOD = os.getenv("OVERLAP_METHOD", "sentence")
    
    # Embedding optimization
    NORMALIZE_EMBEDDINGS = os.getenv("NORMALIZE_EMBEDDINGS", "true").lower() == "true"
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "8"))  # Smaller for Vietnamese model
    
    # Domain-specific settings
    DOMAIN = "mental_health_advisor"
    ENABLE_CRISIS_DETECTION = os.getenv("ENABLE_CRISIS_DETECTION", "true").lower() == "true"

# Model recommendations by use case
EMBEDDING_RECOMMENDATIONS = {
    "vietnamese_optimal": {
        "model": "dangvantuan/vietnamese-embedding",
        "dimension": 768,
        "description": "Tối ưu cho tiếng Việt - chuyên biệt cho Vietnamese content",
        "chunk_size": 800,
        "chunk_overlap": 150,
        "top_k": 3,
        "threshold": 0.65
    },
    "mental_health_multilingual": {
        "model": "intfloat/multilingual-e5-base",
        "dimension": 768,
        "description": "Tối ưu cho domain tâm lý - balanced Vietnamese + medical terms",
        "chunk_size": 800,
        "chunk_overlap": 150,
        "top_k": 3,
        "threshold": 0.65
    },
    "mental_health_fast": {
        "model": "intfloat/multilingual-e5-small", 
        "dimension": 384,
        "description": "Nhanh hơn cho demo, vẫn tốt cho psychological content",
        "chunk_size": 600,
        "chunk_overlap": 100,
        "top_k": 3,
        "threshold": 0.60
    },
    "gte_multilingual": {
        "model": "Alibaba-NLP/gte-multilingual-base",
        "dimension": 768,
        "description": "GTE multilingual model - state-of-the-art performance",
        "chunk_size": 800,
        "chunk_overlap": 150,
        "top_k": 3,
        "threshold": 0.70,
        "requires_trust_code": True
    }
}

# Domain-specific keywords for enhanced retrieval
MENTAL_HEALTH_KEYWORDS = {
    "psychological_conditions": [
        "trầm cảm", "depression", "lo âu", "anxiety", "stress", "căng thẳng",
        "rối loạn tâm lý", "mental disorder", "PTSD", "rối loạn ăn uống", "tự kỷ", "autism"
    ],
    "symptoms": [
        "triệu chứng", "symptom", "buồn chán", "mệt mỏi", "mất ngủ", "insomnia",
        "cô đơn", "lonely", "tuyệt vọng", "hopeless", "tự hại", "self-harm"
    ],
    "interventions": [
        "điều trị", "treatment", "tư vấn", "counseling", "therapy", "liệu pháp",
        "hỗ trợ", "support", "can thiệp", "intervention", "phòng ngừa", "prevention"
    ],
    "student_specific": [
        "học sinh", "student", "sinh viên", "university student", "áp lực học tập", "academic pressure",
        "kỳ thi", "exam", "bài tập", "homework", "học bài", "study", "bạn bè", "friendship"
    ],
    "support_resources": [
        "gia đình", "family", "giáo viên", "teacher", "bạn bè", "friends",
        "chuyên gia", "professional", "tâm lý học", "psychology", "đường dây nóng", "hotline"
    ],
    "crisis_indicators": [
        "tự tử", "suicide", "tự hại", "self-harm", "tự sát", "khẩn cấp", "emergency",
        "nguy hiểm", "danger", "cần giúp đỡ gấp", "urgent help"
    ]
}

# Emergency contact information
EMERGENCY_CONTACTS = {
    "suicide_prevention_hotline": "113",
    "mental_health_hotline": "18001567",
    "youth_support_hotline": "15009",
    "unicef_support": "18001524"
}
