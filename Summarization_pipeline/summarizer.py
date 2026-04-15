"""
summarizer.py — DSPy modules and facade for local Ollama.
Includes explicit Module classes for DSPy Optimization.
"""

from __future__ import annotations
import logging
import os
import dspy
import requests
from config import config

logger = logging.getLogger(__name__)
OPTIMIZED_PATH = "optimized_summarizer.json"

# ── Ollama LM adapter ─────────────────────────────────────────────────────────

class OllamaLM(dspy.LM):
    def __init__(self):
        super().__init__(
            model=f"ollama/{config.ollama_model}",
            model_type="chat",
            temperature=config.llm_temperature,      
            max_tokens=config.llm_max_new_tokens,    
            cache=False,
        )
        self.ollama_model = config.ollama_model
        self.base_url = config.ollama_base_url.rstrip("/")

        logger.info("Ollama LM initialized: model=%s", self.ollama_model)

    def __call__(self, prompt=None, messages=None, **kwargs) -> list[str]:
        temperature = kwargs.get("temperature", self.kwargs.get("temperature", 0.1))
        max_tokens = kwargs.get("max_tokens", 1024) 

        if messages is None:
            messages = [{"role": "user", "content": prompt or ""}]

        payload = {
            "model": self.ollama_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": 8192, 
            },
        }

        try:
            resp = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=180)
            resp.raise_for_status()
            text = resp.json().get("message", {}).get("content", "") or ""
        except Exception as e:
            logger.error(f"Ollama API Error: {e}")
            text = ""

        self.history.append({"messages": messages, "response": text})
        return [text]

def get_lm() -> dspy.LM:
    return OllamaLM()

# ── DSPy Signatures ───────────────────────────────────────────────────────────

class ParentChunkSummary(dspy.Signature):
    """تلخيص المقاطع النصية بدقة عالية مع الحفاظ على الأرقام والتواريخ."""
    chunk_text = dspy.InputField(desc="النص الأصلي")
    doc_title = dspy.InputField(desc="سياق المستند")
    summary = dspy.OutputField(desc="ملخص مركز وحقائقي")

class DocumentSummary(dspy.Signature):
    """دمج ملخصات الأقسام في هيكل تنظيمي واحد (نظرة عامة + نقاط)."""
    parent_summaries = dspy.InputField(desc="قائمة ملخصات الأقسام")
    doc_title = dspy.InputField()
    overview = dspy.OutputField(desc="فقرة شاملة توضح الغرض الأساسي")
    key_points = dspy.OutputField(desc="أهم المحاور الجوهرية (نقاط)")
    keywords = dspy.OutputField(desc="أهم 5 كلمات مفتاحية مفصولة بفاصلة")

class ClusterSummary(dspy.Signature):
    """تلخيص مجموعة مستندات تشترك في موضوع واحد مع استخراج الروابط الكبرى."""
    doc_summaries = dspy.InputField()
    topic_tag = dspy.InputField()
    summary = dspy.OutputField(desc="تحليل موضوعي موحد")
    keywords = dspy.OutputField(desc="كلمات مفتاحية للمجموعة")

class TopicTagger(dspy.Signature):
    """استخراج عنوان قصير جداً (كلمتين) يعبر عن صلب المحتوى."""
    texts = dspy.InputField()
    topic = dspy.OutputField(desc="اسم الموضوع")

# ── DSPy Modules (Explicitly defined for the Optimizer) ──────────────────────

class ParentSummarizerModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.summarize = dspy.ChainOfThought(ParentChunkSummary)

    def forward(self, chunk_text: str, doc_title: str) -> dspy.Prediction:
        return self.summarize(chunk_text=chunk_text, doc_title=doc_title)

class DocSummarizerModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.summarize = dspy.ChainOfThought(DocumentSummary)

    def forward(self, parent_summaries: str, doc_title: str) -> dspy.Prediction:
        return self.summarize(parent_summaries=parent_summaries, doc_title=doc_title)
    
class SummaryJudge(dspy.Signature):
    """
    أنت خبير في تقييم جودة النصوص. قم بتقييم الملخص بناءً على النص الأصلي بدقة.
    المعايير:
    1. الأمانة (Faithfulness): هل المعلومات حقيقية ومن صلب النص؟
    2. الصلة (Relevance): هل شمل الملخص النقاط الجوهرية؟
    3. التماسك (Coherence): هل اللغة العربية رصينة ومترابطة؟
    """
    original_text = dspy.InputField(desc="النص الأصلي")
    summary = dspy.InputField(desc="الملخص المراد تقييمه")

    faithfulness = dspy.OutputField(desc="النتيجة من 1-5")
    relevance = dspy.OutputField(desc="النتيجة من 1-5")
    coherence = dspy.OutputField(desc="النتيجة من 1-5")
    critique = dspy.OutputField(desc="شرح بسيط لسبب التقييم بالعربي")



# ── Summarizer Facade (Public Interface) ──────────────────────────────────────

class Summarizer:
    def __init__(self):
        lm = get_lm()
        dspy.configure(lm=lm)

        # Initialize the specific modules
        self.parent_mod = ParentSummarizerModule()
        self.doc_mod = DocSummarizerModule()
        self.cluster_mod = dspy.ChainOfThought(ClusterSummary)
        self.misc_mod = dspy.Predict(ParentChunkSummary) # Re-using Parent logic for misc
        self.tagger_mod = dspy.Predict(TopicTagger)
        self.judge_mod = dspy.Predict(SummaryJudge)

        # Load optimized prompts if they exist
        if os.path.exists(OPTIMIZED_PATH):
            self.parent_mod.load(OPTIMIZED_PATH)
            logger.info("Loaded optimized DSPy logic.")

    def summarize_parent(self, chunk_text: str, doc_title: str) -> str:
        pred = self.parent_mod(chunk_text=chunk_text, doc_title=doc_title)
        return pred.summary.strip()

    def summarize_document(self, parent_summaries: list[str], doc_title: str) -> dict:
        if len(parent_summaries) > 10:
            intermediates = []
            for i in range(0, len(parent_summaries), 5):
                batch = "\n".join(parent_summaries[i:i+5])
                res = self.doc_mod(parent_summaries=batch, doc_title=doc_title)
                intermediates.append(res.overview)
            parent_summaries = intermediates

        joined = "\n\n".join(parent_summaries)
        pred = self.doc_mod(parent_summaries=joined, doc_title=doc_title)
        
        return {
            "text": f"【نظرة عامة】\n{pred.overview}\n\n【أهم المحاور】\n{pred.key_points}",
            "keywords": [k.strip() for k in pred.keywords.split(",")]
        }

    def summarize_cluster(self, doc_summaries: list[str], topic_tag: str) -> dict:
        joined = "\n\n".join(doc_summaries)
        pred = self.cluster_mod(doc_summaries=joined, topic_tag=topic_tag)
        return {
            "text": pred.summary.strip(),
            "keywords": [k.strip() for k in pred.keywords.split(",")]
        }

    def generate_topic_tag(self, texts: list[str]) -> str:
        joined = "\n".join(texts[:3])
        pred = self.tagger_mod(texts=joined)
        return pred.topic.strip()
    