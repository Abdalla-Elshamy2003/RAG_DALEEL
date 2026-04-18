"""
summarizer.py — DSPy modules and facade for local Ollama.
Includes explicit Module classes for DSPy Optimization.
"""

from __future__ import annotations
import logging
import os
import re
import dspy
import requests
from config import config

logger = logging.getLogger(__name__)
OPTIMIZED_PATH = "optimized_summarizer.json"

# ── Language Filtering ───────────────────────────────────────────────────────

def filter_non_target_languages(text: str) -> str:
    """
    Remove Chinese, Japanese, and other non-Arabic/English text.
    Keep only: Arabic, English, numbers, punctuation, and spaces.
    """
    # Pattern: Keep Arabic (0600-06FF), English, numbers, common punctuation/spaces
    # Remove CJK characters (4E00–9FFF for Chinese, 3040–309F for Japanese, etc.)
    pattern = r'[\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF\uAC00-\uD7AF]+'
    return re.sub(pattern, '', text).strip()

def clean_output(text: str) -> str:
    """Clean and validate output text."""
    text = filter_non_target_languages(text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def clean_json_response(json_str: str) -> str:
    """
    Clean JSON response by removing Chinese characters and fixing common issues.
    """
    # Remove Chinese characters
    json_str = filter_non_target_languages(json_str)
    
    # Fix common JSON issues
    json_str = json_str.strip()
    
    # Remove any trailing commas before closing braces/brackets
    json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
    
    return json_str

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
            resp = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=config.ollama_timeout)
            resp.raise_for_status()
            text = resp.json().get("message", {}).get("content", "") or ""
            
            # Clean the JSON response to remove Chinese characters and fix JSON issues
            text = clean_json_response(text)
            
        except Exception as e:
            logger.error(f"Ollama API Error: {e}")
            text = ""

        self.history.append({"messages": messages, "response": text})
        return [text]

def get_lm() -> dspy.LM:
    return OllamaLM()

# ── DSPy Signatures ───────────────────────────────────────────────────────────

class ParentChunkSummary(dspy.Signature):
    """تلخيص المقاطع النصية بدقة عالية مع الحفاظ على الأرقام والتواريخ.
    
    ⚠️ CRITICAL WARNING: Your response MUST be in Arabic ONLY or English ONLY.
    DO NOT use Chinese, Japanese, or any other language. Use ONLY Arabic or English.
    If the input text is Arabic, respond in Arabic. If English, respond in English.
    
    تحذير هام: يجب أن تكون إجابتك بالعربية فقط أو بالإنجليزية فقط.
    لا تستخدم الصينية أو اليابانية أو أي لغة أخرى."""
    chunk_text = dspy.InputField(desc="النص الأصلي")
    doc_title = dspy.InputField(desc="سياق المستند")
    summary = dspy.OutputField(desc="ملخص مركز وحقائقي (عربي أو إنجليزي فقط)")

class DocumentSummary(dspy.Signature):
    """دمج ملخصات الأقسام في هيكل تنظيمي واحد (نظرة عامة + نقاط).
    
    ⚠️ CRITICAL WARNING: ALL responses MUST be in Arabic ONLY or English ONLY.
    DO NOT use Chinese, Japanese, or any other language. Use ONLY Arabic or English.
    Match the language of the input summaries.
    
    تحذير هام: يجب أن تكون جميع الإجابات بالعربية فقط أو بالإنجليزية فقط.
    لا تستخدم الصينية أو اليابانية أو أي لغة أخرى. استخدم نفس لغة الملخصات المدخلة."""
    parent_summaries = dspy.InputField(desc="قائمة ملخصات الأقسام")
    doc_title = dspy.InputField()
    overview = dspy.OutputField(desc="فقرة شاملة توضح الغرض الأساسي (عربي أو إنجليزي فقط)")
    key_points = dspy.OutputField(desc="أهم المحاور الجوهرية - نقاط (عربي أو إنجليزي فقط)")
    keywords = dspy.OutputField(desc="أهم 5 كلمات مفتاحية مفصولة بفاصلة (عربي أو إنجليزي فقط)")

class ClusterSummary(dspy.Signature):
    """تلخيص مجموعة مستندات تشترك في موضوع واحد مع استخراج الروابط الكبرى.
    
    ⚠️ تحذير هام: يجب أن تكون الإجابة بالعربية فقط أو بالإنجليزية فقط.
    لا تستخدم الصينية أو أي لغة أخرى."""
    doc_summaries = dspy.InputField()
    topic_tag = dspy.InputField()
    summary = dspy.OutputField(desc="تحليل موضوعي موحد (عربي أو إنجليزي فقط)")
    keywords = dspy.OutputField(desc="كلمات مفتاحية للمجموعة (عربي أو إنجليزي فقط)")

class TopicTagger(dspy.Signature):
    """استخراج عنوان قصير جداً (كلمتين) يعبر عن صلب المحتوى.
    
    ⚠️ تحذير: استخدم العربية أو الإنجليزية فقط. بدون صينية أو لغات أخرى."""
    texts = dspy.InputField()
    topic = dspy.OutputField(desc="اسم الموضوع (عربي أو إنجليزي فقط، كلمتين فقط)")

class SummaryJudge(dspy.Signature):
    """
    أنت خبير في تقييم جودة النصوص. قم بتقييم الملخص بناءً على النص الأصلي بدقة.
    المعايير:
    1. الأمانة (Faithfulness): هل المعلومات حقيقية ومن صلب النص؟
    2. الصلة (Relevance): هل شمل الملخص النقاط الجوهرية؟
    3. التماسك (Coherence): هل اللغة رصينة ومترابطة؟
    
    ⚠️ تحذير: أجب بالعربية فقط. لا تستخدم الصينية أو لغات أخرى.
    """
    original_text = dspy.InputField(desc="النص الأصلي")
    summary = dspy.InputField(desc="الملخص المراد تقييمه")

    faithfulness = dspy.OutputField(desc="النتيجة من 1-5 (رقم فقط)")
    relevance = dspy.OutputField(desc="النتيجة من 1-5 (رقم فقط)")
    coherence = dspy.OutputField(desc="النتيجة من 1-5 (رقم فقط)")
    critique = dspy.OutputField(desc="شرح بسيط لسبب التقييم بالعربية فقط")

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

    def _retry_with_strict_instructions(self, module, **kwargs):
        """Retry with stricter language instructions if parsing fails."""
        try:
            return module(**kwargs)
        except Exception as e:
            if "JSONAdapter failed to parse" in str(e):
                logger.warning(f"Parsing failed, retrying with stricter instructions: {e}")
                # Add strict language enforcement to the prompt
                original_prompt = kwargs.get('parent_summaries', kwargs.get('chunk_text', ''))
                strict_prompt = f"""⚠️ CRITICAL: Respond ONLY in Arabic or English. NO Chinese/Japanese characters allowed.

تحذير: أجب بالعربية أو الإنجليزية فقط. لا صينية مطلقاً.

{original_prompt}"""
                
                if 'parent_summaries' in kwargs:
                    kwargs['parent_summaries'] = strict_prompt
                elif 'chunk_text' in kwargs:
                    kwargs['chunk_text'] = strict_prompt
                
                return module(**kwargs)
            else:
                raise e

    def summarize_parent(self, chunk_text: str, doc_title: str) -> str:
        pred = self._retry_with_strict_instructions(
            self.parent_mod, 
            chunk_text=chunk_text, 
            doc_title=doc_title
        )
        text = pred.summary.strip() if pred.summary else ""
        return clean_output(text)

    def summarize_document(self, parent_summaries: list[str], doc_title: str) -> dict:
        if len(parent_summaries) > 10:
            intermediates = []
            for i in range(0, len(parent_summaries), 5):
                batch = "\n".join(parent_summaries[i:i+5])
                res = self._retry_with_strict_instructions(
                    self.doc_mod, 
                    parent_summaries=batch, 
                    doc_title=doc_title
                )
                intermediates.append(clean_output(res.overview or ""))
            parent_summaries = intermediates

        joined = "\n\n".join(parent_summaries)
        pred = self._retry_with_strict_instructions(
            self.doc_mod, 
            parent_summaries=joined, 
            doc_title=doc_title
        )
        
        overview = clean_output(pred.overview or "")
        key_points = clean_output(pred.key_points or "")
        keywords_str = pred.keywords or ""
        
        return {
            "text": f"【نظرة عامة】\n{overview}\n\n【أهم المحاور】\n{key_points}",
            "keywords": [k.strip() for k in keywords_str.split(",") if k.strip()]
        }

    def summarize_cluster(self, doc_summaries: list[str], topic_tag: str) -> dict:
        joined = "\n\n".join(doc_summaries)
        pred = self.cluster_mod(doc_summaries=joined, topic_tag=topic_tag)
        summary_text = clean_output(pred.summary.strip() if pred.summary else "")
        keywords_str = pred.keywords or ""
        return {
            "text": summary_text,
            "keywords": [k.strip() for k in keywords_str.split(",") if k.strip()]
        }

    def generate_topic_tag(self, texts: list[str]) -> str:
        joined = "\n".join(texts[:3])
        pred = self.tagger_mod(texts=joined)
        return clean_output(pred.topic.strip() if pred.topic else "")
