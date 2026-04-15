"""
optimize_prompts.py — Strong Optimizer for Large Arabic Chunks.
Using high-density training examples to teach the model how to distill 
complex information into professional summaries.
"""

import logging
import os
import sys
import argparse

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dspy
from dspy.teleprompt import BootstrapFewShot
from summarizer import get_lm, ParentSummarizerModule
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("optimizer")

# ── High-Density Training Examples ──────────────────────────────────────────
# These are large, complex "Parent Chunks" to challenge the model
TRAINING_EXAMPLES = [
    {
        "chunk_text": """
        شهد العصر العباسي الأول (750-847م) تحولات جذرية في بنية الدولة الإسلامية، حيث انتقل مركز الثقل السياسي من دمشق إلى بغداد التي أسسها المنصور لتكون عاصمة العلم والتجارة. تميزت هذه الفترة بظهور "بيت الحكمة" في عهد الخليفة المأمون، والذي لم يكن مجرد مكتبة، بل أكاديمية علمية ضخمة ضمت مترجمين من مختلف الأديان والأعراق. تمت ترجمة أمهات الكتب اليونانية لأرسطو وأفلاطون، والكتب الهندية في الفلك والرياضيات مثل "السند هند". هذا الانفتاح المعرفي أدى إلى بزوغ فجر النهضة العلمية العربية، حيث برز علماء مثل الخوارزمي في الرياضيات، والكندي في الفلسفة. استمر هذا الازدهار بفضل الدعم المالي السخي من الخلفاء لطلاب العلم والمترجمين، مما جعل اللغة العربية لغة العلم العالمية الأولى في ذلك العصر، ومهد الطريق لاحقاً لعصر النهضة الأوروبية من خلال الأندلس وصقلية.
        """,
        "doc_title": "تاريخ الحضارة العباسية - الجزء الأول",
        "summary": "اتسم العصر العباسي الأول بنهضة علمية شاملة مركزها بغداد، حيث لعب بيت الحكمة دوراً محورياً في ترجمة العلوم اليونانية والهندية. وبفضل دعم الخلفاء، تحولت العربية لغة العلم العالمية وظهر رواد مثل الخوارزمي والكندي."
    },
    {
        "chunk_text": """
        تعد أمراض القلب والأوعية الدموية (CVDs) السبب الرئيسي للوفاة عالمياً، حيث تودي بحياة الملايين سنوياً. تشمل هذه الأمراض مجموعة من الاضطرابات التي تصيب القلب والشرايين، مثل مرض الشرايين التاجية، والسكتة الدماغية، وفشل القلب الاحتقاني. تكمن خطورة هذه الأمراض في تراكم اللويحات الدهنية (تصلب الشرايين) التي تعيق تدفق الدم والأكسجين إلى الأعضاء الحيوية. هناك عوامل خطر رئيسية يمكن التحكم بها، أهمها ارتفاع ضغط الدم، وارتفاع الكوليسترول الضار (LDL)، والسكري، والسمنة المفرطة الناتجة عن نمط الحياة الخامل. تنصح الجمعية الأمريكية للقلب بضرورة الفحص الدوري لمستويات ضغط الدم والدهون، واتباع نظام غذائي غني بالألياف وقليل الدهون المشبعة، بالإضافة إلى ممارسة النشاط البدني لمدة 150 دقيقة أسبوعياً للوقاية من المضاعفات الخطيرة مثل النوبات القلبية المفاجئة التي قد تؤدي إلى تلف دائم في عضلة القلب.
        """,
        "doc_title": "الدليل الطبي للوقاية من أمراض القلب",
        "summary": "تعتبر أمراض القلب والأوعية الدموية المسبب الأول للوفيات عالمياً، وتنتج غالباً عن تصلب الشرايين. تؤكد التوصيات الطبية على ضرورة التحكم في عوامل الخطر كضغط الدم والسكري، مع الالتزام بنظام غذائي صحي ونشاط بدني دوري للوقاية."
    },
    {
        "chunk_text": """
        يعتبر نجيب محفوظ المؤسس الحقيقي للرواية العربية الحديثة، حيث نقلها من المحلية إلى العالمية بفوزه بجائزة نوبل عام 1988. بدأت مسيرته بالمرحلة التاريخية التي استلهم فيها تاريخ مصر القديم، ثم انتقل إلى المرحلة الواقعية التي تجلت في "الثلاثية" الشهيرة (بين القصرين، قصر الشوق، السكرية). في هذه المرحلة، غاص محفوظ في تفاصيل الحارة المصرية، مصوراً الصراعات الاجتماعية والسياسية والتحولات الفكرية للطبقة المتوسطة بين الحربين العالميتين. بعد ذلك، اتجه إلى المرحلة الرمزية والفلسفية كما في "أولاد حارتنا"، حيث طرح أسئلة عميقة حول العدالة والوجود والصراع بين العلم والميتافيزيقا. تميز أسلوب محفوظ بالدقة اللغوية والقدرة الهائلة على بناء الشخصيات المعقدة، مما جعل أعماله مرآة صادقة للمجتمع المصري وتاريخه الحديث، وتحولت معظم رواياته إلى أعمال سينمائية وتلفزيونية شكلت وجدان المشاهد العربي لسنوات طويلة.
        """,
        "doc_title": "أعلام الأدب العربي الحديث",
        "summary": "أرسى نجيب محفوظ دعائم الرواية العربية الحديثة عبر ثلاث مراحل: التاريخية، والواقعية (الثلاثية)، والرمزية الفلسفية. تميز بقدرته على تصوير تحولات المجتمع المصري وبناء شخصيات معقدة، مما أهله لنيل جائزة نوبل وتخليد أعماله أدبياً وسينمائياً."
    }
]

# ── Stronger Metric Logic ───────────────────────────────────────────────────

def stronger_arabic_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> float:
    """
    Stricter metric for large chunks:
    - Penalizes summaries that copy more than 80% of original words.
    - Penalizes summaries that are too long or empty.
    - Rewards high F1 score (overlap with Gold Summary).
    """
    pred_summary = prediction.summary.strip() if prediction.summary else ""
    gold_summary = example.summary.strip()

    if not pred_summary: return 0.0

    # 1. Density Penalty: If summary is almost as long as the original, it's not a summary
    if len(pred_summary) > len(example.chunk_text) * 0.7:
        return 0.1

    # 2. Brevity Check: Minimum 10 words for these large chunks
    if len(pred_summary.split()) < 10:
        return 0.1

    # 3. Word Overlap (F1 Score)
    pred_words = set(pred_summary.lower().split())
    gold_words = set(gold_summary.lower().split())
    
    overlap = len(pred_words & gold_words)
    recall = overlap / max(len(gold_words), 1)
    precision = overlap / max(len(pred_words), 1)
    f1 = 2 * recall * precision / max(recall + precision, 1e-10)

    # 4. Language Consistency
    arabic_chars = sum(1 for c in pred_summary if '\u0600' <= c <= '\u06FF')
    if arabic_chars < 10:
        return 0.0 

    return f1

# ── Optimization Process ─────────────────────────────────────────────────────

def run_optimization(output_path: str):
    logger.info("Starting local optimization on large chunks...")
    lm = get_lm()
    dspy.configure(lm=lm)

    # Convert to DSPy Example format
    trainset = [
        dspy.Example(
            chunk_text=ex["chunk_text"],
            doc_title=ex["doc_title"],
            summary=ex["summary"],
        ).with_inputs("chunk_text", "doc_title")
        for ex in TRAINING_EXAMPLES
    ]

    # Use BootstrapFewShot to learn the best prompting strategy for large text
    optimizer = BootstrapFewShot(
        metric=stronger_arabic_metric,
        max_bootstrapped_demos=3,
        max_labeled_demos=len(trainset),
    )

    module = ParentSummarizerModule()
    
    logger.info("Compiling... this might take 5-10 minutes on local CPU/GPU.")
    optimized_module = optimizer.compile(module, trainset=trainset)

    optimized_module.save(output_path)
    logger.info(f"Success! Intelligence saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="optimized_summarizer.json")
    args = parser.parse_args()

    run_optimization(output_path=args.output)