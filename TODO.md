# TODO - Remove OCR Dependencies

- [x] فهم المطلوب: شيل أي حاجة معتمدة على OCR
- [ ] تعديل `ingest_app/config.py` وإزالة `OCRConfig` وحقل `ocr` من `AppConfig`
- [ ] تعديل `ingest_app/payload_builders.py` وإزالة أي import/logic خاص بـ OCR
- [ ] تعديل `ingest_app/main_pipeline.py` لإلغاء تمرير إعدادات OCR
- [ ] تحديث `ingest_app/__init__.py` لو فيه وصف متعلق بـ OCR
- [ ] فحص نهائي للتأكد إنه لا توجد مراجع OCR مستخدمة داخل الـ pipeline
