# SpotMe RAG

نسخة من SpotMe مبنية **بنفس هيكل worldcubChatbot بالظبط**: بيانات → chunks → embeddings
(SentenceTransformer) → ChromaDB → retrieval بالـ cosine similarity → LLM (Groq بدل
Gemini) يجاوب من الـ context بس، من غير أي function calling.

## الهيكل

```
SpotMeRAG/
├── backend/
│   ├── embeddedmanger.py     # EmbeddingManager (SentenceTransformer)
│   ├── vectorstore.py        # VectorStore (ChromaDB wrapper)
│   ├── RAG.py                # RAGRetriever (بحث بالـ cosine similarity)
│   ├── llm.py                # GroqLLM (RAG-style، بدون tools)
│   └── spotmechatbot.py      # SpotMeChatbot (orchestrator + session history)
├── data/
│   ├── players.json          # نفس بيانات اللاعبين الأصلية
│   ├── text_files/           # يتولد تلقائي: كل رياضة => ملف .txt
│   └── vector_store/         # يتولد تلقائي: قاعدة ChromaDB
├── frontend/
│   └── index.html            # واجهة شات بسيطة (RTL)
├── data_preparation.py       # يحول players.json -> نصوص -> embeddings -> Chroma
├── main.py                   # FastAPI app (health / chat / reset)
├── requirements.txt
└── .env.example
```

## إزاي تشغله

```bash
pip install -r requirements.txt
cp .env.example .env        # وحط مفتاح GROQ_API_KEY بتاعك جواه

python data_preparation.py  # يبني الـ vector store مرة واحدة بس
python main.py               # أو: uvicorn main:app --reload
```

هيشتغل على `http://localhost:8000`.

## الفرق الجوهري عن SpotMe الأصلي (app.py) — اقرأ ده كويس

الأصل كان مبني على **function calling**: الموديل بيستدعي functions حقيقية
(`query_players`, `compare_players`, `recommend_players`, `aggregate_players`...)
بتشتغل على الـ JSON مباشرة بمنطق Python دقيق 100%. ده معناه لما حد يسأل
"لاعبين عمرهم أقل من 20 سنة في الأهلي"، كان فيه فلتر برمجي فعلي بيرجع إجابة مضبوطة.

النسخة دي (RAG) بقت شغالة إزاي بالظبط:

1. كل لاعب اتحول لفقرة نصية عادية (اسم، سن، نادي، ai_score... إلخ في جملة واحدة).
2. كل فقرة اتحولت لـ embedding وخُزنت في ChromaDB.
3. أي سؤال بييجي، بياخد embedding ليه، ويدور على أقرب 5 لاعبين بالتشابه الدلالي
   (semantic similarity)، مش بفلترة حقيقية.
4. اللي يترجع من نتايج ده بيتحط كـ "context" ويتبعت للـ LLM (Groq) يلخصه ويجاوب منه.

### يعني إيه ده عملياً؟

| النوع | مثال سؤال | هيشتغل كويس؟ |
|---|---|---|
| بحث دلالي/وصفي | "لاعب سريع في كرة القدم" | ✅ كويس، embeddings بتفهم المعنى |
| بحث باسم لاعب معروف | "احكيلي عن Ali Khaled" | ✅ كويس عادةً |
| فلترة رقمية دقيقة | "لاعبين عمرهم أقل من 20 بالظبط" | ⚠️ مش مضمون — ممكن يرجع لاعبين قريبين مش مطابقين تمامًا |
| ترتيب/Top-N دقيق | "أفضل 3 لاعبين بالظبط حسب ai_score" | ⚠️ ضعيف — الـ retrieval بيرجع أقرب دلالياً مش أعلى رقمياً |
| مقارنة بين لاعبين محددين | "قارن بين X و Y" | ⚠️ ممكن الاتنين ميظهروش في نفس الـ top-5 |
| تجميع/إحصائيات (متوسط، عدد) | "متوسط عمر لاعبين الزمالك" | ❌ مش هيشتغل خالص، محتاج حساب فعلي مش استرجاع نصوص |

**خلاصة:** المقايضة اللي اتفقنا عليها هي إنك كسبت نفس معمارية worldcubChatbot
(أبسط، أسهل تفهمها وتشرحها، RAG classic)، وخسرت الدقة الرقمية اللي كانت في الأصل.
لو حبيت ترجع جزء من الدقة، الحل المتوسط (اللي مقترحه سابقاً) هو تسيب الأسئلة
الرقمية/التجميعية تستخدم دوال Python حقيقية زي الأصل، وتسيب RAG بس للأسئلة
الوصفية/العامة (زي "احكيلي عن فلسفة الكشف عن المواهب" أو أسئلة مفتوحة).

## تحسين ممكن مستقبلاً

- لو حبيت تحسن الدقة شوية من غير ما ترجع لـ function calling بالكامل: زود
  `top_k` في `RAG.py` (حالياً 5) لما السؤال يبان إنه محتاج مقارنة بين أكتر من لاعب.
- الموديل المستخدم `paraphrase-multilingual-MiniLM-L12-v2` بديل عن الأصلي
  `all-MiniLM-L6-v2` لأن بياناتنا فيها عربي (أسماء أندية ومراكز).
