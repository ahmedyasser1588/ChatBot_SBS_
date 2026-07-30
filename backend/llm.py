import os
from typing import List, Dict, Optional

from dotenv import load_dotenv
from groq import Groq

load_dotenv()


class GroqLLM:
    """نسخة RAG بحتة (مفيش function calling خالص) — بديل GeminiLLM الأصلي، بس بموديل Groq.

    الفرق عن app.py الأصلي: هنا الموديل مش بيستدعي أي أدوات (query_players,
    compare_players...)، هو بس بياخد الـ context اللي جاله من الـ retriever ويجاوب
    منه. ده معناه إنه ممكن يجاوب غلط أو غير دقيق في أسئلة الفلترة/المقارنة الرقمية
    الدقيقة، لأنه مش بيشغل كود فعلي على البيانات زي الأصل.
    """

    SYSTEM_PROMPT = (
        "أنت SpotMe Assistant، مساعد جوه منصة SpotMe لاكتشاف المواهب الرياضية. "
        "جاوب على أسئلة الكشافة عن اللاعبين (كرة قدم، كرة سلة، كرة يد، كرة طائرة) "
        "باستخدام السياق (Context) اللي هيتبعتلك فقط. لو السياق مفيهوش إجابة واضحة، "
        "قول إنك مش متأكد بدل ما تخترع رقم أو اسم لاعب. اكتب إجابة كاملة وواضحة "
        "بفقرة أو فقرتين قصار (مش جملة واحدة مقتضبة)، اشرح الأرقام والمعطيات اللي "
        "بتذكرها ووضح ليه اخترت اللاعب/اللاعبين دول لو السؤال عن ترشيح أو مقارنة، "
        "بنفس لغة المستخدم."
    )

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API key must be provided via constructor or GROQ_API_KEY env var")

        self.model = model or os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.client = Groq(api_key=self.api_key)
        print("Groq LLM ready")

    def invoke(self, query: str, context: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        user_message = f"Context:\n{context}\n\nQuestion: {query}"

        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        if history:
            # history already stored as {"role": "user"|"assistant", "content": ...}
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.4,
            max_tokens=2048,
        )
        return response.choices[0].message.content

    def clear_chat(self):
        # مفيش حالة محادثة متخزنة هنا، الـ history بيتخزن في SpotMeChatbot زي الأصل.
        pass
