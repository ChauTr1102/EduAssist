from api.services import *


class LLM:
    def __init__(self, gemini_apikey=None):
        self.model_llm = None
        self.gemini_apikey = gemini_apikey

    def prompt_summarize(self, script):
        prompt_summarize_template = ChatPromptTemplate.from_messages(
            [
                ("system", f"{PROMPT_SUMMARIZE}\n"),
                ("system", "{script}\n"),
            ]
        )
        prompt = prompt_summarize_template.format(script=script)
        return prompt

    def prompt_qa_script(self, user_input, summarize_script):
        prompt_qa_template = ChatPromptTemplate.from_messages(
            [
                ("system", f"{PROMPT_QA}\n"),
                ("system", "Bản tóm tắt cuộc họp:\n"),
                ("system", "{summarize_script}\n"),
                ("system", "Hãy trả lời câu hỏi của người dùng: {user_input}\n"),
            ]
        )
        prompt = prompt_qa_template.format(summarize_script=summarize_script, user_input=user_input)
        return prompt

    async def send_message_gemini(self, prompt: str) -> str:
        self.model_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=self.gemini_apikey,
            # streaming=False,
            temperature=0.4
        )

        # Dùng HumanMessage để wrap prompt
        response = await self.model_llm.ainvoke(prompt)

        # Trả về nội dung chính
        return response.content