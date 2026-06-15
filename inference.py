import requests

def chat_with_ollama(context, question):
    url = "http://localhost:11434/api/generate"
    prompt = f"Dựa vào thông tin sau:\n{context}\n\nCâu hỏi: {question}"
    
    payload = {
        "model": "gemma_rag_vn",
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return response.json().get("response", "")
        return f"Error: {response.status_code}"
    except Exception as e:
        return f"Connection error: {e}"

if __name__ == "__main__":
    test_context = "Gemma là một dòng mô hình ngôn ngữ mở, gọn nhẹ được xây dựng từ cùng một nghiên cứu và công nghệ tạo ra Gemini. Gemma được cung cấp bằng các kích thước 2B và 7B, 9B."
    test_question = "Gemma có những kích thước nào?"
    
    answer = chat_with_ollama(test_context, test_question)
    print("Câu trả lời từ model:")
    print(answer)
