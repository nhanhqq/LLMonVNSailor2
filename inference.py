import requests
import json

def chat_with_ollama(question):
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": "gemma_rag_vn",
        "prompt": question,
        "stream": True
    }
    
    try:
        response = requests.post(url, json=payload, stream=True)
        if response.status_code == 200:
            print("Model: ", end="", flush=True)
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    print(chunk.get("response", ""), end="", flush=True)
            print()
        else:
            print(f"Error: {response.status_code}")
    except Exception as e:
        print(f"Connection error: {e}")

if __name__ == "__main__":
    print("="*50)
    print("Chatbot Gemma RAG VN - Gõ 'exit' hoặc 'quit' để thoát")
    print("="*50)
    
    while True:
        try:
            print("\n" + "-"*50)
            question = input("Bạn: ")
            if question.lower() in ['exit', 'quit']:
                break
                
            if not question.strip():
                continue
                
            chat_with_ollama(question)
            
        except KeyboardInterrupt:
            print("\nĐã thoát ứng dụng.")
            break
