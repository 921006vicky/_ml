import os
import ollama
from langchain_core.tools import Tool

response = ollama.chat(model='llama3.2:3b', messages=[        
        {
            'role': 'user',
            'content': 'What day is today?',
            },
        ])
    
output = response['message']['content']

print("="*50)
print("result=", output)