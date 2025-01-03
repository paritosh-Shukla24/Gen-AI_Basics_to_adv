# Please install OpenAI SDK first: `pip3 install openai`

# Install the OpenAI SDK if you haven't already: pip install openai

import openai

# Set your API key
openai.api_key = "sk-91c58b0c23884b52b97699b788418e70"

# If using a custom API endpoint, set the base_url (optional)
openai.api_base = "https://api.deepseek.com"

# Create a chat completion request
response = openai.ChatCompletion.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ]
)

# Print the response
print(response['choices'][0]['message']['content'])
