from openai import OpenAI
tweet = "nigga what is the blac tace made for, nothing"
client = OpenAI(api_key = "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")
chatmessage = client.chat.completions.create(model = "gpt-3.5 turbo",
messages = [
{
"role": "user",
"content": f"Does this content contain cyberbullying elements?, reply with Yes or No:{tweet}"

}
]

)

print(chatmessage.message.content)
