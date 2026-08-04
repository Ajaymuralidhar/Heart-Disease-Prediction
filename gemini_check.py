import google.generativeai as genai

genai.configure(api_key="AIzaSyASXt9YPdqmdFUKa5RUV9CZ4QQcN370zko")
model = genai.GenerativeModel("gemini-2.5-flash")
response = model.generate_content("Summarize this: The heart pumps blood throughout the body, supplying oxygen and nutrients.")
print(response.text)