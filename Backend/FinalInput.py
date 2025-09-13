import requests

url = "https://apifreellm.com/api/chat"
headers = {
  "Content-Type": "application/json"
}

def Query_llm(data):
  resp = requests.post(url, headers=headers, json=data)
  js = resp.json()
  if js.get('status') == 'success':
    return(js['response'])
  else:
    return ('Error:', js.get('error'), js.get('status'))