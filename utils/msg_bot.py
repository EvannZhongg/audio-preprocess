web_hook_url = f"http://in.qyapi.weixin.qq.com/cgi-bin/webhook/send?key=025e4c8d-e8d7-4d23-bd76-a6084de66b1b"

import requests
import json

def send_msg(msg):
    headers = { 'Content-Type': 'application/json' }
    data = {
        "msgtype": "text",
        "text": {
            "content": f"{msg}"
        }   
    }
    response = requests.post(web_hook_url, headers=headers, data=json.dumps(data))

if __name__ == "__main__":
    send_msg("hello world")