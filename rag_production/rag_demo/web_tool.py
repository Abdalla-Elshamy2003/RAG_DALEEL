import requests
import os

class WebSearchTool:
    def __init__(self):
        self.api_key = os.getenv("TAVILY_API_KEY")

    def search(self, query):
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": "advanced",
            "max_results": 3
        }
        response = requests.post(url, json=payload).json()
        
        web_results = []
        for res in response.get("results", []):
            web_results.append({
                "text": res.get("content"),
                "source": res.get("url"),
                "score": 1.0 # External source
            })
        return web_results