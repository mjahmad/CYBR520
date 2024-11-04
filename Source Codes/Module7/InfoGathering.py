# Example of a simple script to gather information about a website (Passive Reconnaissance)
import requests

url = 'http://www.wvu.edu'
response = requests.get(url)

print("Headers:", response.headers)
print("Server Type:", response.headers.get('Server'))
