from urllib.request import urlopen

url = "https://jsonplaceholder.typicode.com/posts/1"
response = urlopen(url)
print(response.read())
