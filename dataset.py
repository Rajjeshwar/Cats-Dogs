import requests


url = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip"
r = requests.get(url, allow_redirects=True)

open("PetImages.zip", "wb").write(r.content)
