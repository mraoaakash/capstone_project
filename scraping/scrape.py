import requests
from bs4 import BeautifulSoup

# Using beautiful soup to scrape the data


page=1
url = f'https://www.medindia.net/patients/doctor_search/dr_result.asp?page={page}&specialist=Pathology'
page = requests.get(url)
soup = BeautifulSoup(page.content, 'html.parser')
class_name = "dirlist_box list-page"
results = soup.find_all('div', class_=class_name)
print(results[0].prettify())

# iterating through the results and storing the data in a list
# data = []
# for result in results:
#     name = result.find('h3').text
#     address = result.find('p').text
#     data.append((name, address))
