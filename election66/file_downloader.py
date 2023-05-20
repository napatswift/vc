import requests
import gdown
from bs4 import BeautifulSoup

page = requests.get('https://www.ect.go.th/ect_th/news_page.php?nid=21139');
page_bs = BeautifulSoup(page.content)

table_data = page_bs.find('div', attrs={'class':'bodynewspage'}).find_all('td')

province_list = {}
for td in table_data:
    if not td.find('a'): continue
    prov = td.find('a')
    province_list[prov.text] = prov.attrs['href']

for prov, url in province_list.items():
    if 'drive.google.com' not in url: continue
    
    gdown.download_folder(url, output=prov)