#https://www.ncdc.noaa.gov/cdo-web/api/v2/{endpoint}
#
#Email:	jcremaldi@gmail.com
#Token:	tZVpouYOrIcrLDPIUXCDbVCMJutlLQfA



import pandas as pd
import requests
from pandas.io.json import json_normalize
pd.options.display.max_columns = None

headers = {'token':'tZVpouYOrIcrLDPIUXCDbVCMJutlLQfA'}

startdate = '2018-01-01'
enddate = '2018-12-31'
limit = 1000
city = 'GHCND:USC00504094'

codes = ['USW00026451', 'USW00025503', 'USC00478919']

results = pd.DataFrame(columns = ['code','attributes','datatype','date','station','value'])

# replace 'myToken' with the actual token, below


for code_primer in codes:
    code = 'GHCND:%s' % (code_primer)
    print(code_primer)
    try:
        url = 'http://www.ncdc.noaa.gov/cdo-web/api/v2/data?datasetid=GHCND&stationid=%s&startdate=%s&enddate=%s&limit=1000&includemetadata=false&datatypeid=PRCP&datatypeid=SNOW&units=metric' % (code,startdate,enddate)
        response = requests.get(url, headers = headers).json()
        response = json_normalize(response['results'])
        response['code'] = code_primer

        results = pd.concat([results,response])
        print(results)
    except:
        pass
print(len(results))
results.to_csv('raw_Data\scraped_weather_data.csv')































