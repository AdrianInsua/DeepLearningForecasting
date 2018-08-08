# General libraries
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import csv
import pdb
import json

# Queries
from Service.Queries import query

class EntregasService:
    def __init__(self, group, v, conn):
        self.group = group
        self.v = v
        self.conn = conn
        self.initDate = None
        self.daysGap = None
        self.variables = None

    def composeQuery(self, query):
        if self.variables is None:
            return query
        endDate = self.initDate - timedelta(days=self.daysGap)

        # Concatenate where clause
        for var in self.variables:
            query = query.replace(var, self.variables[var])

        # Replace dates
        query = query.replace('@end_date', "'"+self.initDate.strftime('%Y-%m-%d')+"'")
        query = query.replace('@start_date', "'"+endDate.strftime('%Y-%m-%d')+"'")
        return query

    def get_corpus(self, model, initDate = datetime.now(), daysGap=30):
        self.initDate = initDate
        self.daysGap = daysGap
        if model=='prevision_retraso':
            return self.get_prevision_retraso()
        elif model=='prevision_pedidos':
            return self.get_prevision_pedidos()
        elif model=='prevision_presupuesto':
            return self.get_prevision_presupuesto_train()
        elif model=='paises':
            return self.get_paises()

    def get_prevision_retraso(self):
        self.variables = {
            '$where': 'date(fecha_pedido) BETWEEN @start_date AND @end_date'
        }
        finalQuery = self.composeQuery(query['prevision_ko'])
        print('Query: '+finalQuery) if self.v >= 2 else None
        result = pd.read_sql(finalQuery, self.conn)
        print(result) if self.v >= 3 else None
        return result, None

    def get_weather(self):
        weather = pd.read_csv('clima.csv', quoting=csv.QUOTE_NONE, names=['DATE','PRCP','SNWD','TAVG','TMAX','TMIN'], error_bad_lines=True, encoding='utf-8')
        weather['PRCP'] = weather['PRCP'].apply(lambda x: None if x == -9999 else x)
        weather['SNWD'] = weather['SNWD'].apply(lambda x: None if x == -9999 else x)
        weather['TAVG'] = weather['TAVG'].apply(lambda x: None if x == -9999 else x)
        weather['TMAX'] = weather['TMAX'].apply(lambda x: None if x == -9999 else x)
        weather['TMIN'] = weather['TMIN'].apply(lambda x: None if x == -9999 else x)
        weatherGrouped = weather.groupby('DATE').mean()
        print(weatherGrouped) if self.v >= 3 else None
        return weatherGrouped

    def get_prevision_pedidos(self):
        finalQuery = self.composeQuery(query['prevision_pedidos'])
        print('Query: '+finalQuery) if self.v >= 2 else None
        queries = finalQuery.split('--')
        result = [pd.read_sql_query(fq, self.conn) for fq in queries]
        result = result[len(result)-1]
        df = result.groupby(['id_fecha'], as_index=False).sum()
        print(df['id_fecha'])
        weatherGrouped = self.get_weather()
        for i, r in df.iterrows():
            print(weatherGrouped.filter(like=str(r['id_fecha']), axis=0)['PRCP'])
            try:
                df.set_value(i, 'PRCP', weatherGrouped.filter(like=str(r['id_fecha']), axis=0)['PRCP'].values[0])
                df.set_value(i, 'SNWD', weatherGrouped.filter(like=str(r['id_fecha']), axis=0)['SNWD'].values[0])
                df.set_value(i, 'TAVG', weatherGrouped.filter(like=str(r['id_fecha']), axis=0)['TAVG'].values[0])
                df.set_value(i, 'TMAX', weatherGrouped.filter(like=str(r['id_fecha']), axis=0)['TMAX'].values[0])
                df.set_value(i, 'TMIN', weatherGrouped.filter(like=str(r['id_fecha']), axis=0)['TMIN'].values[0])
            except Exception as e:
                print(e)

        del df['codigo_pais']
        print(result) if self.v >= 3 else None
        print(df) if self.v >= 3 else None
        return result, df

    def get_prevision_presupuesto_train(self):
        self.variables = {
            '$where': 's.id_fecha >= \'2015-01-01\' s.and id_fecha < \'2017-01-01\'',
            '$group': self.group
        }
        finalQuery = self.composeQuery(query['prevision_presupuesto'])
        print('Query:' + finalQuery) if self.v >= 2 else None
        result = pd.read_sql(finalQuery, self.conn)
        # weather = self.get_weather()
        # for i, r in result.iterrows():
        #     try:
        #         result.set_value(i, 'TAVG', weather.filter(like=str(r['id_fecha']), axis=0)['TAVG'].values[0])
        #         result.set_value(i, 'TMAX', weather.filter(like=str(r['id_fecha']), axis=0)['TMAX'].values[0])
        #         result.set_value(i, 'TMIN', weather.filter(like=str(r['id_fecha']), axis=0)['TMIN'].values[0])
        #     except Exception as e:
        #         print(e)

        # # pdb.set_trace()
        
        # del result['id_fecha']

        result = result.fillna(0)
        print(result)

        return [result, None], None

    def get_paises(self):
        finalQuery = self.composeQuery(query['paises'])
        result = pd.read_sql(finalQuery, self.conn)
        with open('countries.json', 'r') as f:
            data = json.load(f)
        new_data = {
            "type": "FeatureCollection",

            "features": []
        }
        for i in data['features']:
            # new_data['features'].append((result.filter(like = i['properties']['ISO_A3'])))
            new_row = result[result['alpha3'] == i['properties']['ISO_A3']]
            print(new_row)
            if(len(new_row) > 0):
                i['properties']['ISO_A2'] = new_row['alpha2'].values[0]
                print(i)
                new_data['features'].append(i)
        
        with open('new-countries.json', 'w') as f:
            json.dump(new_data, f)