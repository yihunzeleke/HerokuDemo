# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 15:38:22 2020

@author: yihun
"""

import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url, json={'experience':2, 'test_score':9,'interview_score':6})
                
print(r.json())