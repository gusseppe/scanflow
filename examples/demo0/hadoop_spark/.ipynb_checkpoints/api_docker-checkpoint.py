import requests 
import json
import time
import sys

sample_input = { 
        "columns": [ 
            "x_1", 
            "x_2", 
            "x_3", 
            "x_4", 
        ], 
        "data": [ 
            [1.646629	,  0.811971, -0.712092,  1.570961],
	    [-0.989919	,  0.987401, 0.266892,  0.485329]  
        ] 
}     
port = int(sys.argv[1])
n_requests = int(sys.argv[2])

url = f'http://localhost:{port}/invocations'

start = time.time()
for _ in range(n_requests):
	response = requests.post( 
		         url=url, data=json.dumps(sample_input), 
		          headers={"Content-type": "application/json; format=pandas-split"})
	response_json = json.loads(response.text)
	#print(response_json)

end = time.time()
print(f'Container in port: {port} | n_requests: {n_requests}')
print(f'Time elapsed: {end-start}')
