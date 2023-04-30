# traffic-sign-recognizer
## Install missing package
`pip install -r requirements.txt`

### Run app
`python ./app/app.py`

`flask run --port=5000 --host=0.0.0.0`

### GET _/hello_

### POST _/image_


### TEST endpoint
curl localhost:5000/image -F file=@abc.jpg    

## Server Config

### create traffic-sign-recognizer.service
`sudo nano /etc/systemd/system/traffic-sign-recognizer.service`

### Add content
[Unit]    
Description=My Flask App - traffic-sign-recognizer    
After=network.target       

[Service]      
User=trung     
WorkingDirectory=/home/trung/traffic-sign-recognizer      
ExecStart=/usr/bin/python3 -m flask run --host=0.0.0.0 --port=5001     
Restart=always       

[Install]      
WantedBy=multi-user.target      


### Restart
`sudo systemctl daemon-reload`

### Start
`sudo systemctl start traffic-sign-recognizer.service`

### Start on Boot
`sudo systemctl enable traffic-sign-recognizer.service`


