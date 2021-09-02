from flask import Flask, request
import json
import smtplib




gmail_user = 'wisrovi.rodriguez@gmail.com'
gmail_password = 'ywmxtmeaxpgdjbhr'
host = 'smtp.gmail.com'
puerto = 465
    
app = Flask(__name__)

@app.route('/', methods=['POST'])
def hello():
	if request.method == 'POST':		
		data = request.json
		print(data)
		
		gmail_user_post = data.get("user")
		if gmail_user_post is not None:
			gmail_user = gmail_user_post
		
		gmail_password_post = data.get("password")
		if gmail_password_post is not None:
			gmail_password = gmail_password_post
			
		host_post = data.get("host")
		if host_post is not None:
			host = host_post
			
		port_post = data.get("port")
		if port_post is not None:
			puerto = port_post
		
		sent_from = gmail_user
		
		subject = data.get("subject")
		to = data.get("to")
		message = data.get("message")
		if message is not None and subject is not None and to is not None:
			print(message, subject, to)
			email_text = """\
From: %s
To: %s
Subject: %s


%s
""" % (sent_from, ", ".join([to]), subject, message)
			
			try:
				server = smtplib.SMTP_SSL(host, puerto)
				server.ehlo()
				server.login(gmail_user, gmail_password)
				server.sendmail(sent_from, to, email_text)
				server.close()
				return "email send"
			except:
				return "error to send email"
		else:
			return "incorrect data"		
	else:
		return 'Hello, World!'


@app.route('/help')
def help_service():
    OBJ = dict()
    OBJ["variables"] = "La api recibe tres variables: subject, to y message, estas se deben enviar en formato json"
    OBJ["subject"] = "asunto del correo"
    OBJ["to"] = "destinatario del correo"
    OBJ["message"] = "mensaje del correo o body"
    return json.dumps(OBJ, indent=4)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)