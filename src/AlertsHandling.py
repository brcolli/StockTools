import smtplib
import ssl


class AlertsManager:

    def __init__(self, receiver_info):

        # @TODO encrypt
        self.port = 465
        self.sender_email = 'stocktoolalerts@gmail.com'
        self.sender_email_password = '204436Brc!'

        self.receiver_info = receiver_info

        context = ssl.create_default_context()
        self.server = smtplib.SMTP_SSL('smtp.gmail.com', self.port, context=context)
        self.server.login(self.sender_email, self.sender_email_password)

    def send_email_alert(self, msg):
        self.server.sendmail(self.sender_email, self.receiver_info['email'], msg)


def main():

    receiver_info = {'email': 'sstben@gmail.com'}
    am = AlertsManager(receiver_info)
    am.send_email_alert('test')


if __name__ == '__main__':
    main()
