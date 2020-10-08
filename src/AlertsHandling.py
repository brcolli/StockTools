import smtplib
import ssl


class AlertsManager:

    def __init__(self, receiver_info):

        self.carriers = {
            'att': '@mms.att.net',
            'tmobile': '@tmomail.net',
            'verizon': '@vtext.com',
            'sprint': '@page.nextel.com'
        }

        # @TODO encrypt
        self.sender_email = 'stocktoolalerts@gmail.com'
        self.sender_email_password = '204436Brc!'

        self.receiver_info = receiver_info

        context = ssl.create_default_context()
        self.server = smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context)
        self.server.ehlo()
        self.server.login(self.sender_email, self.sender_email_password)

    def send_email_alert(self, msg):
        self.server.sendmail(self.sender_email, self.receiver_info['email'], msg)

    def send_sms_alert(self, msg):
        to_num = self.receiver_info['phone_number'] + self.carriers[self.receiver_info['carrier']]
        self.server.sendmail(self.sender_email, to_num, msg)


def main():
    receiver_info = {'email': 'sstben@gmail.com', 'phone_number': '9514522487', 'carrier': 'tmobile'}
    am = AlertsManager(receiver_info)
    #am.send_email_alert('test')
    am.send_sms_alert('Your automated alert message has been triggered')


if __name__ == '__main__':
    main()
