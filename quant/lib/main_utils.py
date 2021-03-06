'''
Created on 22 Jun 2017

@author: wayne
'''
import logging
import os
import sys
import time
import smtplib
import cPickle as pickle
import pandas as pd
import numpy as np
from datetime import datetime as dt
from dateutil.relativedelta import relativedelta
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email import encoders

logging.basicConfig(format='%(asctime)s %(message)s')
logger = logging.getLogger('quant')
logger.setLevel(20)

MODEL_PATH = '/home/wayne/TempWork/models/'

EMPTY_EMAIL = '''
<html>
    <head></head>
    <body>
        <p>%s
        </p>
    </body>
</html>
'''

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_pickle(filename):
    ans = None
    if os.path.isfile(filename):
        with open(filename, 'rb') as f:
            ans = pickle.load(f)
            f.close()
    else:
        logger.warn('Could not find model file %s' % filename)
    return ans


def write_pickle(data, filename):
    try:
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
            f.close()
    except Exception as e:
        logger.warn('Failed to write pickle %s:\n%s' % (filename, str(e)))


try:
    EMAILADDRESS, EMAILPASSWORD = load_pickle('/home/wayne/quant.dat')
except:
    logger.info('Environment variable not loaded')


def try_once(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:
            logger.warn('Failed for %s' % str(*args))
            return None
    return wrapper


def try_and_check(func):
    def wrapper(*args, **kwargs):
        try:
            x = func(*args, **kwargs)
            if x is None:
                logger.info('No results for %s' % str(*args))
            return x
        except:
            logger.warn('Failed for %s' % str(*args))
            return None
    return wrapper


def try_again(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:
            logger.info('Wait 10 seconds and try again...')
            time.sleep(10)
            try:
                return func(*args, **kwargs)
            except:
                logger.warn('Failed for %s' % str(*args))
                return None
    return wrapper


def get_timeline(start_date, end_date):
    return pd.Series([0., 0.], index=[start_date, end_date]).resample('B').last()


def get_last_business_day(run_date=dt.today()):
    return pd.Series([0, 0], index=[run_date - relativedelta(days=5), run_date]).resample('B').last().index[-1]


def get_table_html(table, inc_index=True, inc_columns=True, width=None):
    assert isinstance(table, pd.DataFrame)
    data = table.copy()
    ans = ''''''
    if inc_index:
        data = pd.concat([pd.Series(data.index, index=data.index, name='Index').to_frame(), data], axis=1)
    if inc_columns:
        data = pd.concat([pd.Series(data.columns, index=data.columns, name='Columns').to_frame().T, data], axis=0)
    nrows = len(data)
    ncols = len(data.columns)
    cell_width = None if width is None else width / ncols
    for i in xrange(nrows):
        line_text = ''''''
        if i % 2 == 0:
            bg_text = '''bgcolor="#D9D9D9"'''
        else:
            bg_text = ''''''
        for j in xrange(ncols):
            cell_text = '''<font face='Arial'>%s</font>''' % str(data.iloc[i, j])
            if i == 0:
                cell_text = '''<strong>%s</strong>''' % cell_text
            cell_text = '''<p align='center'>%s</p>''' % cell_text
            if cell_width is None:
                cell = '''<td %s>%s
                </td>''' % (bg_text, cell_text)
            else:
                cell = '''<td width=%.1f %s>%s
                </td>''' % (cell_width, bg_text, cell_text)
            line_text += cell
        line = '''<tr>%s
        </tr>''' % line_text
        ans += line
    if width is None:
        width_text = ''
    else:
        width_text = ''' width="%d" ''' % width
    ans = '''<table border=0 cellspacing="0"%s>%s
    </table>''' % (width_text, ans)
    return ans


class Email(object):
    def __init__(self, from_address, to_addresses, subject, template=None):
        self.from_address = from_address
        self.to_addresses = to_addresses
        self.subject = subject
        self.template = template
        self.content = ''''''
        self.attachments = []
        self.images = []
    
    def add_text(self, text, align='left', bold=False):
        center = '''<font face='Arial'>%s</font>''' % text
        if bold:
            center = '''<strong>''' + center + '''</strong>'''
        self.content += '''<p align="%s", line-height="1.1">%s<br></p>
        ''' % (align, center)
    
    def add_color_text(self, texts, align='left', bold=False):
        center = ''''''
        for text, color in texts:
            center += '''<font face='Arial', color='%s'>%s</font>''' % (color, text)
        if bold:
            center = '''<strong>''' + center + '''</strong>'''
        self.content += '''<p align='%s', line-height='1.1'>%s<br></p>
        ''' % (align, center)
    
    def add_date(self, date_time):
        self.add_text(date_time.strftime('%B %d, %Y'), 'right', True)
    
    def add_image(self, image, width=None, height=None):
        if os.path.isfile(image):
            scale_arg = ''''''
            if width is not None:
                scale_arg += ''' width='%d' ''' % width
            if height is not None:
                scale_arg += ''' height='%d' ''' % height
            self.content += '''<img src='cid:%s'%s/><br>''' % (image, scale_arg)
            self.images.append(image)
        else:
            logger.warn('%s does not exist' % image)
    
    def add_table(self, table, inc_index=True, inc_columns=True, width=None):
        table_text = get_table_html(table, inc_index, inc_columns, width)
        self.content += table_text
    
    def add_attachment(self, filename):
        if os.path.isfile(filename):
            self.attachments.append(filename)
        else:
            logger.warn('%s does not exist' % filename)
    
    def create_content(self):
        self.email_body = EMPTY_EMAIL % self.content
    
    def create_message(self):
        self.create_content()
        self.msg = MIMEMultipart()
        self.msg['From'] = self.from_address
        self.msg['To'] = ', '.join(self.to_addresses)
        self.msg['Subject'] = self.subject
        self.msg.attach(MIMEText(self.email_body, 'html'))
        if len(self.images) > 0:
            for f in self.images:
                fp = open(f, 'rb')
                img = MIMEImage(fp.read())
                fp.close()
                img.add_header('Content-ID', '{}'.format(f))
                self.msg.attach(img)
        if len(self.attachments) > 0:
            for f in self.attachments:
                with open(f, "rb") as fil:
                    part = MIMEApplication(
                        fil.read(),
                        Name=os.path.basename(f)
                    )
                part['Content-Disposition'] = 'attachment; filename="%s"' % os.path.basename(f)
                self.msg.attach(part)
    
    def send_email(self):
        self.create_message()
        try:
            server = smtplib.SMTP('smtp.live.com:587')
            server.ehlo()
            server.starttls()
            server.login(EMAILADDRESS, EMAILPASSWORD)
            server.sendmail(self.from_address, self.to_addresses, self.msg.as_string())
            server.quit()
        except Exception as e:
            logger.warn('Failed to send email: %s' % str(e))
