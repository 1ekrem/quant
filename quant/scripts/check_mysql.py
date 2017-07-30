'''
Created on 30 Jul 2017

@author: wayne
'''
import subprocess
from quant.lib import data_utils as du
from quant.lib.main_utils import logger

FIND_MYSQL = "ps aux | grep mysql"
SUDO_SCRIPT ='echo gone0290eric | sudo -S %s'
KILL_PROCESS = "kill %s"
START_MYSQL = "service mysql start"


def check_mysql():
    return du.get_database_connection() is not None


def execute_script_and_return_output(script):
    return subprocess.check_output(script, shell=True)


def find_mysql_process():
    output = execute_script_and_return_output(FIND_MYSQL)
    ans = []
    for x in output.split('\n'):
        tmp = x.split(' ')
        if tmp[0] == 'mysql':
            i = 1
            while tmp[i] == '':
                i += 1
            ans.append(tmp[i])
    return ans


def kill_process(idx):
    execute_script_and_return_output(KILL_PROCESS % idx)


def start_mysql():
    execute_script_and_return_output(START_MYSQL)


def restart_mysql():
    p = find_mysql_process()
    if len(p) > 0:
        logger.info('Found mysql process: %s' % str(p))
    for idx in p:
        logger.info('Killing process %s' % idx)
        kill_process(idx)
    logger.info('Starting mysql')
    start_mysql()


def main():
    if check_mysql():
        logger.info('MYSQL is running fine.')
    else:
        logger.info('Attempting to restart MYSQL')
        restart_mysql()


if __name__ == '__main__':
    main()


