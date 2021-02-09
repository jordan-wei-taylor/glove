from   datetime import datetime, timedelta
import logging

clear_line  = ' ' * 150

logging.basicConfig(format = f'\r{clear_line}\r[%(asctime)s] %(message)s', datefmt='%d-%m-%Y %X', level = logging.INFO)
logging.StreamHandler.terminator = ""

rprint = logging.info

def logger(msg):
    rprint(msg + '\n')
    
class Update():
    
    def __init__(self, msg, n, start = 0):
        self.msg    = msg
        self.n      = n
        self.start  = start
        
        self.count  = start
        self.time   = datetime.now()
        
    def __repr__(self):
        return f'Update()'
    
    def increment(self):
        self.count += 1
        
    def display(self, **kwargs):
        if kwargs:
            extra = ' [' + ', '.join([f'{k} = {v:,.3e}' for k, v in kwargs.items()]) + ']'
        else:
            extra = ''
        msg = f'{self.msg} {self.count:,} of {self.n:,} ({self.count / self.n:.2%}){extra}'
        
        if self.start < self.count < self.n:
            p       = (self.count - self.start) / self.n
            current = datetime.now()
            delta   = (current - self.time).total_seconds()
            seconds = delta / p * (1 - p)
            msg     = f"{msg} | eta : {(current + timedelta(seconds = seconds)).strftime('%d-%m-%Y %X')}"
        rprint(msg)
