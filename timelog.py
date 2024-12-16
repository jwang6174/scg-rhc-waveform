from time import time, strftime

def timelog(message, start_time):
  curr_time = time()
  diff_time = curr_time - start_time
  hrs, rem = divmod(diff_time, 3600)
  mins, secs = divmod(rem, 60)
  curr_time_str = strftime('%c')
  diff_time_str = '{:02}:{:02}:{:02}'.format(int(hrs), int(mins), int(secs))
  return f'{curr_time_str} | {diff_time_str} | {message}'
