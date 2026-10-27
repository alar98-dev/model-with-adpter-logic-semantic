import sys
import time
sys.path.append('/home/alifer/swa-aeon')
from cortex.organs.fs_sense import FileSystemSenseOrgan
print('FS Sense acordado. Aguardando 10s...')
organ = FileSystemSenseOrgan()
time.sleep(10)
