import os
if not os.path.exists('./checkpoint'):
  os.mkdir('./checkpoint')

if not os.path.exists('./weights'):
  os.mkdir('./weights')

if not os.path.exists('./log'):
  os.mkdir('./log')

ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
ENC_HID_DIM = 512
DEC_HID_DIM = 512
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
DEVICE = 'cuda:0'
NUM_ITERS = 80000
BEAM_SEARCH = False
PRINT_PER_ITER = 50
VALID_PER_ITER = 500
MAX_SAMPLE_VALID = 10000

MAX_LR =  0.01
PCT_START = 0.1

LOG = "./log/loger_luong.log"
CHECKPOINT = './checkpoint/seq2seq_luong_checkpoint.pth'
EXPORT = './weights/seq2seq_luong.pth'

MAXLEN = 46
NGRAM = 5
alphabets = 'aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~ ]'
PERCENT_NOISE= 0.3 # 2 in 5 word
