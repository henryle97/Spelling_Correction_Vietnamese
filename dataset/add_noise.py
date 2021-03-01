import numpy as np
import re
import unidecode
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import string


class SynthesizeData(object):
    """
    Uitils class to create artificial miss-spelled words
    Args:
        vocab_path: path to vocab file. Vocab file is expected to be a set of words, separate by ' ', no newline charactor.
    """

    def __init__(self, vocab_path=""):

        # self.vocab = open(vocab_path, 'r', encoding = 'utf-8').read().split()
        self.tokenizer = word_tokenize
        self.word_couples = [['sương', 'xương'], ['sĩ', 'sỹ'], ['sẽ', 'sẻ'], ['sã', 'sả'], ['sả', 'xả'], ['sẽ', 'sẻ'],
                             ['mùi', 'muồi'],
                             ['chỉnh', 'chỉn'], ['sữa', 'sửa'], ['chuẩn', 'chẩn'], ['lẻ', 'lẽ'], ['chẳng', 'chẵng'],
                             ['cổ', 'cỗ'],
                             ['sát', 'xát'], ['cập', 'cặp'], ['truyện', 'chuyện'], ['xá', 'sá'], ['giả', 'dả'],
                             ['đỡ', 'đở'],
                             ['giữ', 'dữ'], ['giã', 'dã'], ['xảo', 'sảo'], ['kiểm', 'kiễm'], ['cuộc', 'cục'],
                             ['dạng', 'dạn'],
                             ['tản', 'tảng'], ['ngành', 'nghành'], ['nghề', 'ngề'], ['nổ', 'nỗ'], ['rảnh', 'rãnh'],
                             ['sẵn', 'sẳn'],
                             ['sáng', 'xán'], ['xuất', 'suất'], ['suôn', 'suông'], ['sử', 'xử'], ['sắc', 'xắc'],
                             ['chữa', 'chửa'],
                             ['thắn', 'thắng'], ['dỡ', 'dở'], ['trải', 'trãi'], ['trao', 'trau'], ['trung', 'chung'],
                             ['thăm', 'tham'],
                             ['sét', 'xét'], ['dục', 'giục'], ['tả', 'tã'], ['sông', 'xông'], ['sáo', 'xáo'],
                             ['sang', 'xang'],
                             ['ngã', 'ngả'], ['xuống', 'suống'], ['xuồng', 'suồng']]

        self.vn_alphabet = ['a', 'ă', 'â', 'b', 'c', 'd', 'đ', 'e', 'ê', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'ô',
                            'ơ', 'p', 'q', 'r', 's', 't', 'u', 'ư', 'v', 'x', 'y']
        self.alphabet_len = len(self.vn_alphabet)
        self.char_couples = [['i', 'y'], ['s', 'x'], ['gi', 'd'],
                             ['ă', 'â'], ['ch', 'tr'], ['ng', 'n'],
                             ['nh', 'n'], ['ngh', 'ng'], ['ục', 'uộc'], ['o', 'u'],
                             ['ă', 'a'], ['o', 'ô'], ['ả', 'ã'], ['ổ', 'ỗ'], ['ủ', 'ũ'], ['ễ', 'ể'],
                             ['e', 'ê'], ['à', 'ờ'], ['ằ', 'à'], ['ẩn', 'uẩn'], ['ẽ', 'ẻ'], ['ùi', 'uồi'], ['ă', 'â'],
                             ['ở', 'ỡ'], ['ỹ', 'ỷ'], ['ỉ', 'ĩ'], ['ị', 'ỵ'],
                             ['ấ', 'á'], ['n', 'l'], ['qu', 'w'], ['ph', 'f'], ['d', 'z'], ['c', 'k'], ['qu', 'q'],
                             ['i', 'j'], ['gi', 'j'],
                             ]

        self.teencode_dict = {'mình': ['mk', 'mik', 'mjk'], 'vô': ['zô', 'zo', 'vo'], 'vậy': ['zậy', 'z', 'zay', 'za'],
                              'phải': ['fải', 'fai', ], 'biết': ['bit', 'biet'],
                              'rồi': ['rùi', 'ròi', 'r'], 'bây': ['bi', 'bay'], 'giờ': ['h', ],
                              'không': ['k', 'ko', 'khong', 'hk', 'hong', 'hông', '0', 'kg', 'kh', ],
                              'đi': ['di', 'dj', ], 'gì': ['j', ], 'em': ['e', ], 'được': ['dc', 'đc', ], 'tao': ['t'],
                              'tôi': ['t'], 'chồng': ['ck'], 'vợ': ['vk']

                              }

        # self.typo={"ă":"aw","â":"aa","á":"as","à":"af","ả":"ar","ã":"ax","ạ":"aj","ắ":"aws","ổ":"oor","ỗ":"oox","ộ":"ooj","ơ":"ow",
        #           "ằ":"awf","ẳ":"awr","ẵ":"awx","ặ":"awj","ó":"os","ò":"of","ỏ":"or","õ":"ox","ọ":"oj","ô":"oo","ố":"oos","ồ":"oof",
        #           "ớ":"ows","ờ":"owf","ở":"owr","ỡ":"owx","ợ":"owj","é":"es","è":"ef","ẻ":"er","ẽ":"ex","ẹ":"ej","ê":"ee","ế":"ees","ề":"eef",
        #           "ể":"eer","ễ":"eex","ệ":"eej","ú":"us","ù":"uf","ủ":"ur","ũ":"ux","ụ":"uj","ư":"uw","ứ":"uws","ừ":"uwf","ử":"uwr","ữ":"uwx",
        #           "ự":"uwj","í":"is","ì":"if","ỉ":"ir","ị":"ij","ĩ":"ix","ý":"ys","ỳ":"yf","ỷ":"yr","ỵ":"yj","đ":"dd",
        #           "Ă":"Aw","Â":"Aa","Á":"As","À":"Af","Ả":"Ar","Ã":"Ax","Ạ":"Aj","Ắ":"Aws","Ổ":"Oor","Ỗ":"Oox","Ộ":"Ooj","Ơ":"Ow",
        #           "Ằ":"AWF","Ẳ":"Awr","Ẵ":"Awx","Ặ":"Awj","Ó":"Os","Ò":"Of","Ỏ":"Or","Õ":"Ox","Ọ":"Oj","Ô":"Oo","Ố":"Oos","Ồ":"Oof",
        #           "Ớ":"Ows","Ờ":"Owf","Ở":"Owr","Ỡ":"Owx","Ợ":"Owj","É":"Es","È":"Ef","Ẻ":"Er","Ẽ":"Ex","Ẹ":"Ej","Ê":"Ee","Ế":"Ees","Ề":"Eef",
        #           "Ể":"Eer","Ễ":"Eex","Ệ":"Eej","Ú":"Us","Ù":"Uf","Ủ":"Ur","Ũ":"Ux","Ụ":"Uj","Ư":"Uw","Ứ":"Uws","Ừ":"Uwf","Ử":"Uwr","Ữ":"Uwx",
        #           "Ự":"Uwj","Í":"Is","Ì":"If","Ỉ":"Ir","Ị":"Ij","Ĩ":"Ix","Ý":"Ys","Ỳ":"Yf","Ỷ":"Yr","Ỵ":"Yj","Đ":"Dd"}
        self.typo = {"ă": ["aw", "a8"], "â": ["aa", "a6"], "á": ["as", "a1"], "à": ["af", "a2"], "ả": ["ar", "a3"],
                     "ã": ["ax", "a4"], "ạ": ["aj", "a5"], "ắ": ["aws", "ă1"], "ổ": ["oor", "ô3"], "ỗ": ["oox", "ô4"],
                     "ộ": ["ooj", "ô5"], "ơ": ["ow", "o7"],
                     "ằ": ["awf", "ă2"], "ẳ": ["awr", "ă3"], "ẵ": ["awx", "ă4"], "ặ": ["awj", "ă5"], "ó": ["os", "o1"],
                     "ò": ["of", "o2"], "ỏ": ["or", "o3"], "õ": ["ox", "o4"], "ọ": ["oj", "o5"], "ô": ["oo", "o6"],
                     "ố": ["oos", "ô1"], "ồ": ["oof", "ô2"],
                     "ớ": ["ows", "ơ1"], "ờ": ["owf", "ơ2"], "ở": ["owr", "ơ2"], "ỡ": ["owx", "ơ4"], "ợ": ["owj", "ơ5"],
                     "é": ["es", "e1"], "è": ["ef", "e2"], "ẻ": ["er", "e3"], "ẽ": ["ex", "e4"], "ẹ": ["ej", "e5"],
                     "ê": ["ee", "e6"], "ế": ["ees", "ê1"], "ề": ["eef", "ê2"],
                     "ể": ["eer", "ê3"], "ễ": ["eex", "ê3"], "ệ": ["eej", "ê5"], "ú": ["us", "u1"], "ù": ["uf", "u2"],
                     "ủ": ["ur", "u3"], "ũ": ["ux", "u4"], "ụ": ["uj", "u5"], "ư": ["uw", "u7"], "ứ": ["uws", "ư1"],
                     "ừ": ["uwf", "ư2"], "ử": ["uwr", "ư3"], "ữ": ["uwx", "ư4"],
                     "ự": ["uwj", "ư5"], "í": ["is", "i1"], "ì": ["if", "i2"], "ỉ": ["ir", "i3"], "ị": ["ij", "i5"],
                     "ĩ": ["ix", "i4"], "ý": ["ys", "y1"], "ỳ": ["yf", "y2"], "ỷ": ["yr", "y3"], "ỵ": ["yj", "y5"],
                     "đ": ["dd", "d9"],
                     "Ă": ["Aw", "A8"], "Â": ["Aa", "A6"], "Á": ["As", "A1"], "À": ["Af", "A2"], "Ả": ["Ar", "A3"],
                     "Ã": ["Ax", "A4"], "Ạ": ["Aj", "A5"], "Ắ": ["Aws", "Ă1"], "Ổ": ["Oor", "Ô3"], "Ỗ": ["Oox", "Ô4"],
                     "Ộ": ["Ooj", "Ô5"], "Ơ": ["Ow", "O7"],
                     "Ằ": ["AWF", "Ă2"], "Ẳ": ["Awr", "Ă3"], "Ẵ": ["Awx", "Ă4"], "Ặ": ["Awj", "Ă5"], "Ó": ["Os", "O1"],
                     "Ò": ["Of", "O2"], "Ỏ": ["Or", "O3"], "Õ": ["Ox", "O4"], "Ọ": ["Oj", "O5"], "Ô": ["Oo", "O6"],
                     "Ố": ["Oos", "Ô1"], "Ồ": ["Oof", "Ô2"],
                     "Ớ": ["Ows", "Ơ1"], "Ờ": ["Owf", "Ơ2"], "Ở": ["Owr", "Ơ3"], "Ỡ": ["Owx", "Ơ4"], "Ợ": ["Owj", "Ơ5"],
                     "É": ["Es", "E1"], "È": ["Ef", "E2"], "Ẻ": ["Er", "E3"], "Ẽ": ["Ex", "E4"], "Ẹ": ["Ej", "E5"],
                     "Ê": ["Ee", "E6"], "Ế": ["Ees", "Ê1"], "Ề": ["Eef", "Ê2"],
                     "Ể": ["Eer", "Ê3"], "Ễ": ["Eex", "Ê4"], "Ệ": ["Eej", "Ê5"], "Ú": ["Us", "U1"], "Ù": ["Uf", "U2"],
                     "Ủ": ["Ur", "U3"], "Ũ": ["Ux", "U4"], "Ụ": ["Uj", "U5"], "Ư": ["Uw", "U7"], "Ứ": ["Uws", "Ư1"],
                     "Ừ": ["Uwf", "Ư2"], "Ử": ["Uwr", "Ư3"], "Ữ": ["Uwx", "Ư4"],
                     "Ự": ["Uwj", "Ư5"], "Í": ["Is", "I1"], "Ì": ["If", "I2"], "Ỉ": ["Ir", "I3"], "Ị": ["Ij", "I5"],
                     "Ĩ": ["Ix", "I4"], "Ý": ["Ys", "Y1"], "Ỳ": ["Yf", "Y2"], "Ỷ": ["Yr", "Y3"], "Ỵ": ["Yj", "Y5"],
                     "Đ": ["Dd", "D9"]}
        self.all_word_candidates = self.get_all_word_candidates(self.word_couples)
        self.string_all_word_candidates = ' '.join(self.all_word_candidates)
        self.all_char_candidates = self.get_all_char_candidates()
        self.keyboardNeighbors = self.getKeyboardNeighbors()

    def replace_teencode(self, word):
        candidates = self.teencode_dict.get(word, None)
        if candidates is not None:
            chosen_one = 0
            if len(candidates) > 1:
                chosen_one = np.random.randint(0, len(candidates))
            return candidates[chosen_one]

    def getKeyboardNeighbors(self):
        keyboardNeighbors = {}
        keyboardNeighbors['a'] = "aáàảãạăắằẳẵặâấầẩẫậ"
        keyboardNeighbors['ă'] = "aáàảãạăắằẳẵặâấầẩẫậ"
        keyboardNeighbors['â'] = "aáàảãạăắằẳẵặâấầẩẫậ"
        keyboardNeighbors['á'] = "aáàảãạăắằẳẵặâấầẩẫậ"
        keyboardNeighbors['à'] = "aáàảãăắằẳẵâấầẩẫ"
        keyboardNeighbors['ả'] = "aảã"
        keyboardNeighbors['ã'] = "aáàảãạăắằẳẵặâấầẩẫậ"
        keyboardNeighbors['ạ'] = "aáàảãạăắằẳẵặâấầẩẫậ"
        keyboardNeighbors['ắ'] = "aáàảãạăắằẳẵặâấầẩẫậ"
        keyboardNeighbors['ằ'] = "aáàảãạăắằẳẵặâấầẩẫậ"
        keyboardNeighbors['ẳ'] = "aáàảãạăắằẳẵặâấầẩẫậ"
        keyboardNeighbors['ặ'] = "aáàảãạăắằẳẵặâấầẩẫậ"
        keyboardNeighbors['ẵ'] = "aáàảãạăắằẳẵặâấầẩẫậ"
        keyboardNeighbors['ấ'] = "aáàảãạăắằẳẵặâấầẩẫậ"
        keyboardNeighbors['ầ'] = "aáàảãạăắằẳẵặâấầẩẫậ"
        keyboardNeighbors['ẩ'] = "aáàảãạăắằẳẵặâấầẩẫậ"
        keyboardNeighbors['ẫ'] = "aáàảãạăắằẳẵặâấầẩẫậ"
        keyboardNeighbors['ậ'] = "aáàảãạăắằẳẵặâấầẩẫậ"
        keyboardNeighbors['b'] = "bh"
        keyboardNeighbors['c'] = "cgn"
        keyboardNeighbors['d'] = "đctơở"
        keyboardNeighbors['đ'] = "d"
        keyboardNeighbors['e'] = "eéèẻẽẹêếềểễệbpg"
        keyboardNeighbors['é'] = "eéèẻẽẹêếềểễệ"
        keyboardNeighbors['è'] = "eéèẻẽẹêếềểễệ"
        keyboardNeighbors['ẻ'] = "eéèẻẽẹêếềểễệ"
        keyboardNeighbors['ẽ'] = "eéèẻẽẹêếềểễệ"
        keyboardNeighbors['ẹ'] = "eéèẻẽẹêếềểễệ"
        keyboardNeighbors['ê'] = "eéèẻẽẹêếềểễệá"
        keyboardNeighbors['ế'] = "eéèẻẽẹêếềểễệố"
        keyboardNeighbors['ề'] = "eéèẻẽẹêếềểễệ"
        keyboardNeighbors['ể'] = "eéèẻẽẹêếềểễệôốồổỗộ"
        keyboardNeighbors['ễ'] = "eéèẻẽẹêếềểễệ"
        keyboardNeighbors['ệ'] = "eéèẻẽẹêếềểễệ"
        keyboardNeighbors['g'] = "qgộ"
        keyboardNeighbors['h'] = "h"
        keyboardNeighbors['i'] = "iíìỉĩịat"
        keyboardNeighbors['í'] = "iíìỉĩị"
        keyboardNeighbors['ì'] = "iíìỉĩị"
        keyboardNeighbors['ỉ'] = "iíìỉĩị"
        keyboardNeighbors['ĩ'] = "iíìỉĩị"
        keyboardNeighbors['ị'] = "iíìỉĩịhự"
        keyboardNeighbors['k'] = "klh"
        keyboardNeighbors['l'] = "ljidđ"
        keyboardNeighbors['m'] = "mn"
        keyboardNeighbors['n'] = "mnedư"
        keyboardNeighbors['o'] = "oóòỏọõôốồổỗộơớờởợỡ"
        keyboardNeighbors['ó'] = "oóòỏọõôốồổỗộơớờởợỡ"
        keyboardNeighbors['ò'] = "oóòỏọõôốồổỗộơớờởợỡ"
        keyboardNeighbors['ỏ'] = "oóòỏọõôốồổỗộơớờởợỡ"
        keyboardNeighbors['õ'] = "oóòỏọõôốồổỗộơớờởợỡ"
        keyboardNeighbors['ọ'] = "oóòỏọõôốồổỗộơớờởợỡ"
        keyboardNeighbors['ô'] = "oóòỏọõôốồổỗộơớờởợỡ"
        keyboardNeighbors['ố'] = "oóòỏọõôốồổỗộơớờởợỡ"
        keyboardNeighbors['ồ'] = "oóòỏọõôốồổỗộơớờởợỡ"
        keyboardNeighbors['ổ'] = "oóòỏọõôốồổỗộơớờởợỡ"
        keyboardNeighbors['ộ'] = "oóòỏọõôốồổỗộơớờởợỡ"
        keyboardNeighbors['ỗ'] = "oóòỏọõôốồổỗộơớờởợỡ"
        keyboardNeighbors['ơ'] = "oóòỏọõôốồổỗộơớờởợỡ"
        keyboardNeighbors['ớ'] = "oóòỏọõôốồổỗộơớờởợỡ"
        keyboardNeighbors['ờ'] = "oóòỏọõôốồổỗộơớờởợỡà"
        keyboardNeighbors['ở'] = "oóòỏọõôốồổỗộơớờởợỡ"
        keyboardNeighbors['ợ'] = "oóòỏọõôốồổỗộơớờởợỡ"
        keyboardNeighbors['ỡ'] = "oóòỏọõôốồổỗộơớờởợỡ"
        # keyboardNeighbors['p'] = "op"
        # keyboardNeighbors['q'] = "qọ"
        # keyboardNeighbors['r'] = "rht"
        # keyboardNeighbors['s'] = "s"
        # keyboardNeighbors['t'] = "tp"
        keyboardNeighbors['u'] = "uúùủũụưứừữửựhiaạt"
        keyboardNeighbors['ú'] = "uúùủũụưứừữửự"
        keyboardNeighbors['ù'] = "uúùủũụưứừữửự"
        keyboardNeighbors['ủ'] = "uúùủũụưứừữửự"
        keyboardNeighbors['ũ'] = "uúùủũụưứừữửự"
        keyboardNeighbors['ụ'] = "uúùủũụưứừữửự"
        keyboardNeighbors['ư'] = "uúùủũụưứừữửựg"
        keyboardNeighbors['ứ'] = "uúùủũụưứừữửự"
        keyboardNeighbors['ừ'] = "uúùủũụưứừữửự"
        keyboardNeighbors['ử'] = "uúùủũụưứừữửự"
        keyboardNeighbors['ữ'] = "uúùủũụưứừữửự"
        keyboardNeighbors['ự'] = "uúùủũụưứừữửựg"
        keyboardNeighbors['v'] = "v"
        keyboardNeighbors['x'] = "x"
        keyboardNeighbors['y'] = "yýỳỷỵỹụ"
        keyboardNeighbors['ý'] = "yýỳỷỵỹ"
        keyboardNeighbors['ỳ'] = "yýỳỷỵỹ"
        keyboardNeighbors['ỷ'] = "yýỳỷỵỹ"
        keyboardNeighbors['ỵ'] = "yýỳỷỵỹ"
        keyboardNeighbors['ỹ'] = "yýỳỷỵỹ"
        # keyboardNeighbors['w'] = "wv"
        # keyboardNeighbors['j'] = "jli"
        # keyboardNeighbors['z'] = "zs"
        # keyboardNeighbors['f'] = "ft"

        return keyboardNeighbors

    def replace_char_noaccent(self, text, onehot_label):

        # find index noise
        idx = np.random.randint(0, len(onehot_label))
        prevent_loop = 0
        while onehot_label[idx] == 1 or text[idx].isnumeric() or text[idx] in string.punctuation:
            idx = np.random.randint(0, len(onehot_label))
            prevent_loop += 1
            if prevent_loop > 10:
                return False, text, onehot_label

        index_noise = idx
        onehot_label[index_noise] = 1
        word_noise = text[index_noise]
        for id in range(0, len(word_noise)):
            char = word_noise[id]

            if char in self.keyboardNeighbors:
                neighbors = self.keyboardNeighbors[char]
                idx_neigh = np.random.randint(0, len(neighbors))
                replaced = neighbors[idx_neigh]
                word_noise = word_noise[: id] + replaced + word_noise[id + 1:]
                text[index_noise] = word_noise
                return True, text, onehot_label

        return False, text, onehot_label

    def replace_word_candidate(self, word):
        """
        Return a homophone word of the input word.
        """
        capital_flag = word[0].isupper()
        word = word.lower()
        if capital_flag and word in self.teencode_dict:
            return self.replace_teencode(word).capitalize()
        elif word in self.teencode_dict:
            return self.replace_teencode(word)

        for couple in self.word_couples:
            for i in range(2):
                if couple[i] == word:
                    if i == 0:
                        if capital_flag:
                            return couple[1].capitalize()
                        else:
                            return couple[1]
                    else:
                        if capital_flag:
                            return couple[0].capitalize()
                        else:
                            return couple[0]

    def replace_char_candidate(self, char):
        """
        return a homophone char/subword of the input char.
        """
        for couple in self.char_couples:
            for i in range(2):
                if couple[i] == char:
                    if i == 0:
                        return couple[1]
                    else:
                        return couple[0]

    def replace_char_candidate_typo(self, char):
        """
        return a homophone char/subword of the input char.
        """
        i = np.random.randint(0, 2)

        return self.typo[char][i]

    def get_all_char_candidates(self, ):

        all_char_candidates = []
        for couple in self.char_couples:
            all_char_candidates.extend(couple)
        return all_char_candidates

    def get_all_word_candidates(self, word_couples):

        all_word_candidates = []
        for couple in self.word_couples:
            all_word_candidates.extend(couple)
        return all_word_candidates

    def remove_diacritics(self, text, onehot_label):
        """
        Replace word which has diacritics with the same word without diacritics
        Args:
            text: a list of word tokens
            onehot_label: onehot array indicate position of word that has already modify, so this
            function only choose the word that do not has onehot label == 1.
        return: a list of word tokens has one word that its diacritics was removed,
                a list of onehot label indicate the position of words that has been modified.
        """
        idx = np.random.randint(0, len(onehot_label))
        prevent_loop = 0
        while onehot_label[idx] == 1 or text[idx] == unidecode.unidecode(text[idx]) or text[idx] in string.punctuation:
            idx = np.random.randint(0, len(onehot_label))
            prevent_loop += 1
            if prevent_loop > 10:
                return False, text, onehot_label

        onehot_label[idx] = 1
        text[idx] = unidecode.unidecode(text[idx])
        return True, text, onehot_label

    def replace_with_random_letter(self, text, onehot_label):
        """
        Replace, add (or remove) a random letter in a random chosen word with a random letter
        Args:
            text: a list of word tokens
            onehot_label: onehot array indicate position of word that has already modify, so this
            function only choose the word that do not has onehot label == 1.
        return: a list of word tokens has one word that has been modified,
                a list of onehot label indicate the position of words that has been modified.
        """
        idx = np.random.randint(0, len(onehot_label))
        prevent_loop = 0
        while onehot_label[idx] == 1 or text[idx].isnumeric() or text[idx] in string.punctuation:
            idx = np.random.randint(0, len(onehot_label))
            prevent_loop += 1
            if prevent_loop > 10:
                return False, text, onehot_label

        # replace, add or remove? 0 is replace, 1 is add, 2 is remove
        coin = np.random.choice([0, 1, 2])
        if coin == 0:
            chosen_letter = text[idx][np.random.randint(0, len(text[idx]))]
            replaced = self.vn_alphabet[np.random.randint(0, self.alphabet_len)]
            try:
                text[idx] = re.sub(chosen_letter, replaced, text[idx])
            except:
                return False, text, onehot_label
        elif coin == 1:
            chosen_letter = text[idx][np.random.randint(0, len(text[idx]))]
            replaced = chosen_letter + self.vn_alphabet[np.random.randint(0, self.alphabet_len)]
            try:
                text[idx] = re.sub(chosen_letter, replaced, text[idx])
            except:
                return False, text, onehot_label
        else:
            chosen_letter = text[idx][np.random.randint(0, len(text[idx]))]
            try:
                text[idx] = re.sub(chosen_letter, '', text[idx])
            except:
                return False, text, onehot_label

        onehot_label[idx] = 1
        return True, text, onehot_label

    def replace_with_homophone_word(self, text, onehot_label):
        """
        Replace a candidate word (if exist in the word_couple) with its homophone. if successful, return True, else False
        Args:
            text: a list of word tokens
            onehot_label: onehot array indicate position of word that has already modify, so this
            function only choose the word that do not has onehot label == 1.
        return: True, text, onehot_label if successful replace, else False, text, onehot_label
        """
        # account for the case that the word in the text is upper case but its lowercase match the candidates list
        candidates = []
        for i in range(len(text)):
            if text[i].lower() in self.all_word_candidates or text[i].lower() in self.teencode_dict.keys():
                candidates.append((i, text[i]))

        if len(candidates) == 0:
            return False, text, onehot_label

        idx = np.random.randint(0, len(candidates))
        prevent_loop = 0
        while onehot_label[candidates[idx][0]] == 1:
            idx = np.random.choice(np.arange(0, len(candidates)))
            prevent_loop += 1
            if prevent_loop > 5:
                return False, text, onehot_label

        text[candidates[idx][0]] = self.replace_word_candidate(candidates[idx][1])
        onehot_label[candidates[idx][0]] = 1
        return True, text, onehot_label

    def replace_with_homophone_letter(self, text, onehot_label):
        """
        Replace a subword/letter with its homophones
        Args:
            text: a list of word tokens
            onehot_label: onehot array indicate position of word that has already modify, so this
            function only choose the word that do not has onehot label == 1.
        return: True, text, onehot_label if successful replace, else False, None, None
        """
        candidates = []
        for i in range(len(text)):
            for char in self.all_char_candidates:
                if re.search(char, text[i]) is not None:
                    candidates.append((i, char))
                    break

        if len(candidates) == 0:

            return False, text, onehot_label
        else:
            idx = np.random.randint(0, len(candidates))
            prevent_loop = 0
            while onehot_label[candidates[idx][0]] == 1:
                idx = np.random.randint(0, len(candidates))
                prevent_loop += 1
                if prevent_loop > 5:
                    return False, text, onehot_label

            replaced = self.replace_char_candidate(candidates[idx][1])
            text[candidates[idx][0]] = re.sub(candidates[idx][1], replaced, text[candidates[idx][0]])

            onehot_label[candidates[idx][0]] = 1
            return True, text, onehot_label

    def replace_with_typo_letter(self, text, onehot_label):
        """
        Replace a subword/letter with its homophones
        Args:
            text: a list of word tokens
            onehot_label: onehot array indicate position of word that has already modify, so this
            function only choose the word that do not has onehot label == 1.
        return: True, text, onehot_label if successful replace, else False, None, None
        """
        # find index noise
        idx = np.random.randint(0, len(onehot_label))
        prevent_loop = 0
        while onehot_label[idx] == 1 or text[idx].isnumeric() or text[idx] in string.punctuation:
            idx = np.random.randint(0, len(onehot_label))
            prevent_loop += 1
            if prevent_loop > 10:
                return False, text, onehot_label

        index_noise = idx
        onehot_label[index_noise] = 1

        word_noise = text[index_noise]
        for j in range(0, len(word_noise)):
            char = word_noise[j]

            if char in self.typo:
                replaced = self.replace_char_candidate_typo(char)
                word_noise = word_noise[: j] + replaced + word_noise[j + 1:]
                text[index_noise] = word_noise
                return True, text, onehot_label
        return True, text, onehot_label

    def add_noise(self, sentence, percent_err=0.15, num_type_err=5):
        tokens = self.tokenizer(sentence)
        onehot_label = [0] * len(tokens)

        num_wrong = int(np.ceil(percent_err * len(tokens)))
        num_wrong = np.random.randint(1, num_wrong + 1)
        if np.random.rand() < 0.05:
            num_wrong = 0

        for i in range(0, num_wrong):
            err = np.random.randint(0, num_type_err + 1)

            if err == 0:
                _, tokens, onehot_label = self.replace_with_homophone_letter(tokens, onehot_label)
            elif err == 1:
                _, tokens, onehot_label = self.replace_with_typo_letter(tokens, onehot_label)
            elif err == 2:
                _, tokens, onehot_label = self.replace_with_homophone_word(tokens, onehot_label)
            elif err == 3:
                _, tokens, onehot_label = self.replace_with_random_letter(tokens, onehot_label)
            elif err == 4:
                _, tokens, onehot_label = self.remove_diacritics(tokens, onehot_label)
            elif err == 5:
                _, tokens, onehot_label = self.replace_char_noaccent(tokens, onehot_label)
            else:
                continue
            # print(tokens)
        return ' '.join(tokens)

