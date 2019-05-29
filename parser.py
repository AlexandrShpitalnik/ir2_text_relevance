# -*- coding: utf-8 -*-
import pickle
import os
from pymystem3 import Mystem
#from boilerpipe.extract import Extractor
from inscriptis import get_text
from bs4 import BeautifulSoup
import re


class IR_Extractor:
    def __init__(self, path):
        self.m = Mystem()
        self.path = path
        self.max_size = 2000000
        self.min_lems = 20
        self.clear = [u'ё', u'й', u'ц', u'у', u'к', u'е', u'н', u'г', u'ш', u'щ', u'з', u'х', u'ъ', u'ф', u'ы', u'в',
                      u'а', u'п', u'р', u'о', u'л', u'д', u'ж', u'э', u'я', u'ч', u'с', u'м', u'и', u'т', u'ь', u'б',
                      u'ю', u'0', u'1', u'2', u'3', u'4', u'5', u'6', u'7', u'8', u'9', u'ә', u'ғ', u'қ', u'ң', u'ө',
                      u'ұ', u'ү', u'h', u'і', u'ґ', u'є', u'і', u'ї', u'q', u'w', u'e', u'r', u't', u'y', u'u', u'i',
                      u'o', u'p', u'a', u's', u'd', u'f', u'g', u'h', u'j', u'k', u'l', u'z', u'x', u'c', u'v', u'b',
                      u'n', u'm']
        self.clear_d = {}
        for l in self.clear:
            self.clear_d[l] = 1

    def set_name(self, name):
        self.dmp_name = name

    def get_content(self, html_name, mode):
        f = open(html_name)
        url = f.readline()[:-1]
        if (os.fstat(f.fileno()).st_size > self.max_size):
            print('B')
            f.close()
            return url, [[], []]

        html_rest = f.read()

        if mode == 'BS':
            soup = BeautifulSoup(html_rest, "html.parser")
            [s.extract() for s in soup(['style', 'script', '[document]', 'head', 'title'])]
            html_text = soup.get_text()
            headers = self.get_headers(html_rest)
            nolem = self.clean(html_text)
            lem = self.lemmatize(nolem)
            val = [lem, headers]

        else:
            extractor = Extractor(extractor='DefaultExtractor', html=html_rest)
            html_text = extractor.getText()
            nolem = self.clean(html_text)
            lem = self.lemmatize(nolem)
            if len(lem) < self.min_lems:
                html_text = get_text(html_rest)
                html_text = self.get_outer_fields(html_text)
                nolem = self.clean(html_text)
                lem = self.lemmatize(nolem)
            val = [lem]

        f.close()
        return url, val

    def get_outer_fields(self, extr):
        outer_lines = u''
        newline_flag = True
        lr_flag = False
        for w in extr:
            if (w == u'\n'):
                if (not lr_flag):
                    outer_lines += u' '
                newline_flag = True
                lr_flag = False
            elif (w == u' ' and newline_flag) or lr_flag:
                lr_flag = True
                newline_flag = False
            else:
                newline_flag = False
                outer_lines += w
        return outer_lines

    def get_headers(self, html):
        res = []
        headers = re.findall('<title>.+</title>', html)
        for header in headers:
            try:
                header = header[7:-8].decode('utf-8')
            except:
                try:
                    header = header[7:-8].decode('cp1251')
                except:
                    header = header[7:-8]
            nolem = self.clean(header)
            lem = self.lemmatize(nolem)
            res.append(lem)
        return res

    def clean(self, string):
        string = string.lower()
        res = ''
        sp_flag = True
        for w in string:
            if not (w in self.clear_d):
                if (not sp_flag):
                    res += u' '
                    sp_flag = True
            else:
                res += w
                sp_flag = False
        return res

    def lemmatize(self, string):
        lem_res = self.m.lemmatize(string)
        lem_res_fin = []
        for l in lem_res:
            if l != u' ' and l != u'\n' and l != u' \n':
                lem_res_fin.append(l)
        return lem_res_fin

    def add_content(self, html_name, mode):
        html_name = self.path + '/' + html_name
        key = html_name
        value = [[], []]
        try:
            key, value = self.get_content(html_name, mode)
        except:
            print('ERR')
        self.storage[key] = value
        print('@') #####

    def dump(self):
        with open(self.dmp_name, 'wb') as f:
            pickle.dump(self.storage, f)

    def main_func(self, start, stop, mode='BS'):
        self.storage = {}
        file_name_start = 'doc.'
        file_name_end = '.dat'

        for i in range(start, stop):
            num = '%05d' % i
            f = file_name_start + num + file_name_end
            self.add_content(f, mode)

        self.dump()
        print("DONE")
