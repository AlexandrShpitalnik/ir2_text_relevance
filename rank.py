import pickle
import numpy as np
import math
import heapq
import pandas as pd
from pymystem3 import Mystem

class Yandex_CF:
    def __init__(self, k1=1.0, k2=0.00285, single_koef=1.0, pair_koef=0.3, all_koef=0.2,
                 phrase_koef=0.1, titles_koef=2.0, title_one=1.0, title_pair=0.3, title_all=0.2,
                 title_phrase=0.1, start_koef=1.0, start_single=1.0, start_pair=0.3,
                 start_all=0.1, start_phrase = 0.1, alpha=4.1, beta=1.0, dist_koef=0.2, b_len=30,
                 n_lems = 121873290.0, lem_file='CF.pickle'):
        f = open(lem_file, 'rb')
        self.CF = pickle.load(f)
        f.close()
        self.k1 = k1
        self.k2 = k2
        self.total_lem = n_lems
        self.doc_num = 74258
        self.single_koef = single_koef
        self.pair_koef = pair_koef
        self.all_koef = all_koef
        self.phrase_koef = phrase_koef
        self.dist_koef = dist_koef
        self.titles_koef = titles_koef
        self.title_one = title_one
        self.title_pair = title_pair
        self.title_all = title_all
        self.title_phrase = title_phrase
        self.start_koef = start_koef
        self.start_single = start_single
        self.start_pair = start_pair
        self.start_all = start_all
        self.start_phrase = start_phrase
        self.alpha = alpha
        self.beta = beta
        self.b_len = b_len
        self.eps = 0.001

    def _get_word_doc_score(self, word, doc):
        cf = self.CF.get(word, 0.001)
        p = (self.total_lem) / cf
        pos = []
        self.single_word_p.append(p)
        ICF = np.log(p)
        TF = 0.0
        DL = len(doc)
        for i in range(DL):
            if word == doc[i]:
                pos.append(i)
                TF += 1.0

        if TF == 0.0:
            self.unk_mask.append(1)
        else:
            self.unk_mask.append(0)
            self.positions.append(pos)

        mul1 = TF / (TF + self.k1 + self.k2 * DL)
        score = mul1 * ICF
        return score

    def set_query(self, query):
        self.query_words = query

    def _skip(self, doc, title):
        if title is None:
            l = 0
        else:
            l = len(title)
        for i in range(len(title)):
            if title[i] != doc[i]:
                return doc[i:i + self.b_len]
        return doc[l:l + self.b_len]

    def get_doc_score(self, doc, titles_list, bdoc):
        self.positions = []
        self.single_word_p = []
        self.unk_mask = []
        self.boiler_doc = bdoc
        titles = []
        for t in titles_list:
            for w in t:
                titles.append(w)

        if doc == [] or len(doc) < 20:
            return -1000, [0, 0, 0, 0]
        if (doc[-1] == ' '):
            doc = doc[:-1]

        #begining
        if len(titles_list) > 0 and len(self.boiler_doc) - len(titles_list[0]) >= self.b_len:
            begining = self._skip(self.boiler_doc, titles_list[0])
            start_s = self._get_single_score(begining)
            start_p = self._get_pairs_score(begining)
            start_a = self._get_all_score()
            start_ph = self._get_phrase_score(begining)

        elif len(titles_list) == 0 and len(self.boiler_doc) >= self.b_len:
            start_s = self._get_single_score(self.boiler_doc[:self.b_len])
            start_p = self._get_pairs_score(self.boiler_doc[:self.b_len])
            start_a = self._get_all_score()
            start_ph = self._get_phrase_score(self.boiler_doc[:self.b_len])
        else:
            start_s = 0.0
            start_p = 0.0
            start_a = 0.0
            start_ph = 0.0

        start_score = (self.start_single * start_s + self.start_pair * start_p + self.start_all * start_a +
                       self.start_phrase * start_ph) * self.start_koef

        self.positions = []
        self.single_word_p = []
        self.unk_mask = []

        #full document
        sscore = self.single_koef * self._get_single_score(doc)
        pscore = self.pair_koef * self._get_pairs_score(doc)
        ascore = self.all_koef * self._get_all_score()
        phscore = self.phrase_koef * self._get_phrase_score(doc)

        full_score = sscore + pscore + ascore + phscore

        #document title
        if len(titles) > 0:
            self.unk_mask = []
            t_single_score = self.title_one * self._get_single_score(titles)
            if len(titles) > 1:
                t_pair_score = self.title_pair * self._get_pairs_score(titles)
                t_all_score = self.title_all * self._get_all_score()
                t_phrase_score = self.title_phrase * self._get_phrase_score(titles)
            else:
                t_pair_score = 0.0
                t_all_score = 0.0
                t_phrase_score = 0.0

            tscore = (t_single_score + t_pair_score + t_all_score + t_phrase_score) * self.titles_koef
        else:
            tscore = 0.0
            t_single_score = 0.0
            t_pair_score = 0.0
            t_all_score = 0.0
            t_phrase_score = 0.0

        self.positions = []
        self.unk_mask = []
        self._get_single_score(doc)

        if len(self.positions) > 0:
            dscore = self._get_dist_score()
            if math.isnan(dscore):
                dscore = self.eps
        else:
            dscore = self.eps
        dscore += self.dist_koef

        res = (full_score + tscore + start_score) * dscore
        return res, [sscore, tscore, pscore, ascore, phscore, t_single_score, t_pair_score,
                     t_all_score, t_phrase_score, dscore, start_s, start_p, start_a, start_ph]

    def _get_single_score(self, doc):
        score = 0.0
        for word in self.query_words:
            w_score = self._get_word_doc_score(word, doc)
            score += w_score
        return score

    def _get_pairs_score(self, doc):
        score = 0.0
        for i in range(len(self.query_words) - 1):
            score += self._get_one_pair_score(i, doc)
        return score

    def _get_one_pair_score(self, i, doc):
        bonus = 0.0
        w1 = self.query_words[i]
        w2 = self.query_words[i + 1]
        p1 = self.single_word_p[i]
        p2 = self.single_word_p[i + 1]
        if i != 0:
            w3 = self.query_words[i - 1]
            p3 = self.single_word_p[i - 1]
            btf = self._get_bonus_TF(w3, w2, doc)
            if btf != 0:
                bonus = (np.log(p3) + np.log(p2)) * (btf / (btf + 1.0))
        tf = self._get_TF(w1, w2, doc)
        score = (np.log(p1) + np.log(p2)) * (tf / (tf + 1.0))
        return score + bonus

    def _get_TF(self, w1, w2, doc):
        count = 0.0
        for i in range(len(doc) - 2):
            if w1 == doc[i] and w2 == doc[i + 1]:
                count += 1.0
            elif w2 == doc[i] and w1 == doc[i + 1]:
                count += 0.5
            elif w1 == doc[i] and w2 == doc[i + 2]:
                count += 0.5
        if w1 == doc[-2] and w2 == doc[-1]:
            count += 1.0
        elif w2 == doc[-2] and w1 == doc[-1]:
            count += 0.5
        return count

    def _get_bonus_TF(self, w1, w2, doc):
        count = 0.0
        for i in range(len(doc) - 1):
            if w1 == doc[i] and w2 == doc[i + 1]:
                count += 0.1
        return count

    def _get_all_score(self):
            num_unk = sum(self.unk_mask)
            koef = 0.03 ** num_unk
            score = 1.0
            for i in range(len(self.query_words)):
                if not (self.unk_mask[i] == 1):
                    p = self.single_word_p[i]
                    score += np.log(p)
            score *= koef
            return score

    def _get_phrase_score(self, doc):
        query_l = len(self.query_words)
        if sum(self.unk_mask) != 0:
            return 0.0
        tf = 0.0
        for i in range(0, len(doc) - query_l):
            if self.query_words == doc[i:i + query_l]:
                tf += 1.0
        score = 0.0
        for i in range(query_l):
            p = self.single_word_p[i]
            score += np.log(p)
        score *= (tf / (tf + 1))
        return score

    def _get_min_dist(self):
        queue = []
        # node = [first_pos, last_pos, lvl]
        max_lvl = len(self.positions) - 2

        for pos1 in self.positions[0]:
            is_pushed_flag = True
            for pos2_n in range(len(self.positions[1])):
                if self.positions[1][pos2_n] > pos1:
                    is_pushed_flag = False
                    pos2 = self.positions[1][pos2_n]
                    heapq.heappush(queue, (abs(pos1 - pos2), [min(pos1, pos2), max(pos1, pos2), 0]))
                    if pos2_n != 0:
                        pos2 = self.positions[1][pos2_n - 1]
                        heapq.heappush(queue, (abs(pos1 - pos2), [min(pos1, pos2), max(pos1, pos2), 0]))
                    break
            if is_pushed_flag:
                pos2 = self.positions[1][-1]
                heapq.heappush(queue, (abs(pos1 - pos2), [min(pos1, pos2), max(pos1, pos2), 0]))
                break

        while True:
            item = heapq.heappop(queue)
            is_pushed_flag = False
            window_size = item[0]
            node = item[1]
            lvl = node[2]
            if lvl < max_lvl:

                for pos in self.positions[lvl + 2]:
                    if pos >= node[0] and pos <= node[1]:
                        heapq.heappush(queue, (window_size, [node[0], node[1], lvl + 1]))
                        is_pushed_flag = True
                        break
                if not is_pushed_flag:
                    for pos_n in range(len(self.positions[lvl + 2])):
                        pos = self.positions[lvl + 2][pos_n]
                        if pos > node[1]:
                            heapq.heappush(queue, (window_size + abs(pos - node[1]), [node[0], pos, lvl + 1]))
                            if pos_n != 0:
                                pos = self.positions[lvl + 2][pos_n - 1]
                                heapq.heappush(queue, (window_size + abs(pos - node[0]), [pos, node[1], lvl + 1]))
                            is_pushed_flag = True
                            break
                if not is_pushed_flag:
                    pos = self.positions[lvl + 2][-1]
                    heapq.heappush(queue, (window_size + abs(pos - node[0]), [pos, node[1], lvl + 1]))
            else:
                return window_size + 1

    def _get_dist_score(self):
        if len(self.positions) == 1:
            m = 1
        else:
            m = self._get_min_dist()

        n = 0
        S_q = 0.0
        S_n = 0.0

        for i in range(len(self.unk_mask)):
            if self.unk_mask[i] == 0:
                n += 1.0
                p = self.single_word_p[i]
                S_n += np.log(p)

        for w in self.query_words:
            cf = self.CF.get(w, self.total_lem)
            p = (self.total_lem) / cf
            S_q += np.log(p)

        d = np.log(m - n + self.alpha)
        mul1 = np.log(self.alpha) / d
        mul2 = S_n / (S_q + self.beta * (S_q - S_n))
        score = mul1 * mul2
        return score


class DB:
    def __init__(self, bs_dir, bp_dir, url_file):

        file_names = ['dmp0k-5k.pickle', 'dmp5k-10k.pickle',
                        'dmp10k-15k.pickle', 'dmp15k-20k.pickle', 'dmp20k-25k.pickle', 'dmp25k-30k.pickle',
                        'dmp30k-35k.pickle', 'dmp35k-40k.pickle', 'dmp40k-45k.pickle', 'dmp45k-50k.pickle',
                        'dmp50k-55k.pickle', 'dmp55k-60k.pickle', 'dmp60k-65k.pickle', 'dmp65k-70k.pickle',
                        'dmp70k-end.pickle']

        self.border = [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000,
                         60000, 65000, 70000, 75000]

        self.map_list_bp = []
        self.map_list_bs = []


        # url to file number map 'url_file.pickle'
        f = open(url_file, 'rb')
        self.url_file_map = pickle.load(f)
        f.close()

        for name in file_names:
            with open(bp_dir +'/' + name, 'rb') as f:
                content_split = pickle.load(f)
                self.map_list_bp.append(content_split)

            with open(bs_dir +'/' + name, 'rb') as f:
                content_split = pickle.load(f)
                self.map_list_bs.append(content_split)

    def _get_idx(self, url):
        file_num = self.url_file_map[url]
        for i in range(len(self.border)):
            if file_num < self.border[i]:
                return i
        return -1

    def get_content(self, url, mode):
        idx = self._get_idx(url)
        if idx > 0:
            if mode == 'bs':
                split = self.map_list_bs[idx]
            elif mode == 'bp':
                split = self.map_list_bp[idx]
            content = split.get(url, [[], []])
            return content[0]
        return []

    def get_header(self, url):
        idx = self._get_idx(url)
        if idx > 0:
            split = self.map_list_bs[idx]
            content = split.get(url, [[], []])
            return content[1]
        return []


class IR_rank:
    def __init__(self, algo, db, stop_list, sample='sample_submission.txt', queries_file='queries_new.txt',
                 urls_file='urls.numerate.txt'):
        self.number_url_map = {}
        self.number_query_map = {}
        self.stop = {}
        self.max_q_len = 45
        self.m = Mystem()
        self.db = db

        self.global_rank_res = []
        self.global_docs_scores = []
        self.global_query_stats = []

        self.df_sample_submission = pd.read_csv(sample, sep=',')
        self.algo = algo

        for w in stop_list:
            st = self.m.lemmatize(w)[0]
            if st not in self.stop:
                self.stop[st] = 1

        with open(queries_file) as f:
            for line in f:
                num_query = line[:-1].split('\t')
                self.number_query_map[int(num_query[0])] = num_query[1]

        with open(urls_file) as f:
            for line in f:
                num_url = line[:-1].split('\t')
                self.number_url_map[int(num_url[0])] = num_url[1]


    def lemmatize(self, string):
        lem_res = self.m.lemmatize(string)
        lem_res_fin = []
        for l in lem_res:
            if l != u' ' and l != u'\n' and l != u' \n':
                lem_res_fin.append(l)
        return lem_res_fin

    def delete_stop(self, doc):
        res = []
        for w in doc:
            if w not in self.stop:
                res.append(w)
        return res

    def main(self):
        for i in range(1, len(self.number_query_map.keys()) + 1):
            query = self.number_query_map[i]
            query = query.decode('utf-8')

            df_candidates = self.df_sample_submission.loc[self.df_sample_submission['QueryId'] == i]
            candidates_list_order = df_candidates["DocumentId"].values

            query_words = self.delete_stop(self.lemmatize(query))

            if len(query_words) == 0 or len(query_words) > self.max_q_len:
                self.global_rank_res.append(candidates_list_order)
                self.global_docs_scores.append(candidates_list_order)
                self.global_query_stats.append(candidates_list_order)
                continue

            if query_words[-1] == ' ':
                query_words = query_words[:-1]
            self.algo.set_query(query_words)
            query_result = []  # (doc_id;score;stat)
            for doc_id in candidates_list_order:
                url = self.number_url_map[doc_id]
                content = self.delete_stop(self.db.get_content(url, 'bs'))
                bp_content = self.delete_stop(self.db.get_content(url, 'bp'))
                titles_list = self.db.get_header(url)
                for i in range(len(titles_list)):
                    titles_list[i] = self.delete_stop(titles_list[i])
                score, stat = self.algo.get_doc_score(content, titles_list, bp_content)
                query_result.append((doc_id, score, stat))

            query_result_sorted = sorted(query_result, key=lambda x: -x[1] if x[1] > -999 else 0)

            print('$')

            docs_order = list(map(lambda a: a[0], query_result_sorted))
            docs_scores = list(map(lambda a: a[1], query_result_sorted))
            query_stat = list(map(lambda a: a[2], query_result_sorted))
            self.global_rank_res.append(docs_order)
            self.global_docs_scores.append(docs_scores)
            self.global_query_stats.append(query_stat)


    def dump(self, name, mode='subm'):
        with open(name, 'wb') as f:
            f.write("QueryId,DocumentId\n")
            for cur_query_num in range(len(self.global_rank_res)):
                for i in range(len(self.global_rank_res[cur_query_num])):
                    doc_id = self.global_rank_res[cur_query_num][i]
                    score = self.global_docs_scores[cur_query_num][i]
                    stat = str(self.global_query_stats[cur_query_num][i])
                    if mode == 'stat':
                        f.write(str(cur_query_num + 1) + ',' + str(doc_id) + ',' + str(score) + ' | ' + str(stat) + '\n')
                    elif mode == 'subm':
                        f.write(str(cur_query_num + 1) + ',' + str(doc_id) + '\n')

class CF_counter:
    def __init__(self, parsed_dir):
        self.CF = {}
        self.dir = parsed_dir +'/'
        self.files = ['dmp0k-5k.pickle', 'dmp5k-10k.pickle',
                      'dmp10k-15k.pickle', 'dmp15k-20k.pickle', 'dmp20k-25k.pickle', 'dmp25k-30k.pickle',
                      'dmp30k-35k.pickle', 'dmp35k-40k.pickle', 'dmp40k-45k.pickle', 'dmp45k-50k.pickle',
                      'dmp50k-55k.pickle', 'dmp55k-60k.pickle', 'dmp60k-65k.pickle', 'dmp65k-70k.pickle',
                      'dmp70k-end.pickle']
        self.total_lemms = 0.0

    def count(self):
        for file_name in self.files:
            print '@'
            f = open(self.dir + file_name, 'rb')
            file_d = pickle.load(f)
            items = file_d.items()
            for item in items:
                self._parse_item(item)
            f.close()

        f_out = open('CF.pickle', 'wb')
        pickle.dump(self.CF, f_out)
        f_out.close()
        return self.total_lemms

    def _parse_item(self, it):
        if it[1] != [[], []]:
            words = it[1][0]
            self.total_lemms += len(words)
            for word in words:
                if word != ' ':
                    if word in self.CF:
                        self.CF[word] += 1
                    else:
                        self.CF[word] = 1
