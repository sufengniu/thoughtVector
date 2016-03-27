from __future__ import division
from __future__ import unicode_literals
import os
import collections
import cPickle
import numpy as np
import json
import urllib
import argparse
import sys
from topia.termextract import extract
import StringIO


class K_Engine():

    def __init__(self,xdata_dir,ydata_dir,check_dir):
        self.xdata_dir = xdata_dir
        self.ydata_dir = ydata_dir
        self.check_dir = check_dir
        self.target_list = []
        if not (os.path.exists(self.xdata_dir) and os.path.exists(self.ydata_dir)):
            print "Creating new knowledge description"
        else:
            print "reading data from exsiting file"
            print "checking file integrity"
            with open(self.xdata_dir, 'r') as xf:
                xline = 0
                for line in xf:
                    xline += 1
            with open(self.ydata_dir, 'r') as yf:
                yline = 0
                for line in yf:
                    yline += 1
            with open(self.check_dir, 'r') as cf:
                cline = 0
                for line in cf:
                    cline += 1
            if xline == yline == cline:
                print "checking file integrity finished"
                print "initializing target file"
                with open(check_dir, 'r') as cf:
                    for line in cf:
                        self.target_list.append(line)
                    cf.closed
                print len(self.target_list), 'entites currently included.'
            else:
                raise Exception("error: wrong line_number. Make sure both files have the same line nubmer.")


    def read_knowledge_graph(self, target, limit=20):
        api_key = open('.api_key').read()
        query = target
        service_url = 'https://kgsearch.googleapis.com/v1/entities:search'
        params = {
            'query': query.encode('utf-8'),
            'limit': limit,
            'indent': True,
            'key': api_key,
        }
        url = service_url + '?' + urllib.urlencode(params)
        response = json.loads(urllib.urlopen(url).read())
        container={}
        for element in response['itemListElement']:
            if element['result'].get('detailedDescription'):
                container[element['result']['@id']]=[element['result']['name'],
                element['result']['detailedDescription']['articleBody'].replace('\n', ' ')]
            elif element['result'].get('description'):
                container[element['result']['@id']]=[element['result']['name'], element['result']['description']]
        return container

    def keyextractor(self, sentence):
        """
        extract noun from sentence
        """
        extractor = extract.TermExtractor()
        temp = extractor.tagger(sentence)
        result = []
        for key in temp:
            if key[1] == 'NN' or key[1] == 'NNP':
                result.append(key[0])
        return result

    def description_extractor(self, target, layer=1):
        if layer == 20:
            return
        else:
            knowledge=self.read_knowledge_graph(target)
            with open(self.xdata_dir,'a') as xf:
                with open(self.ydata_dir,'a') as yf:
                    with open(self.check_dir,'a') as cf:
                        for element in knowledge:
                            if element not in self.target_list:
                                # self.target_list.append(element['result']['@id'])
                                cf.write(element.encode('utf-8'))
                                cf.write('\n')
                                xf.write(knowledge[element][1].encode('utf-8'))
                                xf.write('\n')
                                yf.write(knowledge[element][0].encode('utf-8'))
                                yf.write('\n')
                        cf.closed
                    yf.closed
                xf.closed
            for element in knowledge:
                if element not in self.target_list:
                    self.target_list.append(element.encode('utf-8'))
                    sys.stdout.write(str(len(self.target_list))+'entities included!'+'\r')
                    print 
                    for key in self.keyextractor(knowledge[element][1]):
                        self.description_extractor(key, layer+1)


                        ##todo:: repeat checking    file writing
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--x_dir', type=str, default='data/xdata',
                       help='xdata_dir to store entity description')
    parser.add_argument('--y_dir', type=str, default='data/ydata',
                       help='ydata_dir to store entity name')
    parser.add_argument('--c_dir', type=str, default='data/cdata',
                       help='check_dir to store entity ID')
    parser.add_argument('--t_dir', type=str, default='data/tdata',
                       help='target_dir to store a target list')
    parser.add_argument('--target', type=str, default='apple',
                       help='a single target')
    args = parser.parse_args()
    engine = K_Engine(args.x_dir,args.y_dir,args.c_dir)
    engine.description_extractor(args.target)


if __name__ == '__main__':
    main()