from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
import json
import pandas as pd
import argparse

def bleu():
    scorer = Bleu(n=4)
    # scorer += (hypo[0], ref1)   # hypo[0] = 'word1 word2 word3 ...'
    #                                 # ref = ['word1 word2 word3 ...', 'word1 word2 word3 ...']
    score, scores = scorer.compute_score(gts, res)

    print('belu = %s' % score)

def cider():
    scorer = Cider()
    # scorer += (hypo[0], ref1)
    (score, scores) = scorer.compute_score(gts, res)
    print('cider = %s' % score)

def meteor():
    scorer = Meteor()
    score, scores = scorer.compute_score(gts, res)
    print('meter = %s' % score)

def rouge():
    scorer = Rouge()
    score, scores = scorer.compute_score(gts, res)
    print('rouge = %s' % score)

def spice():
    scorer = Spice()
    score, scores = scorer.compute_score(gts, res)
    print('spice = %s' % score)

def score():
    bleu()
    cider()
    rouge()
    meteor()
    #spice()

def pre_eval(json_file):
    
    gts = []
    res = []
    gts_dic = {}
    res_dic = {}
    
    with open(json_file,'r') as jsFile:
        data = json.load(jsFile)
    
    for result in data['results']:
        image = result['img_id']
        prediction = result['prediction']
        reference = result['references']
        
        gts_dic[image]=reference
        res_dic[image]=[prediction]
        
    gts.append(gts_dic)
    
    res.append(res_dic)
        
    with open('./examples/gts.json', 'w+') as file:
        file.write(json.dumps(gts[0]))
    
    with open('./examples/res.json', 'w+') as file:
        file.write(json.dumps(res[0]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='visualize data',add_help=True)
    parser.add_argument('-j','--json',default='./results/RSICD_DenseNet169_test_withpath_process.json',help='json config file')
    args = parser.parse_args()
    pre_eval(args.json)
    
    with open('examples/gts.json', 'r') as file:
        gts = json.load(file)
    with open('examples/res.json', 'r') as file:
        res = json.load(file)
    score()