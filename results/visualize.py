import json
import pandas as pd
import argparse


def visualize(json_file):

    with open(json_file,'r') as jsFile:
        data = json.load(jsFile)
    
    ##输出BLEU测试分数
    print('BLEU 1-4',data['bleu'])

    ##输出图片与对应的句子
    for result in data['results']:
        image = result['img_path']
        prediction = result['prediction']
        reference = result['references']
        print(image,'\t',prediction)
        for raw in reference:
            print(raw)
        print('\n')
    
    df = pd.DataFrame(data['results'])
    #print(df)
    
    results = data["results"]
    
    with open('../vis/vis.json', 'w+') as file:
        file.write(json.dumps(results))
        
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
        
    with open('gts.json', 'w+') as file:
        file.write(json.dumps(gts[0]))
    
    with open('res.json', 'w+') as file:
        file.write(json.dumps(res[0]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='visualize data',add_help=True)
    parser.add_argument('-j','--json',default='RSICD_DenseNet169_test_withpath_process.json',help='json config file')
    args = parser.parse_args()
    #print(args.json)
    visualize(args.json)
    pre_eval(args.json)
