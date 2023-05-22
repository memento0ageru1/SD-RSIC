import json
import pandas as pd
import argparse
import os


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
        #print(image,'\t',prediction)
        #for raw in reference:
        #    print(raw)
        #print('\n')
    
    df = pd.DataFrame(data['results'])
    #print(df)
    
    results = data["results"]
    
    with open('./vis/vis.json', 'w+') as file:
        file.write(json.dumps(results))
      
def html():
    os.system('python -m http.server')
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='visualize data',add_help=True)
    parser.add_argument('-j','--json',default='./results/RSICD_DenseNet169_test_withpath_process.json',help='json config file')
    args = parser.parse_args()
    #print(args.json)
    visualize(args.json)
    html()
