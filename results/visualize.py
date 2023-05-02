import json
import pandas as pd
import argparse
import matplotlib.pyplot as plt


def fig2html(fig):
    import base64, io
    b = io.BytesIO()
    fig.savefig(b, format='png')
    b64 = base64.b64encode(b.getvalue()).decode('utf-8')
    return f'<img src=\'data:image/png;base64,{b64}\'>'


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
    print(df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='visualize data',add_help=True)
    parser.add_argument('-j','--json',default='RSICD_DenseNet169_test_withpath_process.json',help='json config file')
    args = parser.parse_args()
    print(args.json)
    visualize(args.json)
