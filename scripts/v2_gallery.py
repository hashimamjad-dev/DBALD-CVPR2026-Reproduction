"""Scientific contact sheets of saved outputs; no image generation or retouching."""
import argparse
from v2_common import *
p=argparse.ArgumentParser(); p.add_argument('--final',action='store_true'); p.add_argument('--composite',action='store_true'); args=p.parse_args()
m=json.loads((OUT/'manifest.json').read_text()); lock=json.loads((OUT/'selection.json').read_text()) if (OUT/'selection.json').exists() else None
partition='evaluation' if args.final else 'validation'; records=m['test'] if args.final else m['validation']
seeds=(100,101) if args.final else (0,17)
from PIL import ImageDraw
for ds in ('TuSimple','CULane'):
    cases=[(r,s) for r in records if r['dataset']==ds for s in seeds]
    for page in range((len(cases)+7)//8):
        subset=cases[page*8:page*8+8]; canvas=Image.new('RGB',(840,150*len(subset)+30),'white'); draw=ImageDraw.Draw(canvas)
        for j,title in enumerate(('Reference 20 steps','Teacher 4 steps','Original student','Selected student')): draw.text((j*210+5,8),title,fill='black')
        for i,(r,seed) in enumerate(subset):
            c=conditions(r,ROOT); yy,xx=np.where(c['mask']>0); box=(max(0,int(xx.min())-16),max(0,int(yy.min())-16),min(512,int(xx.max())+17),min(512,int(yy.max())+17))
            base=V1/'evaluation'/r['id'] if args.final else OUT/'validation'/r['id']
            names=['teacher20','teacher4','student4' if args.final else 'student_v1']
            kind='composite' if args.composite else 'raw'
            paths=[base/f'{name}_{seed}_{kind}.png' for name in names]
            if args.final: paths.append(OUT/partition/r['id']/f'student_v2_{seed}_{kind}.png')
            elif lock and lock['selected']['method']!='student_v1': paths.append(base/f"{lock['selected']['method']}_{lock['selected']['step']}_{seed}_{kind}.png")
            else: paths.append(paths[2])
            if args.composite:
                x,y=c['crop_xy']; box=(box[0]+x,box[1]+y,box[2]+x,box[3]+y)
            for j,path in enumerate(paths):
                crop=Image.open(path).crop(box); crop.thumbnail((200,120)); canvas.paste(crop,(j*210+5,150*i+50))
                draw.text((j*210+5,150*i+32),f"{r['id']} seed {seed}",fill='black')
        suffix='_composite' if args.composite else ''
        dest=OUT/'figures'/f'{partition}_{ds}_{page}{suffix}.png'; dest.parent.mkdir(exist_ok=True); canvas.save(dest)
        print(dest)
