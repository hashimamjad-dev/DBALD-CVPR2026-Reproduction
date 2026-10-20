"""Verify source bytes and inspect fixed masks before generating evaluation outputs."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from PIL import Image,ImageDraw
from lane_repro.distillation import conditions
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results/distillation/v1'
rows=json.loads((ROOT/'datasets/distillation/manifest.json').read_text())['samples']; stats=[]
gallery=Image.new('RGB',(1024,4*280),'white'); draw=ImageDraw.Draw(gallery); index=0
for r in rows:
    path=ROOT/r['root']/r['raw_file']; assert hashlib.sha256(path.read_bytes()).hexdigest()==r['image_sha256']
    c=conditions(r,ROOT)
    stats.append(dict(id=r['id'],mask_pixels=int((c['mask']>0).sum()),crop_shape=list(c['crop'].shape)))
    if r['partition']=='evaluation':
        im=c['crop'].copy(); valid=c['mask']>0
        im[valid]=(im[valid]*.6+[100,0,0]).clip(0,255)
        x=(index%4)*256; y=(index//4)*280
        gallery.paste(Image.fromarray(im).resize((256,256)),(x,y+24)); draw.text((x+3,y+5),r['id']+' '+r['dataset'],fill='black'); index+=1
(OUT/'input_validation.json').write_text(json.dumps(stats,indent=2)); gallery.save(OUT/'validated_input_gallery.png')
(OUT/'mask_protocol.json').write_text(json.dumps(dict(tusimple_yx=[352,448,192,320],culane_yx=[312,328,256,352],
    reason='Input-only camera/hood inspection before any held-out generation; no detector feedback',
    lane_protection='Ground-truth lane raster 5px, dilated with 25x25 kernel; oracle condition',
    selected_before_evaluation=True),indent=2))
print(len(stats),'source hashes and mask areas verified')
